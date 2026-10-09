"""
Faz 2 / 3 — Önceden eğitilmiş ses modelleri için tek yükleyici (D-007, D-008, D-034)

BİLİMSEL AMAÇ
    Altı backbone'u (PANNs CNN10 / CNN14 / CNN14_16k, BEATs iter3+ AS2M, WavLM Base+ / Large) RESMİ kodları,
    RESMİ ağırlıkları ve RESMİ girdi biçimleriyle yüklemek ve hepsinden aynı biçimde katman katman, zaman
    ortalamalı gömme almak. Her model için beklenen değerler configs/model_input_contracts.yaml'da; yükleme
    sırasında assert edilir. Böylece "model ön-eğitimde gördüğü girdiyi mi görüyor?" sorusu kodla cevaplanır.

ÇIKTI SÖZLEŞMESİ
    bb = build_backbone("wavlm_large", model_dir=..., device="cuda")
    E = bb.embed(wav)            # wav: torch.float32 (B, T), modelin SR'sinde, [-1, 1]
    E.shape == (B, bb.n_store, bb.dim)  ·  float32  ·  CPU
        PANNs : n_store = 1          (resmi 'embedding' = fc1 + ReLU)
        SSL   : n_store = L + 1      (katman 0 = ilk transformer bloğunun girdisi; i = i'inci bloğun çıkışı)
    bb.primary_layers               birincil temsilde ortalanan katmanlar (PANNs [0]; SSL 1..L) — D-034 madde 3
    Zaman (BEATs'te zaman × frekans yaması) üzerinden ortalama.

KISA KAYITLAR VE DOLGU (D-015, D-034 madde 1)
    embed(wav)                → dolgusuz; her satır kendi gerçek uzunluğunda (batch'teki satırlar eşit uzunlukta olmalı)
    embed(wav, lengths=n)     → wav sağdan sıfırla doldurulmuş (B, T); n = gerçek örnek sayıları. Dolgu bölgesi:
        PANNs   maske yok (resmi API'de yok; Boll gibi düz sıfır dolgusu — dolgu çerçeveleri havuzlamaya girer)
        BEATs   resmi padding_mask (örnek → fbank çerçevesi → yama; BEATs.forward_padding_mask) + yalnız dolu
                yamalar üzerinden ortalama (resmi fine-tune sınıflandırıcısının yaptığı gibi)
        WavLM   resmi özellik çıkarıcı yalnız gerçek kısımla normalize eder; attention_mask yalnız işlemci
                return_attention_mask=True ise (Large) modele verilir (HF önerisi: Base+'ya verilmez); havuzlama
                yalnız gerçek çerçeveler üzerinden (_get_feat_extract_output_lengths)

ŞEKİLLER
    BEATs kancası: (T_yama, B, C)   WavLM kancası: (B, T_çerçeve, C)   → katman başına zaman ortalaması (B, C)
    → katmanlar yığılır (B, n_store, C). Bütün katmanlarda C aynıdır (768 / 1024); assert edilir.

NEDEN KENDİ KANCALARIMIZ?
    BEATs ara katmanları yalnız tgt_layer verilince döndürüyor (resmi kod). transformers'ta `hidden_states`'in
    son öğesinin anlamı sürüme göre değişebiliyor (stable-layer-norm modellerinde v4'te son LayerNorm'lu çıktı,
    v5.19'da ham blok çıktısı). encoder.layers[i] üzerindeki forward hook'ları her sürümde ve iki ailede aynı tanımı
    verir. SMK-001 S8, son kancanın (gerekirse son LayerNorm uygulanmış hâlinin) resmi çıktıyla aynı olduğunu doğrular.

SIZINTI NOTU
    Gömme çıkarımı etiket okumaz ve bizim veride hiçbir parametre öğrenmez. eval() modunda BatchNorm AudioSet
    istatistiklerini kullanır; SpecAugment / maskeleme / dropout kapalıdır. Batch-değişmezlik SMK-001 S7'de test edilir.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministik cuBLAS (CUDA başlamadan önce)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CONTRACTS = REPO / "configs" / "model_input_contracts.yaml"
THIRD = REPO / "third_party"
BACKBONES = {"cnn10": "panns_cnn10", "cnn14": "panns_cnn14", "cnn14_16k": "panns_cnn14_16k",
             "beats": "beats", "wavlm_base_plus": "wavlm_base_plus", "wavlm_large": "wavlm_large"}
VENDORED_SHA256 = {
    "panns/models.py": "7f9af440395ace5160bbb51d654a0dc35fb887fbf5edecb12da61ff6efb306d9",
    "panns/pytorch_utils.py": "1464fcfbfc0fe4c55f690f6b39e1c80eeed5de1e7fd1b7fd30334d304de7dbe9",
    "panns/class_labels_indices.csv": "cdd1049833c4b86127c2773ac0d14a2754b6a6d0d1798002ed5c66e699708429",
    "beats/BEATs.py": "27f289db7c56ce26f2ceb50d3719854b91b2dec1c2830d8b1dd8de1bbee19eeb",
    "beats/backbone.py": "31c0378379a7e0f1d1069f9da444fb86890fe1ea078959a2dcd39640cdcadbaa",
    "beats/modules.py": "edeb6b6cd6a784da749f932c3e0783c0bce556fc768d0d23a4d53d4b819eb424",
}

# Yalnız birim testleri için küçük rastgele yapılandırmalar (gerçek deneyde KULLANILMAZ)
SMALL_BEATS = {"input_patch_size": 16, "embed_dim": 32, "encoder_layers": 2, "encoder_embed_dim": 64,
               "encoder_ffn_embed_dim": 128, "encoder_attention_heads": 4, "conv_pos": 16, "conv_pos_groups": 4,
               "relative_position_embedding": True, "num_buckets": 32, "max_distance": 128, "gru_rel_pos": True,
               "deep_norm": True, "layer_norm_first": False}
SMALL_WAVLM = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "intermediate_size": 128,
               "conv_dim": (32,) * 7, "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 4,
               "num_buckets": 32, "max_bucket_distance": 64}


# ----------------------------------------------------------------------------- yardımcılar
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha256_dir(d: Path) -> str:
    """Klasördeki dosyaların (göreli yol + sha256) listesinin sha256'sı — HF save_pretrained klasörleri için."""
    lines = [f"{p.relative_to(d).as_posix()} {sha256_file(p)}" for p in sorted(d.rglob("*")) if p.is_file()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def check_vendored() -> None:
    for rel, exp in VENDORED_SHA256.items():
        got = sha256_file(THIRD / rel)
        assert got == exp, f"third_party/{rel} değişmiş (sha256 {got[:12]} ≠ {exp[:12]}); resmi kopya bozulmamalı"


def load_contract(name: str) -> dict:
    key = BACKBONES[name]
    c = yaml.safe_load(CONTRACTS.read_text())[key]
    for f in ["family", "class", "checkpoint_file", "sample_rate", "n_layers", "embedding_dim"]:
        assert f in c, f"sözleşmede {key}.{f} yok"
    return c


def set_determinism(seed: int = 0) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@contextlib.contextmanager
def _vendored(sub: str, generic: tuple[str, ...]):
    """third_party/<sub>'u geçici olarak sys.path'e ekle; genel adlı modülleri (models, backbone, ...) sonra kaldır."""
    path = str(THIRD / sub)
    saved = {g: sys.modules.pop(g) for g in generic if g in sys.modules}
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path.remove(path)
        for g in generic:
            sys.modules.pop(g, None)
        sys.modules.update(saved)


def _import_panns():
    if "asthma_panns_models" not in sys.modules:
        with _vendored("panns", ("models", "pytorch_utils")):
            sys.modules["asthma_panns_models"] = importlib.import_module("models")
    return sys.modules["asthma_panns_models"]


def _import_beats():
    if "asthma_beats" not in sys.modules:
        with _vendored("beats", ("BEATs", "backbone", "modules")):
            sys.modules["asthma_beats"] = importlib.import_module("BEATs")
    return sys.modules["asthma_beats"]


def torch_load(path: Path) -> tuple[dict, str]:
    """Önce güvenli yükleme (weights_only=True). Eski PANNs checkpoint'leri numpy nesneleri içerebilir; o zaman
    resmi kaynaktan indirilmiş ve sha256'sı kaydedilmiş dosya için weights_only=False'a düşülür ve bu kaydedilir."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only=True"
    except Exception as e:  # pickle güvenlik kısıtı
        print(f"  UYARI: {path.name} weights_only=True ile yüklenemedi ({type(e).__name__}); "
              "resmi kaynaktan indirilmiş dosya için weights_only=False kullanılıyor", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False), "weights_only=False"


# ----------------------------------------------------------------------------- ortak sınıf
class Backbone:
    """Bir modelin etrafında ince sarmalayıcı: yükleme bilgisi + embed()."""

    def __init__(self, name: str, contract: dict, model: nn.Module, device: str, info: dict):
        self.name, self.contract, self.model, self.device, self.info = name, contract, model, device, info
        self.family = contract["family"]
        self.sr = int(contract["sample_rate"])
        self._capture = False
        self._acts: dict[int, torch.Tensor] = {}
        self.model.to(device).eval()

    # -- alt sınıflar doldurur
    n_store: int
    dim: int
    primary_layers: list[int]

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters()))

    @torch.no_grad()
    def embed(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """wav (B, T) float32 → (B, n_store, dim) float32 CPU. lengths verilirse wav sıfır dolguludur (yukarıya bakın)."""
        assert wav.ndim == 2 and wav.dtype == torch.float32, f"girdi (B, T) float32 olmalı: {tuple(wav.shape)} {wav.dtype}"
        assert not self.model.training, "embed() yalnız eval modunda"
        if lengths is not None:
            lengths = torch.as_tensor(lengths, dtype=torch.long)
            assert lengths.shape == (wav.shape[0],) and (lengths > 0).all() and (lengths <= wav.shape[1]).all()
            pad = torch.arange(wav.shape[1])[None, :] >= lengths[:, None]
            assert (wav[pad] == 0).all(), "dolgu bölgesi sıfır değil"
        out = self._embed(wav.to(self.device), lengths)
        assert out.shape == (wav.shape[0], self.n_store, self.dim), f"{self.name}: çıktı {tuple(out.shape)}"
        assert torch.isfinite(out).all(), f"{self.name}: sonlu olmayan gömme"
        return out.float().cpu()

    def _embed(self, wav, lengths):  # pragma: no cover
        raise NotImplementedError

    @staticmethod
    def _masked_mean(a: torch.Tensor, valid: torch.Tensor | None, time_dim: int) -> torch.Tensor:
        """a: (T, B, C) (time_dim=0) ya da (B, T, C) (time_dim=1); valid: (B, T) bool ya da None → (B, C)."""
        if valid is None:
            return a.mean(time_dim)
        w = valid.T[..., None] if time_dim == 0 else valid[..., None]
        w = w.to(a.dtype)
        assert w.shape[time_dim] == a.shape[time_dim], f"maske uzunluğu {w.shape} ≠ aktivasyon {a.shape}"
        return (a * w).sum(time_dim) / w.sum(time_dim).clamp(min=1)

    def _stack(self, pooled: list[torch.Tensor]) -> torch.Tensor:
        dims = {t.shape[-1] for t in pooled}
        assert dims == {self.dim}, f"{self.name}: katmanlar farklı boyutta {dims}"
        return torch.stack(pooled, dim=1)

    # Faz 3 bellek ölçümü için: eğitim modunda tek bir havuzlanmış vektör (augmentation kapalı, D-016)
    def train_features(self, wav: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError

    def disable_augmentation(self) -> None:
        pass

    def official_last(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        """S8: (bizim son katman kancamız, modelin resmi son çıktısı) — ikisi de zaman ortalamalı, aynı tanımla."""
        raise NotImplementedError


# ----------------------------------------------------------------------------- PANNs
class PannsBackbone(Backbone):
    def __init__(self, name, contract, model, device, info):
        super().__init__(name, contract, model, device, info)
        self.n_store, self.dim, self.primary_layers = 1, int(model.fc1.out_features), [0]

    def _embed(self, wav, lengths):
        # lengths yok sayılır: PANNs'in resmi forward'ında maske yok; sıfır dolgusu (varsa) Boll'daki gibi girdiye dahil
        return self.model(wav)["embedding"][:, None, :]

    @torch.no_grad()
    def clipwise(self, wav: torch.Tensor) -> torch.Tensor:
        assert not self.model.training
        return self.model(wav.to(self.device))["clipwise_output"].float().cpu()

    def disable_augmentation(self) -> None:
        self.model.spec_augmenter = nn.Identity()   # D-016: model içi SpecAugment kapalı

    def train_features(self, wav):
        return self.model(wav)["embedding"]


def _build_panns(name, contract, model_dir, device, random_init):
    m_mod = _import_panns()
    cls = getattr(m_mod, contract["class"])
    model = cls(**contract["constructor"])
    info = {"class": contract["class"], "constructor": contract["constructor"]}
    c = contract["constructor"]
    assert model.bn0.num_features == c["mel_bins"]
    assert tuple(model.logmel_extractor.melW.shape) == (c["window_size"] // 2 + 1, c["mel_bins"]), "mel filtre şekli"
    if not random_init:
        path = Path(model_dir) / contract["checkpoint_file"]
        ckpt, how = torch_load(path)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        res = model.load_state_dict(state, strict=True)
        info.update({"checkpoint": str(path.name), "checkpoint_sha256": sha256_file(path), "load": how,
                     "missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)})
    bb = PannsBackbone(name, contract, model, device, info)
    assert bb.dim == contract["embedding_dim"], f"{name}: fc1 boyutu {bb.dim}"
    return bb


# ----------------------------------------------------------------------------- BEATs
class BeatsBackbone(Backbone):
    def __init__(self, name, contract, model, device, info):
        super().__init__(name, contract, model, device, info)
        layers = model.encoder.layers
        self.L = len(layers)
        self.n_store, self.dim, self.primary_layers = self.L + 1, int(model.cfg.encoder_embed_dim), list(range(1, self.L + 1))
        layers[0].register_forward_pre_hook(self._pre_hook)
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._make_hook(i + 1))

    def _pre_hook(self, mod, args):
        if self._capture:
            self._acts[0] = args[0]

    def _make_hook(self, i):
        def hook(mod, args, out):
            if self._capture:
                self._acts[i] = out[0]          # (T, B, C)
        return hook

    def _run(self, wav, lengths=None):
        """Döner: (resmi çıktı x (B, T_yama, C), geçerli yama maskesi (B, T_yama) ya da None)."""
        pm = None
        if lengths is not None:
            pm = torch.arange(wav.shape[1], device=wav.device)[None, :] >= lengths.to(wav.device)[:, None]   # True = dolgu
        self._acts, self._capture = {}, True
        try:
            x, patch_pm = self.model.extract_features(wav, padding_mask=pm)
        finally:
            self._capture = False
        assert len(self._acts) == self.n_store, f"BEATs: {len(self._acts)} katman yakalandı"
        valid = None if patch_pm is None else ~patch_pm
        if valid is not None:
            assert valid.shape == x.shape[:2] and valid.any(1).all(), "her örnekte en az bir geçerli yama olmalı"
        return x, valid

    def _embed(self, wav, lengths):
        _, valid = self._run(wav, lengths)
        return self._stack([self._masked_mean(self._acts[i], valid, 0) for i in range(self.n_store)])

    @torch.no_grad()
    def official_last(self, wav):
        x, _ = self._run(wav.to(self.device))
        return self._acts[self.L].transpose(0, 1).mean(1).cpu(), x.mean(1).cpu()

    def train_features(self, wav):
        x, _ = self.model.extract_features(wav, padding_mask=None)
        return x.mean(1)


def _build_beats(name, contract, model_dir, device, random_init):
    bmod = _import_beats()
    if random_init:
        cfg = bmod.BEATsConfig(SMALL_BEATS)
        model = bmod.BEATs(cfg)
        info = {"cfg": dict(SMALL_BEATS), "random_init": True}
    else:
        path = Path(model_dir) / contract["checkpoint_file"]
        ckpt, how = torch_load(path)
        cfg = bmod.BEATsConfig(ckpt["cfg"])
        for k, v in contract["expect"].items():
            assert getattr(cfg, k) == v, f"BEATs cfg.{k} = {getattr(cfg, k)} ≠ sözleşme {v}"
        model = bmod.BEATs(cfg)
        res = model.load_state_dict(ckpt["model"], strict=True)
        info = {"checkpoint": path.name, "checkpoint_sha256": sha256_file(path), "load": how,
                "cfg": {k: v for k, v in cfg.__dict__.items() if isinstance(v, (int, float, str, bool))},
                "missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)}
    assert model.predictor is None, "fine-tune edilmemiş (predictor'sız) BEATs bekleniyordu"
    bb = BeatsBackbone(name, contract, model, device, info)
    if not random_init:
        assert bb.L == contract["n_layers"] and bb.dim == contract["embedding_dim"]
    return bb


# ----------------------------------------------------------------------------- WavLM
class WavlmBackbone(Backbone):
    def __init__(self, name, contract, model, device, info, fe):
        super().__init__(name, contract, model, device, info)
        self.fe = fe
        layers = model.encoder.layers
        self.L = len(layers)
        self.n_store, self.dim, self.primary_layers = self.L + 1, int(model.config.hidden_size), list(range(1, self.L + 1))
        layers[0].register_forward_pre_hook(self._pre_hook)
        for i, layer in enumerate(layers):
            layer.register_forward_hook(self._make_hook(i + 1))

    def _pre_hook(self, mod, args):
        if self._capture:
            self._acts[0] = args[0]

    def _make_hook(self, i):
        def hook(mod, args, out):
            if self._capture:
                self._acts[i] = out[0] if isinstance(out, tuple) else out   # (B, T, C)
        return hook

    def preprocess(self, wav: torch.Tensor, lengths: torch.Tensor | None = None):
        """Resmi Wav2Vec2FeatureExtractor (do_normalize sözleşmeye göre).
        lengths yok: bütün girdiler aynı uzunlukta, dolgu yok. lengths var: gerçek kısımlar çıkarıcıya verilir; çıkarıcı
        yalnız gerçek kısımla normalize eder ve T'ye kadar sıfırla doldurur (HF'nin batch davranışı). Döner: (iv, attention_mask|None)."""
        T = wav.shape[1]
        if lengths is None:
            arr = [w.detach().cpu().numpy() for w in wav]
            out = self.fe(arr, sampling_rate=self.sr, return_tensors="np", padding=False)
            am = None
        else:
            arr = [w[: int(n)].detach().cpu().numpy() for w, n in zip(wav, lengths)]
            out = self.fe(arr, sampling_rate=self.sr, return_tensors="np", padding="max_length", max_length=T,
                          return_attention_mask=True)
            am = torch.from_numpy(np.asarray(out["attention_mask"], dtype=np.int64)).to(self.device)
        iv = torch.from_numpy(np.asarray(out["input_values"], dtype=np.float32))
        assert iv.shape == wav.shape, f"özellik çıkarıcı uzunluğu değiştirdi: {tuple(iv.shape)}"
        return iv.to(self.device), am

    def _run(self, wav, lengths=None):
        """Döner: (resmi son çıktı (B, T_çerçeve, C), geçerli çerçeve maskesi (B, T_çerçeve) ya da None)."""
        iv, am = self.preprocess(wav, lengths)
        pass_mask = am is not None and bool(getattr(self.fe, "return_attention_mask", False))
        self._acts, self._capture = {}, True
        try:
            out = self.model(iv, attention_mask=am if pass_mask else None)
        finally:
            self._capture = False
        assert len(self._acts) == self.n_store, f"WavLM: {len(self._acts)} katman yakalandı"
        last = out.last_hidden_state
        valid = None
        if lengths is not None:
            n_fr = self.model._get_feat_extract_output_lengths(lengths.to(last.device))
            valid = torch.arange(last.shape[1], device=last.device)[None, :] < n_fr[:, None]
            assert valid.any(1).all()
        return last, valid

    def _embed(self, wav, lengths):
        _, valid = self._run(wav, lengths)
        return self._stack([self._masked_mean(self._acts[i], valid, 1) for i in range(self.n_store)])

    @torch.no_grad()
    def official_last(self, wav):
        last, _ = self._run(wav)
        ours = self._acts[self.L]
        if getattr(self.model.config, "do_stable_layer_norm", False):
            ours = self.model.encoder.layer_norm(ours)    # resmi çıktı = son LayerNorm(son blok)
        return ours.mean(1).cpu(), last.mean(1).cpu()

    def disable_augmentation(self) -> None:
        self.model.config.apply_spec_augment = False      # D-016: zaman/frekans maskeleme kapalı

    def train_features(self, wav):
        return self.model(self.preprocess(wav)[0]).last_hidden_state.mean(1)


def _build_wavlm(name, contract, model_dir, device, random_init):
    from transformers import Wav2Vec2FeatureExtractor, WavLMConfig, WavLMModel
    exp = contract["expect"]
    if random_init:
        stable = name == "wavlm_large"
        cfg = WavLMConfig(**SMALL_WAVLM, do_stable_layer_norm=stable, feat_extract_norm="layer" if stable else "group")
        model = WavLMModel(cfg)
        fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                      do_normalize=exp["do_normalize"], return_attention_mask=stable)
        info = {"random_init": True}
    else:
        path = Path(model_dir) / contract["checkpoint_file"]
        model, li = WavLMModel.from_pretrained(path, output_loading_info=True)
        fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
        assert not li["missing_keys"], f"WavLM eksik anahtarlar: {li['missing_keys'][:5]}"
        assert not li.get("mismatched_keys"), f"WavLM boyut uyuşmazlığı: {li['mismatched_keys'][:5]}"
        c = model.config
        assert c.hidden_size == exp["hidden_size"] and c.num_hidden_layers == exp["num_hidden_layers"], "WavLM boyut/katman"
        assert bool(fe.do_normalize) == exp["do_normalize"], f"do_normalize {fe.do_normalize} ≠ sözleşme {exp['do_normalize']}"
        assert int(fe.sampling_rate) == exp["sampling_rate"]
        assert bool(getattr(fe, "return_attention_mask", False)) == exp["return_attention_mask"], (
            f"return_attention_mask {getattr(fe, 'return_attention_mask', None)} ≠ sözleşme {exp['return_attention_mask']} "
            "(dolgulu girdide maskenin modele verilip verilmeyeceğini belirler; D-034 madde 1)")
        assert int(np.prod(c.conv_stride)) == 320, "özellik kodlayıcı adımı 320 örnek olmalı"
        rev = path / "REVISION.txt"
        info = {"checkpoint": path.name, "checkpoint_sha256": sha256_dir(path), "load": "from_pretrained",
                "hf_revision": rev.read_text().strip() if rev.exists() else "",
                "unexpected_keys": list(li["unexpected_keys"]),
                "config": {k: getattr(c, k) for k in ["do_stable_layer_norm", "feat_extract_norm", "mask_time_prob",
                                                       "mask_feature_prob", "layerdrop", "apply_spec_augment"]},
                "do_normalize": bool(fe.do_normalize),
                "padding_handling": {"fe_return_attention_mask": bool(getattr(fe, "return_attention_mask", False)),
                                     "feat_extract_norm": c.feat_extract_norm,
                                     "mask_passed_to_model_when_padded": bool(getattr(fe, "return_attention_mask", False)),
                                     "pooling": "yalnız gerçek çerçeveler"}}
    return WavlmBackbone(name, contract, model, device, info, fe)


def build_backbone(name: str, model_dir: str | Path | None = None, device: str = "cpu",
                   random_init: bool = False) -> Backbone:
    """random_init=True yalnız birim testleri içindir (ağırlık indirmeden kod yollarını sınar)."""
    assert name in BACKBONES, f"bilinmeyen backbone {name}; seçenekler {list(BACKBONES)}"
    contract = load_contract(name)
    if not random_init:
        assert model_dir is not None, "model_dir gerekli"
    builder = {"panns": _build_panns, "beats": _build_beats, "wavlm": _build_wavlm}[contract["family"]]
    bb = builder(name, contract, model_dir, device, random_init)
    bb.info.update({"backbone": name, "n_params": bb.n_params(), "random_init": random_init,
                    "torch": torch.__version__, "n_store": bb.n_store, "dim": bb.dim,
                    "primary_layers": bb.primary_layers})
    return bb


def window_starts(n: int, win: int, hop: int) -> list[tuple[int, int]]:
    """4 s / 2 s pencereler; son pencere sona hizalı; n <= win ise tüm kayıt tek parça (0, n).
    Kısa parçanın sıfırla doldurulup doldurulmayacağı çağıranın kararıdır (extract_embeddings --short-policy; D-034).
    Döner: [(başlangıç, uzunluk), ...]"""
    assert n > 0 and win > 0 and hop > 0
    if n <= win:
        return [(0, n)]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return [(s, win) for s in starts]
