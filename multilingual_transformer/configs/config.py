from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

DEFAULT_LANGUAGE_CONFIG = Path(__file__).resolve().parent / "language_config.toml"
T = TypeVar("T")


def _build(cls: type[T], raw: dict[str, Any]) -> T:
    """Instantiate a dataclass from a dict, warning about (and ignoring) unknown keys."""
    valid = {f.name for f in fields(cls) if f.init}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - valid)
    if unknown:
        warnings.warn(f"Ignoring unknown config keys for {cls.__name__}: {unknown}")
    return cls(**{k: v for k, v in raw.items() if k in valid})  # type: ignore[call-arg]


@dataclass
class ProjectConfig:
    name: str = "multilingual_transformer"
    seed: int = 42


@dataclass
class LanguageConfig:
    # User-facing keys
    src_lang: str = "English"
    target_lang: str = "Hindi"
    dataset_name: str = "ai4bharat/samanantar"
    # Derived from language_config.toml (not read from the main toml)
    src_code: str = field(init=False, default="en")
    tgt_lang: str = field(init=False, default="hi")  # target language *code*
    lang_pair: str = field(init=False, default="EN-HI")

    def resolve(self, codes: dict[str, str]) -> None:
        lookup = {k.lower(): v for k, v in codes.items()}
        for name in (self.src_lang, self.target_lang):
            if name.lower() not in lookup:
                raise ValueError(f"Unknown language '{name}'. Supported: {', '.join(codes)}")
        self.src_code = lookup[self.src_lang.lower()]
        self.tgt_lang = lookup[self.target_lang.lower()]
        self.lang_pair = f"{self.src_code}-{self.tgt_lang}".upper()

    @property
    def tag(self) -> str:
        return self.lang_pair.lower().replace("-", "_")


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    tokenizer_dir: Path = Path("tokenizers")
    train_size: int = 50000
    test_size: int = 10000
    max_len: int = 80
    max_length_ratio: float = 6.0
    force_download: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.tokenizer_dir = Path(self.tokenizer_dir)


@dataclass
class TokenizerConfig:
    algo_src: str = "wordpiece"
    algo_tgt: str = "unigram"
    max_vocab_size: int = 22400


@dataclass
class ModelConfig:
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 224
    lr: float = 5.0e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    torch_compile: bool = False
    gen_batch_size: int = 128
    bleu_sample: int = 300


@dataclass
class InferenceConfig:
    sample_sentences: list[str] = field(default_factory=list)


@dataclass
class AppConfig:
    project: ProjectConfig
    language: LanguageConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig

    @classmethod
    def from_toml(cls, path: str | Path, language_config: str | Path | None = None) -> AppConfig:
        path = Path(path)
        with path.open("rb") as f:
            raw = tomllib.load(f)

        if language_config is not None:
            lang_path = Path(language_config)
        elif (path.parent / "language_config.toml").exists():
            lang_path = path.parent / "language_config.toml"
        else:
            lang_path = DEFAULT_LANGUAGE_CONFIG
        with lang_path.open("rb") as f:
            codes = tomllib.load(f).get("languages", {})

        language = _build(LanguageConfig, raw.get("language", {}))
        language.resolve(codes)

        return cls(
            project=_build(ProjectConfig, raw.get("project", {})),
            language=language,
            data=_build(DataConfig, raw.get("data", {})),
            tokenizer=_build(TokenizerConfig, raw.get("tokenizer", {})),
            model=_build(ModelConfig, raw.get("model", {})),
            training=_build(TrainingConfig, raw.get("training", {})),
            inference=_build(InferenceConfig, raw.get("inference", {})),
        )

    # ---- shared artifact paths (single source of truth) ----
    def checkpoint_path(self) -> Path:
        return self.data.checkpoint_dir / f"transformer_{self.language.tag}_best.pt"

    def tokenizer_path(self, side: str) -> Path:
        algo = self.tokenizer.algo_src if side == "src" else self.tokenizer.algo_tgt
        name = f"{self.language.tag}_{side}_{algo}_{self.tokenizer.max_vocab_size}.json"
        return self.data.tokenizer_dir / name
