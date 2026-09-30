from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ProjectConfig:
    name: str = "multilingual_transformer"
    seed: int = 42


@dataclass
class LanguageConfig:
    target_language: str = "Hindi"
    dataset_name: str = "ai4bharat/samanantar"
    tgt_lang: str = "hi"
    lang_pair: str = "EN-HI"


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    train_size: int = 50000
    test_size: int = 10000
    max_len: int = 80
    max_length_ratio: float = 6.0
    oversample: int = 6
    force_download: bool = False


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
    def from_yaml(cls, path: str | Path) -> AppConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return cls(
            project=ProjectConfig(**raw.get("project", {})),
            language=LanguageConfig(**raw.get("language", {})),
            data=DataConfig(
                **{
                    **raw.get("data", {}),
                    "data_dir": Path(raw.get("data", {}).get("data_dir", "data")),
                    "checkpoint_dir": Path(raw.get("data", {}).get("checkpoint_dir", "checkpoints")),
                }
            ),
            tokenizer=TokenizerConfig(**raw.get("tokenizer", {})),
            model=ModelConfig(**raw.get("model", {})),
            training=TrainingConfig(**raw.get("training", {})),
            inference=InferenceConfig(**raw.get("inference", {})),
        )