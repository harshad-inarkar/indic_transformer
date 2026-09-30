from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import torch
from datasets import load_dataset
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def is_clean_pair(en: str, tgt: str, max_len: int, max_ratio: float) -> bool:
    en_toks, tgt_toks = en.split(), tgt.split()
    n_en, n_tgt = len(en_toks), len(tgt_toks)
    if n_en < 3 or n_tgt < 3 or n_en > max_len - 2 or n_tgt > max_len - 2:
        return False
    return (max(n_en, n_tgt) / max(min(n_en, n_tgt), 1)) <= max_ratio


def numericalize(text: str, tokenizer: Any) -> list[int]:
    encoded = tokenizer.encode(text.lower())
    sos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")
    return [sos_id] + encoded.ids + [eos_id]


def pad_sequence(seq: list[int], max_len: int, pad_idx: int) -> Tensor:
    seq = seq[:max_len]
    return torch.tensor(seq + [pad_idx] * (max_len - len(seq)), dtype=torch.long)


class TranslationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: Any,
        tgt_tokenizer: Any,
        max_len: int,
    ) -> None:
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len
        self.pad_src = src_tokenizer.token_to_id("<pad>")
        self.pad_tgt = tgt_tokenizer.token_to_id("<pad>")

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        src_ids = numericalize(self.src_texts[idx], self.src_tok)
        tgt_ids = numericalize(self.tgt_texts[idx], self.tgt_tok)
        return (
            pad_sequence(src_ids, self.max_len, self.pad_src),
            pad_sequence(tgt_ids, self.max_len, self.pad_tgt),
        )


class DataPipeline:
    def __init__(
        self,
        dataset_name: str,
        tgt_lang: str,
        lang_pair: str,
        data_dir: Path,
        max_len: int,
        max_ratio: float,
        oversample: int,
        seed: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.tgt_lang = tgt_lang
        self.lang_pair = lang_pair
        self.data_dir = data_dir
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.oversample = oversample
        self.seed = seed

    def acquire_corpus(self, train_size: int, test_size: int, force_download: bool = False) -> tuple[list[str], list[str], list[str], list[str]]:
        tag = self.lang_pair.lower().replace("-", "_")
        train_path = self.data_dir / f"train_{tag}.jsonl"
        test_path = self.data_dir / f"test_{tag}.jsonl"

        if not force_download and train_path.exists() and test_path.exists():
            return (*self._load_jsonl(train_path), *self._load_jsonl(test_path))

        total = train_size + test_size

        if self.dataset_name in ("cfilt/iitb-english-hindi", "acomquest/Saamayik"):
            ds = load_dataset(self.dataset_name, split="train", streaming=True, trust_remote_code=True)
        else:
            ds = load_dataset(self.dataset_name, self.tgt_lang, split="train", streaming=True, trust_remote_code=True)

        ds = ds.shuffle(seed=self.seed, buffer_size=10_000)

        src, tgt = [], []
        print(f"Streaming and filtering {total:,} random clean pairs from {self.dataset_name}...")

        # Wrap the extraction in a tqdm progress bar
        with tqdm(total=total, desc="Extracting Pairs") as pbar:
            for row in ds:
                if self.dataset_name == "cfilt/iitb-english-hindi":
                    en = row["translation"]["en"].strip()
                    tg = row["translation"]["hi"].strip()
                elif self.dataset_name == "acomquest/Saamayik":
                    rec = row.get("translation", row)
                    en = str(rec.get("en", "")).strip()
                    tg = str(rec.get("sa", "")).strip()
                else:
                    en = row["src"].strip()
                    tg = row["tgt"].strip()

                if is_clean_pair(en, tg, self.max_len, self.max_ratio):
                    src.append(en)
                    tgt.append(tg)
                    pbar.update(1)  # Advance the progress bar by 1
                
                if len(src) >= total:
                    break

        split_idx = int(len(src) * (train_size / total))
        train_src, train_tgt = src[:split_idx], tgt[:split_idx]
        test_src, test_tgt = src[split_idx:], tgt[split_idx:]

        self._save_jsonl(train_path, train_src, train_tgt)
        self._save_jsonl(test_path, test_src, test_tgt)
        return train_src, train_tgt, test_src, test_tgt

    def _save_jsonl(self, path: Path, src: list[str], tgt: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for s, t in zip(src, tgt):
                fh.write(json.dumps({"en": s, "tgt": t}, ensure_ascii=False) + "\n")

    def _load_jsonl(self, path: Path) -> tuple[list[str], list[str]]:
        src, tgt = [], []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    src.append(item["en"])
                    tgt.append(item["tgt"])
        return src, tgt

    @staticmethod
    def create_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        workers = min(4, os.cpu_count() or 2)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=workers,
            persistent_workers=workers > 0,
        )