from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Safely reuse original components
from multilingual_transformer.data.dataset import (
    PadCollator,
    TranslationDataset,
    is_clean_pair,
)


class MultilingualDataPipeline:
    def __init__(
        self,
        dataset_map: dict[str, str],
        data_dir: Path,
        max_len: int,
        max_ratio: float,
        seed: int,
    ) -> None:
        self.dataset_map = dataset_map
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.seed = seed

    def _extract_pair(self, row: dict[str, Any], ds_name: str, tgt_lang: str) -> tuple[str, str]:
        if ds_name == "acomquest/Saamayik":
            rec = row.get("translation", row)
            return str(rec.get("en", "")).strip(), str(rec.get("sa", "")).strip()
        return str(row["src"]).strip(), str(row["tgt"]).strip()

    def acquire_multilingual_corpus(
        self, pairs_per_lang: int, val_per_lang: int, test_per_lang: int
    ) -> dict[str, Any]:
        """Loads English-Indic pairs and duplicates them bidirectionally with <2tgt> tags."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        train_src, train_tgt = [], []
        eval_splits: dict[str, dict[str, list[str]]] = {}

        for tgt_lang, ds_name in self.dataset_map.items():
            print(f"Loading {tgt_lang.upper()} pairs from {ds_name}...")
            total_needed = pairs_per_lang + val_per_lang + test_per_lang

            args = (ds_name, tgt_lang) if ds_name != "acomquest/Saamayik" else (ds_name,)
            try:
                ds = load_dataset(*args, split="train")
            except Exception:
                ds = load_dataset(*args, split="train")
            ds = ds.shuffle(seed=self.seed)

            cur_en, cur_tgt = [], []
            seen = set()
            with tqdm(total=total_needed, desc=f"Extracting {tgt_lang.upper()}") as pbar:
                for row in ds:
                    en, tg = self._extract_pair(row, ds_name, tgt_lang)
                    key = en.lower()
                    if key in seen or not is_clean_pair(en, tg, self.max_len, self.max_ratio):
                        continue
                    seen.add(key)
                    cur_en.append(en)
                    cur_tgt.append(tg)
                    pbar.update(1)
                    if len(cur_en) >= total_needed:
                        break

            tr_e, tr_t = cur_en[:pairs_per_lang], cur_tgt[:pairs_per_lang]
            va_e, va_t = cur_en[pairs_per_lang:pairs_per_lang + val_per_lang], cur_tgt[pairs_per_lang:pairs_per_lang + val_per_lang]
            te_e, te_t = cur_en[pairs_per_lang + val_per_lang:], cur_tgt[pairs_per_lang + val_per_lang:]

            # Bidirectional augmentation
            train_src.extend([f"<2{tgt_lang}> {s}" for s in tr_e])
            train_tgt.extend(tr_t)
            train_src.extend([f"<2en> {s}" for s in tr_t])
            train_tgt.extend(tr_e)

            # Store directional evaluation sets
            eval_splits[f"en-{tgt_lang}"] = {
                "val_src": [f"<2{tgt_lang}> {s}" for s in va_e], "val_tgt": va_t,
                "test_src": [f"<2{tgt_lang}> {s}" for s in te_e], "test_tgt": te_t,
            }
            eval_splits[f"{tgt_lang}-en"] = {
                "val_src": [f"<2en> {s}" for s in va_t], "val_tgt": va_e,
                "test_src": [f"<2en> {s}" for s in te_t], "test_tgt": te_e,
            }

        # Global val set combining all directions for Trainer
        combined_val_src, combined_val_tgt = [], []
        for pair_data in eval_splits.values():
            combined_val_src.extend(pair_data["val_src"])
            combined_val_tgt.extend(pair_data["val_tgt"])

        return {
            "train_src": train_src,
            "train_tgt": train_tgt,
            "val_src": combined_val_src,
            "val_tgt": combined_val_tgt,
            "eval_splits": eval_splits,
        }

    @staticmethod
    def create_loader(dataset: TranslationDataset, batch_size: int, shuffle: bool) -> DataLoader:
        cpu_workers = min(4, os.cpu_count() or 2)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            pin_memory=torch.cuda.is_available(), num_workers=cpu_workers,
            prefetch_factor=2 if cpu_workers > 0 else None,
            persistent_workers=True if cpu_workers > 0 else False,
            collate_fn=PadCollator(dataset.pad_src, dataset.pad_tgt),
        )