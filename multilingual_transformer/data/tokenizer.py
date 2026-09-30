from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]


class BasicVocabTokenizer:
    def __init__(self, vocab: dict[str, int], split_type: str = "whitespace") -> None:
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.split_type = split_type

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab["<unk>"])

    def encode(self, text: str) -> Any:
        tokens = (
            re.findall(r"\w+|[^\w\s]", text.lower())
            if self.split_type == "regex"
            else text.lower().strip().split()
        )

        class Encoding:
            def __init__(self, ids: list[int]):
                self.ids = ids

        return Encoding([self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens])

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = [
            self.id_to_token.get(i, "<unk>")
            for i in ids
            if not (skip_special_tokens and self.id_to_token.get(i) in SPECIAL_TOKENS)
        ]
        return " ".join(tokens)


class TokenizerManager:
    @staticmethod
    def train(sentences: list[str], algo: str, vocab_size: int = 15000) -> Any:
        algo = algo.lower()
        if algo in ["whitespace", "regex"]:
            counter: Counter[str] = Counter()
            for text in sentences:
                tokens = (
                    re.findall(r"\w+|[^\w\s]", text.lower())
                    if algo == "regex"
                    else text.lower().strip().split()
                )
                counter.update(tokens)

            vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
            idx = len(SPECIAL_TOKENS)
            for token, _ in counter.most_common(vocab_size - len(SPECIAL_TOKENS)):
                vocab[token] = idx
                idx += 1
            return BasicVocabTokenizer(vocab, split_type=algo)

        if algo == "bpe":
            tok = Tokenizer(models.BPE(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()]
            )
            tok.decoder = decoders.Metaspace()
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        elif algo == "unigram":
            tok = Tokenizer(models.Unigram())
            tok.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()]
            )
            tok.decoder = decoders.Metaspace()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, unk_token="<unk>"
            )
        elif algo == "wordpiece":
            tok = Tokenizer(models.WordPiece(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            tok.decoder = decoders.WordPiece()
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS
            )
        else:
            raise ValueError(f"Unsupported tokenizer algorithm: {algo}")

        tok.train_from_iterator(sentences, trainer)
        return tok

    @staticmethod
    def save(tokenizer: Any, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(tokenizer, Tokenizer):
            tokenizer.save(str(save_path))

    @staticmethod
    def load(load_path: Path) -> Tokenizer:
        return Tokenizer.from_file(str(load_path))