#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RoomReader - TextTransform using the shared repo SentencePiece tokenizer.

Uses:
- spm/unigram/unigram5000.model
- spm/unigram/unigram5000_units.txt

The shared model vocabulary is uppercase, so we normalize input text with
`.upper()` before encoding.
"""

import os

import sentencepiece
import torch


SP_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "spm",
    "unigram",
    "unigram5000.model",
)

DICT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "spm",
    "unigram",
    "unigram5000_units.txt",
)


class TextTransform:
    """SentencePiece-based text transform."""

    def __init__(self, sp_model_path=SP_MODEL_PATH, dict_path=DICT_PATH):
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)
        units = open(dict_path, encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[1] for unit in units}
        # 0 is CTC blank
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text: str) -> torch.Tensor:
        text = (text or "").upper()
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids: torch.Tensor) -> str:
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")
