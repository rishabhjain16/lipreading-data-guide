#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TCD-TIMIT Dataset - Text Transform with SentencePiece Tokenization

Uses the shared repo-wide SentencePiece model under `spm/unigram/`.
Important: the shared model vocabulary is uppercase, so transcripts are uppercased
before encoding.

This mirrors `GRID/preparation/transforms.py`.
"""

import os
import sentencepiece
import torch


# Path to shared SPM model and vocabulary
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SP_MODEL_PATH = os.path.join(_REPO_ROOT, "spm", "unigram", "unigram5000.model")
DICT_PATH = os.path.join(_REPO_ROOT, "spm", "unigram", "unigram5000_units.txt")


class TextTransform:
    """SentencePiece tokenizer wrapper."""

    def __init__(self, sp_model_path: str = SP_MODEL_PATH, dict_path: str = DICT_PATH):
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        units = open(dict_path, encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[1] for unit in units}
        # 0 is reserved for blank in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text: str):
        text = (text or "").upper()
        tokens = self.spm.EncodeAsPieces(text)
        token_ids = [self.hashmap.get(token, self.hashmap["<unk>"]) for token in tokens]
        return torch.tensor(list(map(int, token_ids)))

    def post_process(self, token_ids: torch.Tensor):
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")
