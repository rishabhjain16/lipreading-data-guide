#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GRID Dataset - Text Transform with SentencePiece Tokenization
Uses the shared SPM tokenizer from root spm/ folder
"""

import os
import sentencepiece
import torch

# Path to shared SPM model and vocabulary
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
    """Mapping Dictionary Class for SentencePiece tokenization."""

    def __init__(
        self,
        sp_model_path=SP_MODEL_PATH,
        dict_path=DICT_PATH,
    ):
        # Load SentencePiece model
        self.spm = sentencepiece.SentencePieceProcessor(model_file=sp_model_path)

        # Load units and create dictionary
        units = open(dict_path, encoding='utf8').read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[1] for unit in units}
        # 0 will be used for "blank" in CTC
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]
        self.ignore_id = -1

    def tokenize(self, text):
        """Tokenize text to token IDs using SentencePiece."""
        text = text.upper()  # SPM model expects uppercase
        # Use SPM's native encoding to get token IDs directly
        token_ids = self.spm.EncodeAsIds(text)
        return torch.tensor(token_ids)

    def post_process(self, token_ids):
        """Convert token IDs back to text."""
        token_ids = token_ids[token_ids != -1]
        text = self._ids_to_str(token_ids, self.token_list)
        text = text.replace("\u2581", " ").strip()
        return text

    def _ids_to_str(self, token_ids, char_list):
        """Convert token IDs to string using character list."""
        token_as_list = [char_list[idx] for idx in token_ids]
        return "".join(token_as_list).replace("<space>", " ")
