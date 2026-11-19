"""A tiny, randomly initialised encoder and a word-level tokenizer, built offline.

Used by the smoke tests and by `python -m fivetag.selfcheck`: it lets the E1
training loop run end to end on a CPU in seconds, with no download. It is not a
model of anything — it exists to exercise the code.
"""
from __future__ import annotations


def tiny_tokenizer(texts):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    pre = pre_tokenizers.Sequence([pre_tokenizers.Whitespace()])
    words = sorted({w for t in texts for w, _ in pre.pre_tokenize_str(t.lower())})
    vocab = {tok: i for i, tok in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words)}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    from tokenizers import normalizers
    tk.normalizer = normalizers.Lowercase()
    tk.pre_tokenizer = pre
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")


def tiny_encoder(vocab_size: int, seed: int = 0, hidden: int = 48):
    import torch
    from transformers import BertConfig, BertModel
    torch.manual_seed(seed)
    return BertModel(BertConfig(vocab_size=vocab_size, hidden_size=hidden, num_hidden_layers=2,
                                num_attention_heads=4, intermediate_size=96,
                                max_position_embeddings=128), add_pooling_layer=False)
