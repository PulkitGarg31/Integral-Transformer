"""Tokenized datasets, verdict-hiding chunk removal, and versioned cache naming."""

import hashlib

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.preprocessing import smart_split_chunks

CACHE_VERSION = 4  # v4 = rebuild after 2026-08-14: v3 caches were tokenized by a
                   # tokenizer whose vocab never loaded (every word -> UNK), so
                   # every document was the same token sequence and all models
                   # collapsed to the class prior. v3 files must never be reused.


def verify_tokenizer(tokenizer):
    """Refuse a tokenizer that maps common English words to UNK.

    A tokenizer whose vocab file failed to load still "works": it emits
    [CLS] UNK ... UNK [SEP] for every text, which silently produced the
    corrupt v3 caches. Catch that BEFORE hours of tokenization.
    """
    probe = "the appeal is dismissed by the court"
    ids = list(tokenizer(probe)["input_ids"])
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and sum(1 for i in ids if i == unk) * 2 >= len(ids):
        raise RuntimeError(
            f"Tokenizer maps common English words to UNK (id {unk}): "
            f"{probe!r} -> {ids}. Its vocabulary failed to load — "
            "re-download the tokenizer before building any dataset cache.")


def dataset_fingerprint(config, controller):
    """Hash of everything that changes the tokenized dataset or its split.

    The doc count alone is NOT a valid cache key: a run with a different
    --seed (or split fractions, max_chunks, max_length, tokenizer) would
    silently reuse the old split while logging the new seed.
    """
    key = "|".join(str(v) for v in (
        controller.seed, controller.train_frac, controller.val_frac,
        controller.test_frac, controller.max_files, controller.year_range,
        controller.months, controller.min_text_length,
        config.max_chunks, config.max_length, config.pretrained_model))
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def sc_cache_name(n_docs, fingerprint):
    return f"dataset_cache_v{CACHE_VERSION}_{n_docs}_docs_{fingerprint}.pt"


def delhi_cache_name(n_docs, fingerprint):
    return f"delhi_dataset_cache_v{CACHE_VERSION}_{n_docs}_docs_{fingerprint}.pt"


class PreprocessedLegalDataset(Dataset):
    """
    🔥 OPTIMIZED: All chunking + tokenization done ONCE at init.
    __getitem__ just returns cached tensors — zero CPU overhead during training.
    """

    def __init__(self, texts, labels, tokenizer, config):
        verify_tokenizer(tokenizer)   # broken vocab -> all-UNK cache (see v4 note)
        self.data = []
        print(f"  Pre-tokenizing {len(texts)} documents...")
        for text, label in tqdm(zip(texts, labels), total=len(texts),
                                desc="  Preprocessing", leave=False):
            item = self._preprocess(text, label, tokenizer, config)
            self.data.append(item)
        print(f"  ✓ Pre-tokenization complete ({len(self.data)} samples cached)")

    def _preprocess(self, text, label, tokenizer, config):
        chunks = smart_split_chunks(text, config.max_chunks)
        input_ids_list = []
        attention_mask_list = []
        for chunk in chunks[:config.max_chunks]:
            encoding = tokenizer(
                chunk, max_length=config.max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            input_ids_list.append(encoding['input_ids'].squeeze(0))
            attention_mask_list.append(encoding['attention_mask'].squeeze(0))

        num_chunks = len(input_ids_list)
        while len(input_ids_list) < config.max_chunks:
            input_ids_list.append(torch.zeros(config.max_length, dtype=torch.long))
            attention_mask_list.append(torch.zeros(config.max_length, dtype=torch.long))

        chunk_mask = torch.zeros(config.max_chunks)
        chunk_mask[:num_chunks] = 1

        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'chunk_mask': chunk_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]  # 🔥 Just return cached tensor — zero CPU work!


# ============================================================================
# 🔥 OPTIMIZED DataLoader factory
# ============================================================================

def make_dataloader(dataset, batch_size, shuffle=False):
    """Create a DataLoader with all performance optimizations."""
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,                       # 🔥 Windows Fix: num_workers>0 with huge pre-tokenized RAM causes massive pickling deadlocks
        pin_memory=use_cuda,                 # 🔥 Faster CPU→GPU transfer
    )


class ChunkRemovedDataset(Dataset):
    """
    Wraps a PreprocessedLegalDataset and removes the last N real chunks
    from each sample by zeroing out input_ids, attention_mask, and chunk_mask.

    Documents with <= N real chunks are SKIPPED (removed from dataset).
    """

    def __init__(self, original_dataset, n_remove=1):
        self.original = original_dataset
        self.n_remove = n_remove
        # Only the indices of the kept documents are stored — chunks are zeroed
        # lazily in __getitem__, so RAM is not doubled by an eager deep copy.
        self.indices = []
        skipped = 0
        for idx in range(len(original_dataset)):
            num_real_chunks = int(original_dataset[idx]['chunk_mask'].sum().item())
            if num_real_chunks <= n_remove:
                skipped += 1
            else:
                self.indices.append(idx)

        print(f"  ChunkRemovedDataset: {len(self.indices)} samples retained, "
              f"{skipped} skipped (had <= {n_remove} chunks)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = self.original[self.indices[idx]]
        num_real_chunks = int(item['chunk_mask'].sum().item())

        # Clone tensors so we don't modify the original cached data
        new_item = {
            'input_ids': item['input_ids'].clone(),
            'attention_mask': item['attention_mask'].clone(),
            'chunk_mask': item['chunk_mask'].clone(),
            'labels': item['labels'].clone() if isinstance(item['labels'], torch.Tensor)
                      else torch.tensor(item['labels'], dtype=torch.long),
        }

        # Zero out the last N real chunks
        for i in range(self.n_remove):
            chunk_idx = num_real_chunks - 1 - i
            new_item['input_ids'][chunk_idx] = 0
            new_item['attention_mask'][chunk_idx] = 0
            new_item['chunk_mask'][chunk_idx] = 0

        return new_item
