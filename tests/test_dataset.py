import importlib.util
import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch not installed locally")


def _fake_item(num_real, max_chunks=6, seq_len=8):
    import torch
    input_ids = torch.zeros(max_chunks, seq_len, dtype=torch.long)
    attention_mask = torch.zeros(max_chunks, seq_len, dtype=torch.long)
    chunk_mask = torch.zeros(max_chunks)
    for c in range(num_real):
        input_ids[c] = c + 1          # chunk c filled with value c+1
        attention_mask[c] = 1
        chunk_mask[c] = 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask,
            'chunk_mask': chunk_mask, 'labels': 1}


class _FakeDataset:
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


def test_chunk_removed_zeroes_last_real_chunks():
    from src.dataset import ChunkRemovedDataset
    ds = ChunkRemovedDataset(_FakeDataset([_fake_item(num_real=4)]), n_remove=2)
    item = ds[0]
    assert item['chunk_mask'].sum().item() == 2
    assert item['input_ids'][3].sum().item() == 0   # last real chunk zeroed
    assert item['input_ids'][2].sum().item() == 0   # second-to-last zeroed
    assert item['input_ids'][1].sum().item() > 0    # earlier chunks intact


def test_chunk_removed_skips_too_short_docs():
    from src.dataset import ChunkRemovedDataset
    ds = ChunkRemovedDataset(_FakeDataset([_fake_item(num_real=1)]), n_remove=1)
    assert len(ds) == 0


def test_chunk_removed_does_not_mutate_original():
    from src.dataset import ChunkRemovedDataset
    items = [_fake_item(num_real=4)]
    ds = ChunkRemovedDataset(_FakeDataset(items), n_remove=1)
    _ = ds[0]; _ = ds[0]                                # access twice
    assert items[0]['input_ids'][3].sum().item() > 0    # original untouched
    assert items[0]['chunk_mask'].sum().item() == 4


def _fingerprint(seed):
    from src.dataset import dataset_fingerprint
    from src.config import Config
    from src.preprocessing import DataController
    ctrl = DataController(train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=seed)
    return dataset_fingerprint(Config(), ctrl)


def test_cache_names_are_versioned_and_fingerprinted():
    from src.dataset import sc_cache_name, delhi_cache_name, CACHE_VERSION
    # v4: v3 caches were built with a tokenizer whose vocab never loaded
    # (every word -> UNK), so v3 files must never be picked up again.
    assert CACHE_VERSION == 4
    fp = _fingerprint(seed=16911)
    assert sc_cache_name(55619, fp) == f"dataset_cache_v4_55619_docs_{fp}.pt"
    assert delhi_cache_name(94257, fp) == f"delhi_dataset_cache_v4_94257_docs_{fp}.pt"


class _AllUnkTokenizer:
    """Mimics the 2026-08-14 failure: vocab never loaded, every word -> UNK(1),
    ALBERT-style cls=2/sep=3. Produced caches where all documents were the
    identical token sequence and every model collapsed to the class prior."""
    unk_token_id = 1

    def __call__(self, text, **kw):
        return {"input_ids": [2] + [1] * len(text.split()) + [3]}


class _HealthyTokenizer:
    unk_token_id = 100

    def __call__(self, text, **kw):
        n = len(text.split())
        return {"input_ids": [101] + list(range(2000, 2000 + n)) + [102]}


def test_verify_tokenizer_rejects_all_unk_tokenizer():
    from src.dataset import verify_tokenizer
    with pytest.raises(RuntimeError, match="UNK"):
        verify_tokenizer(_AllUnkTokenizer())


def test_verify_tokenizer_accepts_healthy_tokenizer():
    from src.dataset import verify_tokenizer
    verify_tokenizer(_HealthyTokenizer())          # must not raise


def test_preprocessed_dataset_refuses_broken_tokenizer():
    from src.config import Config
    from src.dataset import PreprocessedLegalDataset
    with pytest.raises(RuntimeError, match="UNK"):
        PreprocessedLegalDataset(["the appeal is dismissed"], [0],
                                 _AllUnkTokenizer(), Config())


def test_fingerprint_depends_on_seed_and_config():
    from src.dataset import dataset_fingerprint
    from src.config import Config
    from src.preprocessing import DataController

    assert _fingerprint(seed=16911) == _fingerprint(seed=16911)   # deterministic
    assert _fingerprint(seed=16911) != _fingerprint(seed=42)      # seed changes split

    small = Config(); small.max_chunks = 12
    ctrl = DataController(train_frac=0.8, val_frac=0.1, test_frac=0.1, seed=16911)
    assert dataset_fingerprint(small, ctrl) != dataset_fingerprint(Config(), ctrl)

    frac = DataController(train_frac=0.7, val_frac=0.2, test_frac=0.1, seed=16911)
    assert dataset_fingerprint(Config(), frac) != dataset_fingerprint(Config(), ctrl)
