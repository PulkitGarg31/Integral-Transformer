import importlib.util
import pytest

torch_available = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not torch_available, reason="torch not installed locally")


def test_integration_kernel_shapes():
    import torch
    from src.models import LearnedIntegrationKernel
    kernel = LearnedIntegrationKernel(hidden_size=32, kernel_size=8, num_heads=4, max_chunks=6)
    x = torch.randn(2, 6, 32)
    out = kernel(x, mask=torch.ones(2, 6))
    assert out.shape == (2, 6, 32)


def test_integration_kernel_dropout_zero_is_identity_and_default():
    """dropout=0.0 must reproduce the v2 kernel exactly (reversibility knob),
    and omitting the kwarg must mean 0.0 so old call sites are unchanged."""
    import torch
    from src.models import LearnedIntegrationKernel
    torch.manual_seed(0)
    k0 = LearnedIntegrationKernel(hidden_size=32, kernel_size=8, num_heads=4, max_chunks=6)
    k1 = LearnedIntegrationKernel(hidden_size=32, kernel_size=8, num_heads=4, max_chunks=6,
                                  dropout=0.0)
    k1.load_state_dict(k0.state_dict())      # no new params: state dicts are compatible
    assert k0.weight_dropout.p == 0.0 and k0.output_dropout.p == 0.0
    x, m = torch.randn(2, 6, 32), torch.ones(2, 6)
    k0.train(); k1.train()                   # train mode: dropout would fire if p > 0
    assert torch.equal(k0(x, mask=m), k1(x, mask=m))


def test_integration_kernel_dropout_active_only_in_train_mode():
    import torch
    from src.models import LearnedIntegrationKernel
    torch.manual_seed(0)
    k = LearnedIntegrationKernel(hidden_size=32, kernel_size=8, num_heads=4, max_chunks=6,
                                 dropout=0.5)
    x, m = torch.randn(2, 6, 32), torch.ones(2, 6)
    k.eval()
    assert torch.equal(k(x, mask=m), k(x, mask=m))          # deterministic at eval
    k.train()
    assert not torch.equal(k(x, mask=m), k(x, mask=m))      # stochastic in training


def test_adaptive_gate_range():
    import torch
    from src.models import AdaptiveGate
    gate = AdaptiveGate(hidden_size=32)
    g = gate(torch.randn(2, 6, 32), torch.randn(2, 6, 32))
    assert g.shape == (2, 6, 32)
    assert (g >= 0).all() and (g <= 1).all()


def test_registry_has_all_four():
    from src.models import MODEL_REGISTRY
    assert set(MODEL_REGISTRY) == {'integral', 'baseline', 'truncation', 'meanpool'}


def test_model_names_derived_from_registry():
    from src.models import MODEL_NAMES, MODEL_REGISTRY
    assert MODEL_NAMES == list(MODEL_REGISTRY)


def test_masked_max_ignores_masked_slots_even_when_all_values_negative():
    import torch
    from src.models import masked_max
    x = torch.tensor([[[-1., -2.], [-3., -4.], [0., 0.]]])   # slot 3 is padding
    mask = torch.tensor([[1., 1., 0.]])
    out = masked_max(x, mask)
    assert torch.allclose(out, torch.tensor([[-1., -2.]]))   # 0-slot must NOT win


def test_encode_chunks_runs_encoder_only_on_real_chunks():
    import torch
    import torch.nn as nn
    from src.models import encode_chunks

    seen = []

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(100, 16)

        def forward(self, input_ids=None, attention_mask=None):
            seen.append(tuple(input_ids.shape))
            class Out: pass
            o = Out(); o.last_hidden_state = self.emb(input_ids)
            return o

    ids = torch.randint(1, 100, (2, 4, 8))
    attn = torch.ones(2, 4, 8, dtype=torch.long)
    cmask = torch.tensor([[1., 1., 0., 0.], [1., 0., 0., 0.]])
    repr_ = encode_chunks(FakeEncoder(), ids, attn, cmask)
    assert repr_.shape == (2, 4, 16)
    assert seen == [(3, 8)]                          # only the 3 real chunks encoded
    assert repr_[0, 2].abs().sum() == 0              # padding slots stay zero
    assert repr_[1, 1].abs().sum() == 0


# ---------------------------------------------------------------------------
# chunk_mask contract: masked chunk slots must never influence the prediction
# ---------------------------------------------------------------------------

class _TinyCfg:
    pretrained_model = 'fake'
    hidden_size = 32
    num_attention_heads = 4
    num_integration_heads = 4
    integration_kernel_size = 8
    num_classes = 2
    dropout = 0.0
    max_length = 8
    max_chunks = 6
    integration_layers = 1


def _patch_fake_encoder(monkeypatch):
    import torch.nn as nn

    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(1000, 32)

        def forward(self, input_ids=None, attention_mask=None):
            class Out: pass
            o = Out(); o.last_hidden_state = self.emb(input_ids)
            return o

        def gradient_checkpointing_enable(self):
            pass

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, **kw):
            return FakeEncoder()

    monkeypatch.setattr('src.models.AutoModel', FakeAuto)


def _masked_slot_inputs():
    import torch
    torch.manual_seed(0)
    ids = torch.randint(1, 1000, (1, 6, 8))
    attn = torch.ones(1, 6, 8, dtype=torch.long)
    cmask = torch.zeros(1, 6); cmask[0, :2] = 1      # only 2 real chunks
    ids_b = ids.clone()
    ids_b[0, 2:] = torch.randint(1, 1000, (4, 8))    # garbage in MASKED slots only
    return ids, ids_b, attn, cmask


import pytest as _pytest


@_pytest.mark.parametrize("name", ['baseline', 'integral', 'meanpool', 'truncation'])
def test_masked_chunks_cannot_influence_logits(name, monkeypatch):
    import torch
    from src.models import MODEL_REGISTRY
    _patch_fake_encoder(monkeypatch)
    torch.manual_seed(1)
    model = MODEL_REGISTRY[name](_TinyCfg()).eval()
    ids_a, ids_b, attn, cmask = _masked_slot_inputs()
    with torch.no_grad():
        out_a = model(ids_a, attn, cmask)
        out_b = model(ids_b, attn, cmask)
    assert out_a.shape == (1, 2)
    assert torch.allclose(out_a, out_b, atol=1e-6), \
        f"{name}: masked chunk content leaked into the prediction"


@_pytest.mark.parametrize("name", ['baseline', 'integral', 'meanpool', 'truncation'])
def test_padding_slot_count_does_not_change_logits(name, monkeypatch):
    """Same document, padded to 4 vs 6 chunk slots -> identical prediction.
    Fails when padding slots feed the attention layers or the pooling."""
    import torch
    from src.models import MODEL_REGISTRY
    _patch_fake_encoder(monkeypatch)
    torch.manual_seed(2)
    model = MODEL_REGISTRY[name](_TinyCfg()).eval()
    ids, _, attn, cmask = _masked_slot_inputs()      # 2 real chunks of 6 slots
    with torch.no_grad():
        out6 = model(ids, attn, cmask)
        out4 = model(ids[:, :4], attn[:, :4], cmask[:, :4])
    assert torch.allclose(out6, out4, atol=1e-6), \
        f"{name}: number of padding slots changed the prediction"


def test_load_encoder_falls_back_to_download(monkeypatch):
    calls = []

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, **kw):
            calls.append(kw.get('local_files_only', False))
            if kw.get('local_files_only'):
                raise OSError("not in local cache")
            return "downloaded-encoder"

    monkeypatch.setattr('src.models.AutoModel', FakeAuto)
    from src.models import load_encoder
    assert load_encoder('some/model') == "downloaded-encoder"
    assert calls == [True, False]                    # local first, then download


def test_load_tokenizer_falls_back_to_download(monkeypatch):
    calls = []

    class FakeAuto:
        @staticmethod
        def from_pretrained(name, **kw):
            calls.append(kw.get('local_files_only', False))
            if kw.get('local_files_only'):
                raise OSError("not in local cache")
            return "downloaded-tokenizer"

    monkeypatch.setattr('src.models.AutoTokenizer', FakeAuto)
    from src.models import load_tokenizer
    assert load_tokenizer('some/model') == "downloaded-tokenizer"
    assert calls == [True, False]
