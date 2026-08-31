"""Model architectures: Integral Transformer and the three baselines."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def load_encoder(model_name):
    """Load the pretrained encoder from the local HF cache, downloading on a
    fresh machine instead of crashing after hours of preprocessing."""
    try:
        return AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        print(f"  Encoder '{model_name}' not in local cache — downloading...")
        return AutoModel.from_pretrained(model_name)


def load_tokenizer(model_name):
    """Same local-first/download-fallback policy as load_encoder, so the
    tokenizer and encoder can never diverge in offline behavior."""
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        print(f"  Tokenizer '{model_name}' not in local cache — downloading...")
        return AutoTokenizer.from_pretrained(model_name)


def encode_chunks(encoder, input_ids, attention_mask, chunk_mask):
    """Encode ONLY the real chunks (chunk_mask == 1) and mean-pool each chunk's
    tokens; padding chunk slots stay exact zero vectors.

    Running the encoder on padding chunks wasted most of the compute for short
    documents AND let all-masked attention produce garbage representations, so
    zeroed/"removed" chunks could still influence the prediction.
    """
    batch_size, num_chunks, seq_len = input_ids.shape
    flat_ids = input_ids.reshape(batch_size * num_chunks, seq_len)
    flat_attn = attention_mask.reshape(batch_size * num_chunks, seq_len)
    if chunk_mask is None:
        real = torch.ones(batch_size * num_chunks, dtype=torch.bool,
                          device=input_ids.device)
    else:
        real = chunk_mask.reshape(batch_size * num_chunks) > 0
    if not bool(real.any()):
        raise ValueError("encode_chunks: batch contains no real chunks")
    hidden = encoder(input_ids=flat_ids[real],
                     attention_mask=flat_attn[real]).last_hidden_state
    mask_f = flat_attn[real].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
    out = torch.zeros(batch_size * num_chunks, pooled.shape[-1],
                      dtype=pooled.dtype, device=pooled.device)
    out[real] = pooled
    return out.view(batch_size, num_chunks, -1)


def masked_max(x, chunk_mask):
    """Max-pool over chunk slots ignoring masked slots entirely.
    Multiplying by the mask instead lets a masked slot's 0 win the max
    whenever every real value is negative."""
    if chunk_mask is None:
        return x.max(dim=1).values
    fill = torch.finfo(x.dtype).min
    return x.masked_fill(chunk_mask.unsqueeze(-1) == 0, fill).max(dim=1).values


class LearnedIntegrationKernel(nn.Module):
    """
    Learned RBF kernel for dense integration without softmax.
    Improvements v2:
      - Multi-scale: fine + coarse RBF, learned mixture
      - Position bias: later chunks (conclusion) weighted higher
      - Scale clamping: prevents degenerate flat/delta kernels
      - Dropout (v3): same two sites as nn.TransformerEncoderLayer -- on the
        chunk-to-chunk weight matrix and on the output before the residual.
        dropout=0.0 is an exact identity (bit-for-bit the v2 kernel).
    """

    def __init__(self, hidden_size, kernel_size, num_heads=4, max_chunks=16, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.kernel_query = nn.Linear(hidden_size, kernel_size * num_heads)
        self.kernel_key   = nn.Linear(hidden_size, kernel_size * num_heads)
        self.kernel_value = nn.Linear(hidden_size, hidden_size)

        # ✅ Improvement 1: Multi-scale RBF — fine + coarse, learned mix
        self.scale_fine   = nn.Parameter(torch.full((num_heads,), float(kernel_size)))
        self.scale_coarse = nn.Parameter(torch.full((num_heads,), float(kernel_size * 4)))
        self.scale_mix    = nn.Parameter(torch.zeros(num_heads))  # logit; 0 → 50/50

        # ✅ Improvement 2: Position bias — later chunks get higher kernel weight
        # For legal judgments: conclusion is at the end → last chunks matter most
        self.pos_bias = nn.Embedding(max_chunks, num_heads)
        nn.init.zeros_(self.pos_bias.weight)  # start neutral, let model learn

        self.output_proj = nn.Linear(hidden_size, hidden_size)
        self.layer_norm  = nn.LayerNorm(hidden_size)

        # Regularisation parity with the attention pathway: TransformerEncoderLayer
        # drops attention weights and its residual branch; the kernel had neither
        # while learning at integration_lr (25x the encoder LR) -> overfit SC,
        # transfer worse (run-1 ablation: every un-dropped component cost Delhi F1).
        self.weight_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        Q = self.kernel_query(x)
        K = self.kernel_key(x)
        V = self.kernel_value(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.kernel_size).permute(0, 2, 1, 3)
        K = K.view(batch_size, seq_len, self.num_heads, self.kernel_size).permute(0, 2, 1, 3)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        Q_expanded = Q.unsqueeze(3)
        K_expanded = K.unsqueeze(2)
        dist_sq = ((Q_expanded - K_expanded) ** 2).sum(dim=-1)  # (B, H, seq, seq)

        # ✅ Improvement 3: Clamped scales prevent degenerate kernels
        sf = self.scale_fine.clamp(min=1.0).view(1, -1, 1, 1)
        sc = self.scale_coarse.clamp(min=1.0).view(1, -1, 1, 1)
        mix = torch.sigmoid(self.scale_mix).view(1, -1, 1, 1)

        kernel_fine   = torch.exp(-dist_sq / sf)
        kernel_coarse = torch.exp(-dist_sq / sc)
        kernel_weights = mix * kernel_fine + (1.0 - mix) * kernel_coarse

        # ✅ Improvement 2: Add position bias (broadcast over query dim)
        positions = torch.arange(seq_len, device=x.device)
        pb = self.pos_bias(positions)            # (seq_len, num_heads)
        pb = pb.permute(1, 0).unsqueeze(0).unsqueeze(2)  # (1, H, 1, seq_len)
        kernel_weights = kernel_weights + pb

        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            kernel_weights = kernel_weights * mask_expanded
            # Normalize by the REAL chunk count, not the padded slot count, so
            # the integration signal has the same scale for every document.
            denom = mask.sum(dim=1).clamp(min=1.0).view(-1, 1, 1, 1)
        else:
            denom = seq_len + 1e-6

        kernel_weights = kernel_weights / denom
        kernel_weights = self.weight_dropout(kernel_weights)
        integrated = torch.matmul(kernel_weights, V)
        integrated = integrated.permute(0, 2, 1, 3).contiguous()
        integrated = integrated.view(batch_size, seq_len, self.hidden_size)
        integrated = self.output_proj(integrated)
        return self.layer_norm(x + self.output_dropout(integrated))


class AdaptiveGate(nn.Module):
    """
    ✅ Improvement 4: Improved gate with LayerNorm + learnable temperature.
    - LayerNorm stabilises scale of concatenated inputs before gating
    - Temperature: learnable sharpness — gate can learn to be decisive
    - Strong initial bias (-1.0) → starts conservative (trusts attention first)
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.norm        = nn.LayerNorm(hidden_size * 2)
        self.gate_proj   = nn.Linear(hidden_size * 2, hidden_size)
        self.temperature = nn.Parameter(torch.ones(1) * 2.0)  # start sharp
        nn.init.constant_(self.gate_proj.bias, 0.0)  # neutral start — 50/50 mix

    def forward(self, attn, integ):
        combined = self.norm(torch.cat([attn, integ], dim=-1))
        return torch.sigmoid(self.gate_proj(combined) / self.temperature.clamp(min=0.1))


class BaselineModel(nn.Module):
    """BASELINE: Attention-only (standard transformer)."""

    def __init__(self, config):
        super().__init__()
        self.encoder = load_encoder(config.pretrained_model)
        self.encoder.gradient_checkpointing_enable()
        self.chunk_position = nn.Embedding(config.max_chunks, config.hidden_size)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size * 4, dropout=config.dropout, batch_first=True
            ) for _ in range(config.integration_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        batch_size, num_chunks, seq_len = input_ids.shape
        chunk_repr = encode_chunks(self.encoder, input_ids, attention_mask, chunk_mask)
        positions = torch.arange(num_chunks, device=input_ids.device)
        chunk_repr = chunk_repr + self.chunk_position(positions)
        # Same masking semantics as the integral model: padding chunks are
        # excluded from attention and from the max-pool.
        padding = (chunk_mask == 0) if chunk_mask is not None else None
        for layer in self.attention_layers:
            chunk_repr = layer(chunk_repr, src_key_padding_mask=padding)
        doc_repr = masked_max(chunk_repr, chunk_mask)
        return self.classifier(doc_repr)


class TruncationModel(nn.Module):
    """
    TRUNCATION BASELINE: Uses only the FIRST chunk (first 512 tokens).
    Proves: Full document processing matters.
    All remaining chunks are discarded — simulates the common approach
    of truncating long documents to fit BERT's 512-token window.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = load_encoder(config.pretrained_model)
        self.encoder.gradient_checkpointing_enable()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        # Only use the FIRST chunk (index 0) — truncate the rest
        first_ids  = input_ids[:, 0, :]    # (batch, seq_len)
        first_mask = attention_mask[:, 0, :]  # (batch, seq_len)
        outputs = self.encoder(input_ids=first_ids, attention_mask=first_mask)
        hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        # Mean-pool over tokens (masked)
        mask_f = first_mask.unsqueeze(-1).float()
        doc_repr = (hidden * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
        return self.classifier(doc_repr)


class MeanPoolModel(nn.Module):
    """
    MEAN POOL BASELINE: Processes ALL chunks but uses simple mean pooling
    to aggregate them — NO cross-chunk interaction (no attention, no integration).
    Proves: Cross-chunk interaction matters.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = load_encoder(config.pretrained_model)
        self.encoder.gradient_checkpointing_enable()
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        chunk_repr = encode_chunks(self.encoder, input_ids, attention_mask, chunk_mask)
        # Simple mean pool across chunks (masked) — NO cross-chunk interaction
        if chunk_mask is not None:
            mask_2d = chunk_mask.unsqueeze(-1)  # (B, C, 1)
            chunk_repr = chunk_repr * mask_2d
            doc_repr = chunk_repr.sum(dim=1) / (mask_2d.sum(dim=1) + 1e-8)
        else:
            doc_repr = chunk_repr.mean(dim=1)
        return self.classifier(doc_repr)


class IntegralTransformerModel(nn.Module):
    """NOVEL: Attention + Integration module."""

    def __init__(self, config):
        super().__init__()
        self.encoder = load_encoder(config.pretrained_model)
        self.encoder.gradient_checkpointing_enable()
        self.chunk_position = nn.Embedding(config.max_chunks, config.hidden_size)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size, nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_size * 4, dropout=config.dropout, batch_first=True
            ) for _ in range(config.integration_layers)
        ])
        self.integration_layers = nn.ModuleList([
            LearnedIntegrationKernel(config.hidden_size, config.integration_kernel_size,
                                     config.num_integration_heads,
                                     max_chunks=config.max_chunks,
                                     dropout=getattr(config, 'integration_dropout', 0.0))
            for _ in range(config.integration_layers)
        ])
        self.gates = nn.ModuleList([
            AdaptiveGate(config.hidden_size)
            for _ in range(config.integration_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(config.hidden_size, config.num_classes)
        )

    def forward(self, input_ids, attention_mask, chunk_mask=None):
        batch_size, num_chunks, seq_len = input_ids.shape
        chunk_repr = encode_chunks(self.encoder, input_ids, attention_mask, chunk_mask)
        positions = torch.arange(num_chunks, device=input_ids.device)
        chunk_repr = chunk_repr + self.chunk_position(positions)

        padding = (chunk_mask == 0) if chunk_mask is not None else None
        for attn, integ, gate in zip(self.attention_layers, self.integration_layers, self.gates):
            attn_out    = attn(chunk_repr, src_key_padding_mask=padding)
            int_out     = integ(chunk_repr, chunk_mask)
            gate_weight = gate(attn_out, int_out)
            chunk_repr  = (1 - gate_weight) * attn_out + gate_weight * int_out

        doc_repr = masked_max(chunk_repr, chunk_mask)
        return self.classifier(doc_repr)


MODEL_REGISTRY = {
    'integral': IntegralTransformerModel,
    'baseline': BaselineModel,
    'truncation': TruncationModel,
    'meanpool': MeanPoolModel,
}

# The roster itself lives in torch-free src/constants.py so plotting helpers
# work without torch; re-exported here and checked against the registry.
from src.constants import MODEL_NAMES

assert MODEL_NAMES == list(MODEL_REGISTRY), (
    "src/constants.py MODEL_NAMES is out of sync with MODEL_REGISTRY")
