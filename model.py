import torch
import torch.nn as nn


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, key_padding_mask=None):

        attn_output, _ = self.self_attention(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask
        )

        x = self.norm1(
            x + self.dropout(attn_output)
        )

        ff_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(ff_output)
        )

        return x


class ByteLevelTransformer(nn.Module):

    def __init__(
        self,
        vocab_size=257,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        ff_dim=1024,
        max_seq_len=256,
        dropout=0.1
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dim
        )

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)

        self.output_projection = nn.Linear(
            embed_dim,
            vocab_size
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds "
                f"max_seq_len {self.max_seq_len}"
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            -1
        )

        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=input_ids.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        for layer in self.layers:

            x = layer(
                x,
                causal_mask=causal_mask
            )

        x = self.layer_norm(x)

        logits = self.output_projection(x)

        return logits
