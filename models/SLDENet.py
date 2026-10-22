import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN


class SSD(nn.Module):
    """Shared-Lag Structural Decomposition. Input/output: [B, C, T]."""

    def __init__(self, enc_in=21, seq_len=96, d_model=512, window_size=24, detach_svd=True, return_info=False, **kwargs):
        super().__init__()
        self.enc_in, self.seq_len, self.L = enc_in, seq_len, window_size
        self.K = seq_len - window_size + 1
        self.detach_svd, self.return_info = detach_svd, return_info
        if not 1 <= window_size <= seq_len:
            raise ValueError(f"window_size must be in [1, seq_len], got {window_size}.")
        counts = F.fold(torch.ones(1, self.L, self.K), (1, seq_len), (1, self.L), stride=(1, 1))
        self.register_buffer("counts", counts.reshape(1, 1, seq_len), persistent=False)

        self.norm_embedding = nn.Linear(seq_len, d_model)
        self.dominant_embedding = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        self.residual_embedding = nn.Linear(seq_len, d_model)

    def _basis(self, H):
        # H: [B, C, L, K]; shared-lag Gram matrix G: [B, L, L]
        G = torch.einsum("bclk,bcmk->blm", H, H) / (H.shape[1] * H.shape[-1])
        U, S, _ = torch.linalg.svd(G, full_matrices=False)
        gap = S[:, :-1] - S[:, 1:]
        # Preserve the original +2 convention to reproduce existing rank results.
        rank = gap.argmax(dim=-1) + 2
        max_rank = int(rank.max().item())
        U = U[:, :, :max_rank]
        mask = torch.arange(max_rank, device=H.device)[None, :] < rank[:, None]
        U = U * mask[:, None, :].to(U.dtype)
        return U, S, gap, rank

    def _fold(self, H, B, C):
        # Overlapping trajectory windows are averaged to reconstruct [B, C, T].
        y = F.fold(H.reshape(B * C, self.L, self.K), (1, self.seq_len), (1, self.L), stride=(1, 1))
        return y.reshape(B, C, self.seq_len) / self.counts.to(device=y.device, dtype=y.dtype)

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(f"SSD expects [B, C, T], got {tuple(x.shape)}.")
        B, C, T = x.shape
        if C != self.enc_in or T != self.seq_len:
            raise ValueError(f"Expected [B, {self.enc_in}, {self.seq_len}], got {tuple(x.shape)}.")

        H = x.float().unfold(-1, self.L, 1).permute(0, 1, 3, 2).contiguous()
        if self.detach_svd:
            with torch.no_grad():
                U, S, gap, rank = self._basis(H)
        else:
            U, S, gap, rank = self._basis(H)

        coef = torch.einsum("blr,bclk->bcrk", U, H)
        H_low = torch.einsum("blr,bcrk->bclk", U, coef)
        low = self._fold(H_low, B, C).to(dtype=x.dtype)
        high = x - low

        x_norm_emb = self.norm_embedding(x)
        x_d_emb = self.dominant_embedding(low)
        x_r_emb = self.residual_embedding(high)

        outputs = {"low": x_d_emb, "high": x_r_emb, "x_norm":x_norm_emb}
        if self.return_info:
            outputs["info"] = {"singular_values": S.detach(), "spectral_gap": gap.detach(), "adaptive_rank": rank.detach()}
        return outputs


class DRE(nn.Module):
    """Dominant-Residual Enhancement. Input: [B, C, T]; output: [B, C, D]."""

    def __init__(self, seq_len, d_model, num_heads=4, attn_dropout=0.5):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

    
        self.dominant_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=attn_dropout, batch_first=True
        )
        self.residual_projection = nn.Linear(d_model, d_model)

    def forward(self, x_norm_emb, x_d, x_r):
        # All inputs: [B, C, T]; embeddings: [B, C, D]

        # Q = X_norm_emb; K = V = X_D_emb.
        attn_out, _ = self.dominant_attention(
            query=x_norm_emb, key=x_d, value=x_d, need_weights=False
        )
        dominant_out = attn_out + x_d
        residual_out = self.residual_projection(x_r)
        return dominant_out + residual_out


class Model(nn.Module):
    """SLDRNet: RevIN -> SSD -> DRE -> Predictor -> Inverse RevIN."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = getattr(configs, "use_revin", True)

        self.revin_layer = RevIN(self.enc_in, affine=True) if self.use_revin else None
        self.ssd = SSD(
            enc_in=self.enc_in,
            seq_len=self.seq_len,
            d_model=self.d_model,
            window_size=getattr(configs, "window_size", 24),
            detach_svd=getattr(configs, "detach_svd", True),
            return_info=False
        )
        self.dre = DRE(
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_heads=getattr(configs, "n_heads", 4),
            attn_dropout=getattr(configs, "attn_dropout", 0.5)
        )
        self.predictor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU()
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len)
        )

    def forward(self, x):
        # Input/output: [B, T, C]
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"Expected [B, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}.")

        x_norm = self.revin_layer(x, mode="norm") if self.use_revin else x
        x_input = x_norm.transpose(1, 2).contiguous()  # [B, C, T]

        # SSD: decompose normalized input into dominant and residual components.
        components = self.ssd(x_input)
        x_d, x_r ,x_norm_emb= components["low"], components["high"], components["x_norm"]

        # DRE: component-specific enhancement and fusion.
        hidden = self.dre(x_norm_emb, x_d, x_r)  # [B, C, D]

        # Predictor: map enhanced representations to the prediction horizon.
        hidden = self.predictor(hidden)
        output = self.output_proj(hidden).transpose(1, 2).contiguous()  # [B, H, C]

        if self.use_revin:
            output = self.revin_layer(output, mode="denorm")
        return output