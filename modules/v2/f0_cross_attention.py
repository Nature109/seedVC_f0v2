import torch
import torch.nn as nn
import torch.nn.functional as F


class F0CrossAttention(nn.Module):
    """F0特徴量をKey/Valueとして使用するCross-Attention"""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # 残差スケール (sigmoid(0)=0.5 から開始し学習初期の安定化)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        f0_features: torch.Tensor,
        f0_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) DiT層の出力
            f0_features: (B, T_f0, D) F0エンコーダの出力
            f0_mask: (B, T_f0) Trueの位置をマスク (パディング部分)
        Returns:
            (B, T, D)
        """
        B, T, D = x.shape
        T_f0 = f0_features.shape[1]

        residual = x
        x = self.layer_norm(x)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(f0_features).view(B, T_f0, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(f0_features).view(B, T_f0, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_mask = None
        if f0_mask is not None:
            # (B, T_f0) → (B, 1, 1, T_f0) に拡張
            attn_mask = f0_mask.unsqueeze(1).unsqueeze(2)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        output = self.out_proj(attn_output)

        # スケール付き残差接続
        output = residual + self.residual_scale.sigmoid() * self.dropout(output)
        return output
