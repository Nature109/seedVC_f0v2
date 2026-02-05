import torch
import torch.nn as nn

from modules.v2.pretrained_f0_encoder import PretrainedF0Encoder
from modules.v2.f0_cross_attention import F0CrossAttention


class SeedSVC_V1(nn.Module):
    """
    SeedSVC V1.0.0 統合モデル

    V2 DiT を凍結したまま、各 Transformer 層の後に
    F0 Cross-Attention を挿入する「プラグイン方式」のラッパー。
    CFM の estimator として DiT の代わりに使用する。
    """

    def __init__(
        self,
        v1_ckpt_path: str,
        v2_model: nn.Module,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 13,
        dropout: float = 0.1,
        freeze_v2: bool = True,
    ):
        super().__init__()

        # V2 DiT (凍結)
        self.v2_dit = v2_model
        if freeze_v2:
            for param in self.v2_dit.parameters():
                param.requires_grad = False

        # F0 Encoder (V1埋め込み凍結 + proj学習)
        self.f0_encoder = PretrainedF0Encoder(
            v1_ckpt_path=v1_ckpt_path,
            hidden_dim=hidden_dim,
            freeze_embedding=True,
        )

        # F0 Cross-Attention layers (学習)
        self.f0_cross_attn = nn.ModuleList([
            F0CrossAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    @property
    def in_channels(self):
        """CFM が参照するため DiT から委譲"""
        return self.v2_dit.in_channels

    def get_trainable_params(self):
        """Optimizer用: 学習対象パラメータのみ"""
        params = list(self.f0_encoder.get_trainable_params())
        for layer in self.f0_cross_attn:
            params.extend(layer.parameters())
        return params

    def forward(self, x, prompt_x, x_lens, t, style, cond, f0=None, f0_lens=None):
        """
        DiT と同一のインターフェース + F0 引数。

        Args:
            x, prompt_x, x_lens, t, style, cond: DiT.forward と同一
            f0: (B, T_f0) F0値 (Hz)。None の場合は通常 VC として動作。
            f0_lens: (B,) 各サンプルの F0 長
        Returns:
            (B, in_channels, T) 推定されたベクトルフィールド
        """
        if f0 is not None:
            # F0エンコーディング
            f0_features = self.f0_encoder(f0)  # (B, T_f0, hidden_dim)

            # F0マスク (パディング部分を True に)
            f0_mask = None
            if f0_lens is not None:
                f0_mask = (
                    torch.arange(f0.shape[1], device=f0.device).unsqueeze(0)
                    >= f0_lens.unsqueeze(1)
                )

            # 各 DiT 層の後に Cross-Attention を適用する hook
            cross_attn_layers = self.f0_cross_attn

            def layer_hook(h, layer_idx):
                return cross_attn_layers[layer_idx](h, f0_features, f0_mask)

            return self.v2_dit(x, prompt_x, x_lens, t, style, cond, layer_hook=layer_hook)
        else:
            # F0 なし: 通常の VC として動作
            return self.v2_dit(x, prompt_x, x_lens, t, style, cond)
