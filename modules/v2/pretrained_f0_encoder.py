import torch
import torch.nn as nn

from modules.v2.length_regulator import f0_to_coarse


class PretrainedF0Encoder(nn.Module):
    """
    V1の学習済みF0埋め込みを使用するエンコーダ

    - f0_embedding: V1から抽出、凍結 (学習不要)
    - proj: 768→hidden_dim 次元変換 (学習対象)
    """

    N_F0_BINS = 256  # V1の量子化ビン数
    V1_DIM = 768     # V1の埋め込み次元

    def __init__(
        self,
        v1_ckpt_path: str,
        hidden_dim: int = 512,
        freeze_embedding: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # V1チェックポイントからF0埋め込みをロード
        ckpt = torch.load(v1_ckpt_path, map_location='cpu', weights_only=False)
        v1_weights = ckpt['net']['length_regulator']['module.f0_embedding.weight']
        assert v1_weights.shape == (self.N_F0_BINS, self.V1_DIM), (
            f"Expected shape ({self.N_F0_BINS}, {self.V1_DIM}), got {v1_weights.shape}"
        )

        # F0埋め込み (V1から、凍結)
        self.f0_embedding = nn.Embedding(self.N_F0_BINS, self.V1_DIM)
        self.f0_embedding.weight.data = v1_weights
        if freeze_embedding:
            self.f0_embedding.requires_grad_(False)

        # 次元変換 proj (学習対象)
        self.proj = nn.Sequential(
            nn.Linear(self.V1_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0: (B, T) F0値 (Hz)
        Returns:
            (B, T, hidden_dim) F0特徴量
        """
        f0_coarse = f0_to_coarse(f0, self.N_F0_BINS)
        f0_emb = self.f0_embedding(f0_coarse)  # (B, T, 768)
        f0_features = self.proj(f0_emb)         # (B, T, hidden_dim)
        return f0_features

    def get_trainable_params(self):
        """学習対象パラメータのみを返す"""
        return self.proj.parameters()
