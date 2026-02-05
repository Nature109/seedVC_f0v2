# SeedSVC V2.0.0 実装計画

## 複合条件付けによる高精度 F0 制御

V1.0.0 の Cross-Attention をベースに、Concatenation + AdaLN を組み合わせた複合条件付けアーキテクチャ。
V1.0.0 と同様に **V1 の学習済み F0 埋め込みを再利用** する。

---

## 1. 概要

### 1.1 V1.0.0 からの改善点

| 項目 | V1.0.0 | V2.0.0 |
|-----|--------|--------|
| F0条件付け | Cross-Attention のみ | Concat + Cross-Attn + AdaLN |
| F0エンコーダ | V1埋め込み + proj | V1埋め込み + proj + CNN多解像度 |
| F0情報の注入箇所 | 中間層のみ | 入力層 + 中間層 + 全層の正規化 |
| 粒度 | フレーム単位 | マルチスケール (局所〜大域) |

### 1.2 複合条件付けの役割分担

```
┌────────────────────────────────────────────────────────────────────┐
│                    F0 Conditioning Strategy                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  1. Concatenation (入力層)                                         │
│     ├─ 対象: F0_local (高解像度、フレーム単位)                      │
│     └─ 効果: 基本ピッチの正確な再現                                 │
│                                                                    │
│  2. Cross-Attention (中間層)                                       │
│     ├─ 対象: F0_mid (中解像度、4x downsampled)                     │
│     └─ 効果: ビブラート、ポルタメントの自然な表現                     │
│                                                                    │
│  3. AdaLN Modulation (全層)                                        │
│     ├─ 対象: F0_global (大域的な統計量)                              │
│     └─ 効果: 音域に応じた声質変化、安定性向上                        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 1.3 アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SeedSVC V2.0.0                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Source Audio ──→ Content Extractor ──→ Content Emb                │
│                                                                     │
│  Source Audio ──→ RMVPE ──→ HierarchicalF0Encoder                  │
│                              │                                      │
│                    ┌─────────┼─────────┐                           │
│                    ▼         ▼         ▼                           │
│               F0_local   F0_mid    F0_global                       │
│               (B,T,D)   (B,T/4,D)   (B,D)                         │
│                 │          │           │                            │
│  Target ──→ Style Encoder ──→ Style   │                            │
│                                │      │                            │
│  ┌─────────────────────────────┴──────┴────────────────────────┐   │
│  │            CFM Decoder (DiT + Multi-Scale F0)               │   │
│  │                                                             │   │
│  │  Input Layer:                                               │   │
│  │  ├─ [x, prompt, content, F0_local] → proj  ← Concat        │   │
│  │                                                             │   │
│  │  DiT Block × 13:                                           │   │
│  │  ├─ AdaLN (time + style + F0_global)   ← AdaLN Modulation  │   │
│  │  ├─ Self-Attention [凍結 or 学習]                           │   │
│  │  ├─ Cross-Attention (KV: F0_mid)       ← Cross-Attention   │   │
│  │  ├─ AdaLN (time + style + F0_global)                       │   │
│  │  └─ Feed-Forward Network                                   │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                            ↓                                        │
│                    Generated Mel ──→ Vocoder ──→ Audio             │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 段階的学習アプローチ

V1.0.0 と同様の段階的学習を採用。

| Phase | 学習対象 | 凍結 | 目的 |
|-------|---------|-----|------|
| **Phase 0** | V1.0.0 を事前に学習済み | - | ベースラインの確立 |
| **Phase 1** | Concat投影 + AdaLN + HierarchicalEncoder (CNN部) | V2 DiT, F0埋め込み, Cross-Attn (V1.0.0から) | 新規モジュールの学習 |
| **Phase 2** (任意) | 全体 (低LR) | F0埋め込み | 全体の微調整 |

---

## 2. 新規モジュール

### 2.1 HierarchicalF0Encoder

**ファイル**: `modules/v2/hierarchical_f0_encoder.py`

V1.0.0 の `PretrainedF0Encoder` を拡張し、CNN による多解像度特徴抽出を追加。

```python
import torch
import torch.nn as nn
from .pretrained_f0_encoder import PretrainedF0Encoder

class ConvBlock(nn.Module):
    """1D Convolutional Block with residual connection"""
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride, padding),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size, 1, padding),
            nn.GroupNorm(8, channels),
        )
        self.downsample = nn.Conv1d(channels, channels, 1, stride) if stride > 1 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.conv(x) + self.downsample(x))


class HierarchicalF0Encoder(nn.Module):
    """
    階層的F0エンコーダ

    V1.0.0 の PretrainedF0Encoder を内部に持ち、その出力を
    CNN で多解像度に分解する。

    出力:
    - f0_local: (B, T, D) フレーム単位の詳細特徴 (Concat用)
    - f0_mid: (B, T//4, D) 中解像度特徴 (Cross-Attention用)
    - f0_global: (B, D) グローバル特徴 (AdaLN用)
    """
    def __init__(
        self,
        v1_ckpt_path: str,
        hidden_dim: int = 512,
        freeze_embedding: bool = True,
    ):
        super().__init__()

        # V1.0.0 の PretrainedF0Encoder (埋め込み凍結 + proj)
        self.base_encoder = PretrainedF0Encoder(
            v1_ckpt_path=v1_ckpt_path,
            hidden_dim=hidden_dim,
            freeze_embedding=freeze_embedding,
        )

        # Local Encoder (高解像度、元の時間長を維持)
        self.local_encoder = nn.Sequential(
            ConvBlock(hidden_dim, kernel_size=3),
            ConvBlock(hidden_dim, kernel_size=3),
        )

        # Mid Encoder (中解像度、4x downsampling)
        self.mid_encoder = nn.Sequential(
            ConvBlock(hidden_dim, kernel_size=5, stride=2),
            ConvBlock(hidden_dim, kernel_size=5, stride=2),
            ConvBlock(hidden_dim, kernel_size=3),
        )

        # Global Encoder (グローバル統計)
        self.global_encoder = nn.Sequential(
            ConvBlock(hidden_dim, kernel_size=7, stride=4),
            ConvBlock(hidden_dim, kernel_size=7, stride=4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.global_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        f0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            f0: (B, T) F0値 (Hz)
        Returns:
            f0_local: (B, T, D)
            f0_mid: (B, T//4, D)
            f0_global: (B, D)
        """
        # ベースエンコーディング (V1埋め込み + proj)
        f0_base = self.base_encoder(f0)  # (B, T, D)

        # 転置して Conv1d 用に (B, D, T)
        f0_t = f0_base.transpose(1, 2)

        # Local: (B, D, T) → (B, T, D)
        f0_local = self.local_encoder(f0_t).transpose(1, 2)

        # Mid: (B, D, T) → (B, D, T//4) → (B, T//4, D)
        f0_mid = self.mid_encoder(f0_t).transpose(1, 2)

        # Global: (B, D, T) → (B, D, 1) → (B, D)
        f0_global = self.global_encoder(f0_t).squeeze(-1)
        f0_global = self.global_proj(f0_global)

        return f0_local, f0_mid, f0_global
```

### 2.2 MultiConditionAdaLN

**ファイル**: `modules/v2/multi_condition_adaln.py`

```python
import torch
import torch.nn as nn

class MultiConditionAdaLN(nn.Module):
    """
    複数条件を統合した Adaptive Layer Normalization

    条件: timestep + style + f0_global
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        timestep_dim: int = 512,
        style_dim: int = 192,
        f0_dim: int = 512,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        condition_dim = timestep_dim + style_dim + f0_dim
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

        # Zero-init for stable training
        nn.init.zeros_(self.condition_proj[-1].weight)
        nn.init.zeros_(self.condition_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        style_emb: torch.Tensor,
        f0_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
            timestep_emb: (B, D_t)
            style_emb: (B, D_s)
            f0_global: (B, D_f)
        """
        condition = torch.cat([timestep_emb, style_emb, f0_global], dim=-1)
        scale_shift = self.condition_proj(condition)     # (B, D*2)
        scale, shift = scale_shift.chunk(2, dim=-1)      # (B, D), (B, D)

        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x
```

### 2.3 UnifiedDiTBlock

**ファイル**: `modules/v2/unified_dit_block.py`

```python
import torch
import torch.nn as nn
from .f0_cross_attention import F0CrossAttention
from .multi_condition_adaln import MultiConditionAdaLN

class UnifiedDiTBlock(nn.Module):
    """
    複合条件付け DiT Block

    1. AdaLN (time + style + f0_global) → pre-norm
    2. Self-Attention
    3. F0 Cross-Attention (KV: f0_mid)
    4. AdaLN → pre-norm
    5. Feed-Forward
    """
    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        timestep_dim: int = 512,
        style_dim: int = 192,
        f0_dim: int = 512,
    ):
        super().__init__()

        # AdaLN (Self-Attention 前)
        self.adaln1 = MultiConditionAdaLN(hidden_dim, timestep_dim, style_dim, f0_dim)

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # F0 Cross-Attention (V1.0.0 と同じモジュール)
        self.f0_cross_attn = F0CrossAttention(hidden_dim, num_heads, dropout)

        # AdaLN (FFN 前)
        self.adaln2 = MultiConditionAdaLN(hidden_dim, timestep_dim, style_dim, f0_dim)

        # Feed-Forward
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # Residual scaling
        self.gamma1 = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(hidden_dim) * 0.1)

    def forward(
        self,
        x: torch.Tensor,
        timestep_emb: torch.Tensor,
        style_emb: torch.Tensor,
        f0_global: torch.Tensor,
        f0_mid: torch.Tensor,
        x_mask: torch.Tensor = None,
        f0_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # Self-Attention with AdaLN
        residual = x
        x = self.adaln1(x, timestep_emb, style_emb, f0_global)
        x_attn, _ = self.self_attn(x, x, x, key_padding_mask=x_mask)
        x = residual + self.gamma1 * x_attn

        # F0 Cross-Attention
        x = self.f0_cross_attn(x, f0_mid, f0_mask)

        # FFN with AdaLN
        residual = x
        x = self.adaln2(x, timestep_emb, style_emb, f0_global)
        x = residual + self.gamma2 * self.ffn(x)

        return x
```

---

## 3. 統合モデル

### 3.1 SeedSVC_V2

**ファイル**: `modules/v2/seedsvc_v2.py`

V1.0.0 のプラグイン方式とは異なり、DiT 本体を置き換える構成。
ただし Self-Attention の重みは V2 から初期化可能。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hierarchical_f0_encoder import HierarchicalF0Encoder
from .unified_dit_block import UnifiedDiTBlock

class SeedSVC_V2(nn.Module):
    def __init__(
        self,
        v1_ckpt_path: str,
        in_channels: int = 80,
        hidden_dim: int = 512,
        depth: int = 13,
        num_heads: int = 8,
        content_dim: int = 512,
        style_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 入力投影: [mel, prompt, content, f0_local] → hidden_dim
        # Concat により f0_local を入力に含める
        input_dim = in_channels * 2 + content_dim + hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # F0 Encoder (階層的)
        self.f0_encoder = HierarchicalF0Encoder(
            v1_ckpt_path=v1_ckpt_path,
            hidden_dim=hidden_dim,
            freeze_embedding=True,
        )

        # Timestep Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Style Projection
        self.style_proj = nn.Linear(style_dim, style_dim)

        # Transformer Blocks (複合条件付け)
        self.blocks = nn.ModuleList([
            UnifiedDiTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                timestep_dim=hidden_dim,
                style_dim=style_dim,
                f0_dim=hidden_dim,
            )
            for _ in range(depth)
        ])

        # Output
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_embed(emb)

    def forward(
        self,
        x: torch.Tensor,         # (B, T, C_mel) noisy mel
        prompt_x: torch.Tensor,  # (B, T, C_mel) prompt mel
        content: torch.Tensor,   # (B, T, C_content)
        t: torch.Tensor,         # (B,) timestep
        style: torch.Tensor,     # (B, C_style)
        f0: torch.Tensor,        # (B, T_f0) Hz
        x_lens: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # F0をmel長にリサンプル
        if f0.shape[1] != T:
            f0 = F.interpolate(
                f0.unsqueeze(1), size=T, mode='linear', align_corners=False
            ).squeeze(1)

        # 階層的F0エンコーディング
        f0_local, f0_mid, f0_global = self.f0_encoder(f0)

        # Concat: 入力にF0_localを結合
        x_in = torch.cat([x, prompt_x, content, f0_local], dim=-1)
        x_in = self.input_proj(x_in)

        # Condition embeddings
        t_emb = self.timestep_embedding(t)
        style_emb = self.style_proj(style)

        # Masks
        x_mask = None
        if x_lens is not None:
            x_mask = torch.arange(T, device=x.device).unsqueeze(0) >= x_lens.unsqueeze(1)

        T_mid = f0_mid.shape[1]
        f0_mask = None  # 必要に応じて作成

        # Transformer Blocks
        for block in self.blocks:
            x_in = block(
                x_in,
                timestep_emb=t_emb,
                style_emb=style_emb,
                f0_global=f0_global,
                f0_mid=f0_mid,
                x_mask=x_mask,
                f0_mask=f0_mask,
            )

        # Output
        x_out = self.out_norm(x_in)
        x_out = self.out_proj(x_out)
        return x_out
```

### 3.2 V2 Self-Attention 重みの初期化

```python
def init_from_v2_checkpoint(model: SeedSVC_V2, v2_ckpt_path: str):
    """V2 DiT の Self-Attention 重みで初期化"""
    v2_ckpt = torch.load(v2_ckpt_path, map_location='cpu', weights_only=False)
    v2_state = v2_ckpt['net']['cfm']

    for i, block in enumerate(model.blocks):
        prefix = f"module.estimator.transformer.layers.{i}"

        # Self-Attention の重みをコピー
        sa_key = f"{prefix}.attention.wqkv.weight"
        if sa_key in v2_state:
            block.self_attn.in_proj_weight.data = v2_state[sa_key]

        sa_out_key = f"{prefix}.attention.wo.weight"
        if sa_out_key in v2_state:
            block.self_attn.out_proj.weight.data = v2_state[sa_out_key]

    print(f"[init_from_v2] Loaded Self-Attention weights from {v2_ckpt_path}")
```

---

## 4. 学習パイプライン

### 4.1 V1.0.0 からの段階的移行

```
V1.0.0 学習済みモデル
    ↓
Cross-Attention の重みを V2.0.0 に移植
    ↓
新規モジュール (Concat投影, AdaLN, CNN) のみ学習  ← Phase 1
    ↓
全体微調整  ← Phase 2 (任意)
```

### 4.2 Phase 1: 新規モジュールの学習

```python
model = SeedSVC_V2(
    v1_ckpt_path="checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
    hidden_dim=512,
)

# V2 の Self-Attention 重みで初期化
init_from_v2_checkpoint(model, "checkpoints/v2/cfm.pt")

# V1.0.0 の Cross-Attention 重みを移植
load_v1_cross_attn_weights(model, "checkpoints/seedsvc_v1.pt")

# 新規モジュールのみ学習
trainable_modules = [
    model.input_proj,                      # Concat投影
    model.f0_encoder.local_encoder,        # CNN (local)
    model.f0_encoder.mid_encoder,          # CNN (mid)
    model.f0_encoder.global_encoder,       # CNN (global)
    model.f0_encoder.global_proj,          # Global投影
    *[b.adaln1 for b in model.blocks],     # AdaLN
    *[b.adaln2 for b in model.blocks],     # AdaLN
]

for param in model.parameters():
    param.requires_grad = False

for module in trainable_modules:
    for param in module.parameters():
        param.requires_grad = True

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

### 4.3 Phase 2: 全体微調整 (任意)

```python
# F0埋め込み以外を全て解凍
for param in model.parameters():
    param.requires_grad = True

model.f0_encoder.base_encoder.f0_embedding.requires_grad_(False)

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)
```

---

## 5. 設定ファイル

**ファイル**: `configs/v2/seedsvc_v2.yaml`

```yaml
sr: 22050
hop_length: 256

# Checkpoints
v1_ckpt_path: "checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth"
v2_ckpt_path: "checkpoints/v2/cfm.pt"
seedsvc_v1_ckpt_path: "checkpoints/seedsvc_v1.pt"  # V1.0.0の学習済み

# Model
model:
  in_channels: 80
  hidden_dim: 512
  depth: 13
  num_heads: 8
  content_dim: 512
  style_dim: 192

# Training Phase 1: new modules only
training_phase1:
  num_steps: 50000
  learning_rate: 1e-4
  weight_decay: 0.01
  gradient_clip: 1.0
  freeze:
    - f0_embedding
    - self_attn
    - f0_cross_attn
    - ffn

# Training Phase 2 (Optional): full fine-tuning
training_phase2:
  enabled: false
  num_steps: 20000
  learning_rate: 1e-5
  freeze: [f0_embedding]
```

---

## 6. 推論パイプライン

V1.0.0 と同一のインターフェースを維持。

```python
def convert_singing_voice(source_path, target_path, model, rmvpe, pitch_shift=0.0):
    source_audio, sr = load_audio(source_path)
    target_audio, _ = load_audio(target_path)

    f0 = rmvpe.infer_from_audio(source_audio)

    if pitch_shift != 0.0:
        f0 = f0 * (2 ** (pitch_shift / 12))

    generated_mel = model(source_audio, target_audio, f0=f0)
    output_audio = model.vocoder(generated_mel)

    return output_audio, sr
```

---

## 7. 実装手順

| Phase | タスク | ファイル | 前提 |
|-------|-------|---------|------|
| 0 | V1.0.0 の学習完了 | - | 必須 |
| 1 | HierarchicalF0Encoder 実装 | `modules/v2/hierarchical_f0_encoder.py` | |
| 1 | MultiConditionAdaLN 実装 | `modules/v2/multi_condition_adaln.py` | |
| 1 | UnifiedDiTBlock 実装 | `modules/v2/unified_dit_block.py` | |
| 1 | 単体テスト | `tests/test_v2_modules.py` | |
| 2 | SeedSVC_V2 統合モデル | `modules/v2/seedsvc_v2.py` | |
| 2 | V2 Self-Attn 重み初期化 | (上記に含む) | |
| 2 | V1.0.0 Cross-Attn 移植 | (上記に含む) | |
| 2 | 統合テスト | `tests/test_seedsvc_v2.py` | |
| 3 | 設定ファイル作成 | `configs/v2/seedsvc_v2.yaml` | |
| 3 | 学習スクリプト作成 | `train_seedsvc_v2.py` | |
| 3 | Phase 1 学習実行 | - | |
| 4 | 品質評価 | - | |

---

## 8. 評価指標

V1.0.0 よりも高い目標値を設定。

| 指標 | V1.0.0 目標 | V2.0.0 目標 |
|-----|-----------|-----------|
| F0 RMSE | < 20 Hz | < 15 Hz |
| F0 Correlation | > 0.95 | > 0.97 |
| MCD | < 6.0 dB | < 5.5 dB |
| Speaker Similarity | > 0.85 | > 0.87 |

---

## 9. V1.0.0 との関係

```
V1.0.0 (Cross-Attention のみ)
  │
  ├─ 学習済み Cross-Attention 重み → V2.0.0 に移植
  ├─ 学習済み proj 重み → V2.0.0 に流用
  │
  ▼
V2.0.0 (複合条件付け)
  │
  ├─ 新規: Concat (入力層)
  ├─ 流用: Cross-Attention (中間層) ← V1.0.0 から
  ├─ 新規: AdaLN (全層)
  └─ 新規: HierarchicalF0Encoder (CNN部)
```

V1.0.0 の成果を最大限活用し、追加学習を最小化する設計。
