# SeedSVC V1.0.0 実装計画

## Cross-Attention による F0 条件付け（V1学習済み埋め込み使用）

V2 (CFM) アーキテクチャに Cross-Attention ベースの F0 条件付けを追加し、歌唱音声変換に対応する。
**V1 の学習済み F0 埋め込みを再利用**し、最小限の学習で実現する。

---

## 1. 概要

### 1.1 目的

- V2 の CFM ベースアーキテクチャの利点（高速推論、話者特徴の強い抑制）を維持
- V1 の歌唱対応機能（F0 条件付け）を Cross-Attention で実現
- **V1 の学習済み F0 埋め込みを再利用**し、学習コストを最小化
- フレーム単位での精密なピッチコントロールを可能にする

### 1.2 段階的学習アプローチ

```
┌─────────────────────────────────────────────────────────────────┐
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  V1 Checkpoint  │    │  V2 Checkpoint  │                    │
│  │  (F0埋め込み)    │    │  (CFM本体)       │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   SeedSVC V1.0.0                        │   │
│  │                                                         │   │
│  │  [凍結] f0_embedding (256, 768) ← V1から                │   │
│  │      ↓                                                  │   │
│  │  [学習] proj (768 → 512)  ← 次元変換                    │   │
│  │      ↓                                                  │   │
│  │  [学習] Cross-Attention layers × 13                     │   │
│  │      ↓                                                  │   │
│  │  [凍結] V2 DiT 本体 ← V2から                            │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 学習フェーズ

| Phase | 学習対象 | 凍結 | 目的 |
|-------|---------|-----|------|
| **Phase 1** | proj + Cross-Attn | V2本体, F0埋め込み | F0注入方法の学習 |
| **Phase 2** (任意) | 全体 (低LR) | F0埋め込み | 全体の微調整 |

### 1.4 パラメータ数

```
学習対象:  ~14M (約17%)  ← proj + Cross-Attention
凍結:      ~67M (約83%)  ← V1埋め込み + V2 DiT
```

### 1.5 必要なチェックポイント

| チェックポイント | 用途 | パス |
|---------------|------|-----|
| V1 SVC | F0埋め込み抽出 | `checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth` |
| V2 CFM | DiT本体 | `checkpoints/v2/cfm.pt` |

**F0埋め込みの抽出方法:**
```python
ckpt = torch.load("checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
                  map_location='cpu', weights_only=False)
f0_embedding_weights = ckpt['net']['length_regulator']['module.f0_embedding.weight']
# torch.Size([256, 768])
```

### 1.6 アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────────┐
│                      SeedSVC V1.0.0                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Source Audio ──→ Content Extractor (Wide) ──→ Content Emb │
│                                                             │
│  Source Audio ──→ RMVPE ──→ PretrainedF0Encoder            │
│                              │                              │
│                    ┌─────────┴─────────┐                   │
│                    │  [凍結] V1埋め込み  │                   │
│                    │  [学習] proj層      │                   │
│                    └─────────┬─────────┘                   │
│                              ↓                              │
│  Target Audio ──→ Style Encoder ──→ Style Emb              │
│                                    │                        │
│  ┌─────────────────────────────────┴───────────────────┐   │
│  │       CFM Decoder (DiT[凍結] + F0 Cross-Attn[学習])   │   │
│  │  ┌───────────────────────────────────────────────┐  │   │
│  │  │  DiT Block × 13                               │  │   │
│  │  │  ├─ Self-Attention [凍結]                     │  │   │
│  │  │  ├─ Cross-Attention (Query: x, KV: F0) [学習] │  │   │
│  │  │  └─ Feed-Forward Network [凍結]               │  │   │
│  │  └───────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            ↓                                │
│                    Generated Mel ──→ Vocoder ──→ Audio     │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 新規モジュール

### 2.1 PretrainedF0Encoder（V1埋め込み再利用）

**ファイル**: `modules/v2/pretrained_f0_encoder.py`

```python
import torch
import torch.nn as nn
import math

# V1の量子化パラメータ
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * math.log(1 + f0_min / 700)
f0_mel_max = 1127 * math.log(1 + f0_max / 700)

def f0_to_coarse(f0: torch.Tensor, f0_bin: int = 256) -> torch.Tensor:
    """V1と同一のF0量子化処理"""
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse


class PretrainedF0Encoder(nn.Module):
    """
    V1の学習済みF0埋め込みを使用するエンコーダ

    - f0_embedding: V1から抽出、凍結 (学習不要)
    - proj: 768→512 次元変換 (学習対象)
    """
    def __init__(
        self,
        v1_ckpt_path: str,
        hidden_dim: int = 512,
        freeze_embedding: bool = True,
    ):
        super().__init__()

        # V1チェックポイントからF0埋め込みをロード
        ckpt = torch.load(v1_ckpt_path, map_location='cpu', weights_only=False)
        v1_weights = ckpt['net']['length_regulator']['module.f0_embedding.weight']

        self.n_f0_bins = v1_weights.shape[0]  # 256
        self.v1_dim = v1_weights.shape[1]      # 768
        self.hidden_dim = hidden_dim            # 512 (V2)

        # F0埋め込み (V1から、凍結)
        self.f0_embedding = nn.Embedding(self.n_f0_bins, self.v1_dim)
        self.f0_embedding.weight.data = v1_weights
        if freeze_embedding:
            self.f0_embedding.requires_grad_(False)

        # 次元変換 proj (学習対象)
        self.proj = nn.Sequential(
            nn.Linear(self.v1_dim, hidden_dim),
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
        f0_coarse = f0_to_coarse(f0, self.n_f0_bins)
        f0_emb = self.f0_embedding(f0_coarse)  # (B, T, 768)
        f0_features = self.proj(f0_emb)         # (B, T, 512)
        return f0_features

    def get_trainable_params(self):
        """学習対象パラメータのみを返す"""
        return self.proj.parameters()
```

### 2.2 F0CrossAttention

**ファイル**: `modules/v2/f0_cross_attention.py`

```python
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

        # 残差スケール (0から開始で学習初期の安定化)
        self.residual_scale = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        f0_features: torch.Tensor,
        f0_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        T_f0 = f0_features.shape[1]

        residual = x
        x = self.layer_norm(x)

        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(f0_features).view(B, T_f0, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(f0_features).view(B, T_f0, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if f0_mask is not None:
            attn_weights = attn_weights.masked_fill(
                f0_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        output = self.out_proj(attn_output)
        # スケール付き残差接続 (sigmoid で 0〜1 に制限)
        output = residual + self.residual_scale.sigmoid() * self.dropout(output)

        return output
```

---

## 3. 既存モジュールの修正

### 3.1 統合方針

V2 DiT 本体には手を加えず、各層の出力後に Cross-Attention を挿入する「プラグイン方式」を採用する。

```
V2 DiT Layer i (凍結)
    ↓
F0 Cross-Attention i (学習)  ← 新規挿入
    ↓
V2 DiT Layer i+1 (凍結)
```

### 3.2 SeedSVC_V1 統合モデル

**ファイル**: `modules/v2/seedsvc_v1.py`

V2 DiT を改変せず、外からラップする形で Cross-Attention を挿入する。

```python
class SeedSVC_V1(nn.Module):
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

    def get_trainable_params(self):
        """Optimizer用: 学習対象パラメータのみ"""
        params = list(self.f0_encoder.get_trainable_params())
        for layer in self.f0_cross_attn:
            params.extend(layer.parameters())
        return params

    def forward(self, x, prompt_x, x_lens, t, style, cond, f0, f0_lens=None):
        # F0エンコーディング
        f0_features = self.f0_encoder(f0)

        # F0マスク
        f0_mask = None
        if f0_lens is not None:
            f0_mask = torch.arange(f0.shape[1], device=f0.device).unsqueeze(0) >= f0_lens.unsqueeze(1)

        # V2 DiT の各層の後に Cross-Attention を適用
        h = self.v2_dit.prepare_input(x, prompt_x, cond, t, style)

        for dit_layer, cross_attn in zip(self.v2_dit.transformer.layers, self.f0_cross_attn):
            h = dit_layer(h)                            # V2 DiT layer (凍結)
            h = cross_attn(h, f0_features, f0_mask)     # F0 Cross-Attn (学習)

        output = self.v2_dit.output_projection(h)
        return output
```

### 3.3 CFM Module の修正

**ファイル**: `modules/v2/cfm.py`

F0 引数を inference / forward に追加。

```python
class CFM:
    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style,
                  f0=None, f0_lens=None,  # NEW
                  n_timesteps=10, temperature=1.0,
                  inference_cfg_rate=[0.5, 0.5]):
        # CFG用バッチ構築にF0を追加
        if f0 is not None:
            f0_batched = torch.cat([f0, f0, torch.zeros_like(f0)], dim=0)
            f0_lens_batched = torch.cat([f0_lens, f0_lens, f0_lens], dim=0)
        else:
            f0_batched = None
            f0_lens_batched = None

        # estimator呼び出しにF0を渡す
        dphi_dt = self.estimator(
            ..., f0=f0_batched, f0_lens=f0_lens_batched,
        )

    def forward(self, x1, x_lens, prompt_lens, mu, style,
                f0=None, f0_lens=None):  # NEW
        u_pred = self.estimator(
            ..., f0=f0, f0_lens=f0_lens,
        )
```

---

## 4. 学習パイプライン

### 4.1 Phase 1: proj + Cross-Attention のみ学習

**ファイル**: `train_seedsvc_v1.py`

```python
# モデル構築
model = SeedSVC_V1(
    v1_ckpt_path="checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
    v2_model=load_v2_model("checkpoints/v2/cfm.pt"),
    hidden_dim=512,
    freeze_v2=True,  # V2本体を凍結
)

# 学習対象パラメータのみでOptimizer作成
optimizer = AdamW(model.get_trainable_params(), lr=1e-4, weight_decay=0.01)
```

### 4.2 Phase 2: 全体微調整 (任意)

```python
# V2を解凍
for param in model.v2_dit.parameters():
    param.requires_grad = True

# 低い学習率で全体を学習 (F0埋め込みは引き続き凍結)
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-5
)
```

### 4.3 データローダーの修正

**ファイル**: `data/ft_dataset.py` に F0 抽出を追加

```python
# __getitem__ にF0抽出を追加
if self.f0_condition:
    f0 = self.rmvpe.infer_from_audio(audio.numpy(), sample_rate=self.sr,
                                      hop_length=self.hop_length)
    item['f0'] = torch.from_numpy(f0)
```

---

## 5. 設定ファイル

**ファイル**: `configs/v2/seedsvc_v1.yaml`

```yaml
sr: 22050
hop_length: 256

# Checkpoints
v1_ckpt_path: "checkpoints/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth"
v2_ckpt_path: "checkpoints/v2/cfm.pt"

# Model
model:
  hidden_dim: 512
  num_heads: 8
  num_layers: 13

# Training Phase 1: proj + Cross-Attn only
training_phase1:
  num_steps: 50000
  learning_rate: 1e-4
  weight_decay: 0.01
  gradient_clip: 1.0
  freeze: [v2_dit, f0_embedding]

# Training Phase 2 (Optional): full fine-tuning
training_phase2:
  enabled: false
  num_steps: 20000
  learning_rate: 1e-5
  freeze: [f0_embedding]
```

---

## 6. 推論パイプライン

**ファイル**: `inference_seedsvc_v1.py`

```python
def convert_singing_voice(source_path, target_path, model, rmvpe, pitch_shift=0.0):
    source_audio, sr = load_audio(source_path)
    target_audio, _ = load_audio(target_path)

    # F0抽出
    f0 = rmvpe.infer_from_audio(source_audio)

    # ピッチシフト (半音単位)
    if pitch_shift != 0.0:
        f0 = f0 * (2 ** (pitch_shift / 12))

    # 音声変換
    generated_mel = model(source_audio, target_audio, f0=f0)
    output_audio = model.vocoder(generated_mel)

    return output_audio, sr
```

---

## 7. 実装手順

| Phase | タスク | ファイル | 状態 |
|-------|-------|---------|------|
| 1 | PretrainedF0Encoder 実装 | `modules/v2/pretrained_f0_encoder.py` | [x] |
| 1 | F0CrossAttention 実装 | `modules/v2/f0_cross_attention.py` | [x] |
| 1 | 単体テスト | `tests/test_f0_modules.py` | [x] |
| 2 | SeedSVC_V1 統合モデル | `modules/v2/seedsvc_v1.py` | [x] |
| 2 | CFM への F0 引数追加 | `modules/v2/cfm.py` | [x] |
| 2 | DiT への layer_hook 追加 | `modules/v2/dit_model.py`, `modules/v2/dit_wrapper.py` | [x] |
| 2 | 統合テスト | `tests/test_seedsvc_v1.py` | [x] |
| 3 | 設定ファイル作成 | `configs/v2/seedsvc_v1.yaml` | [x] |
| 3 | 学習スクリプト作成 | `train_seedsvc_v1.py` | [x] |
| 3 | ストリーミング対応 | `data/streaming_dataset.py` | [x] |
| 3 | Phase 1 学習実行 | - | [ ] |
| 3 | Phase 2 学習実行 (任意) | - | [ ] |
| 4 | 推論スクリプト作成 | `inference_seedsvc_v1.py` | [ ] |
| 4 | 品質評価 | - | [ ] |

---

## 8. 評価指標

| 指標 | 説明 | 目標値 |
|-----|------|-------|
| F0 RMSE | ピッチ精度 | < 20 Hz |
| F0 Correlation | ピッチ相関 | > 0.95 |
| MCD | メルケプストラム歪み | < 6.0 dB |
| Speaker Similarity | 話者類似度 (ECAPA-TDNN) | > 0.85 |

---

## 9. リスクと対策

| リスク | 影響 | 対策 |
|-------|-----|-----|
| F0とコンテンツの競合 | ピッチが安定しない | residual_scale を 0 から開始、学習率分離 |
| V2 DiT 内部構造への依存 | 層分離が困難 | forward フックで対応 |
| 学習不安定 | 損失発散 | Warmup, Gradient Clipping |
| 推論速度低下 | リアルタイム性喪失 | Flash Attention 検討 |

---

## 10. 今後の拡張

- **V2.0.0**: 複合条件付け（Concat + Cross-Attn + AdaLN）
- ビブラート制御
- リアルタイムストリーミング対応
