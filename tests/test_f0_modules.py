import os
import sys
import tempfile

import pytest
import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.v2.pretrained_f0_encoder import PretrainedF0Encoder
from modules.v2.f0_cross_attention import F0CrossAttention


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def mock_v1_ckpt(tmp_path):
    """V1チェックポイントのモック (f0_embedding.weight のみ含む)"""
    ckpt_path = tmp_path / "v1_mock.pth"
    ckpt = {
        'net': {
            'length_regulator': {
                'module.f0_embedding.weight': torch.randn(256, 768),
            }
        }
    }
    torch.save(ckpt, ckpt_path)
    return str(ckpt_path)


# ── PretrainedF0Encoder Tests ─────────────────────────────────────


class TestPretrainedF0Encoder:

    def test_output_shape(self, mock_v1_ckpt):
        encoder = PretrainedF0Encoder(mock_v1_ckpt, hidden_dim=512)
        f0 = torch.rand(2, 100) * 500 + 50  # (B=2, T=100), 50-550 Hz
        out = encoder(f0)
        assert out.shape == (2, 100, 512)

    def test_output_shape_different_hidden_dim(self, mock_v1_ckpt):
        encoder = PretrainedF0Encoder(mock_v1_ckpt, hidden_dim=256)
        f0 = torch.rand(1, 50) * 500 + 50
        out = encoder(f0)
        assert out.shape == (1, 50, 256)

    def test_embedding_frozen(self, mock_v1_ckpt):
        encoder = PretrainedF0Encoder(mock_v1_ckpt, freeze_embedding=True)
        assert not encoder.f0_embedding.weight.requires_grad

    def test_embedding_unfrozen(self, mock_v1_ckpt):
        encoder = PretrainedF0Encoder(mock_v1_ckpt, freeze_embedding=False)
        assert encoder.f0_embedding.weight.requires_grad

    def test_proj_trainable(self, mock_v1_ckpt):
        encoder = PretrainedF0Encoder(mock_v1_ckpt, freeze_embedding=True)
        for param in encoder.proj.parameters():
            assert param.requires_grad

    def test_get_trainable_params(self, mock_v1_ckpt):
        encoder = PretrainedF0Encoder(mock_v1_ckpt, hidden_dim=512)
        trainable = list(encoder.get_trainable_params())
        # proj = Linear(768,512) + Linear(512,512) → 4つのパラメータ (weight+bias) × 2
        assert len(trainable) == 4

    def test_v1_weights_loaded(self, mock_v1_ckpt):
        """V1チェックポイントの重みが正しくロードされることを確認"""
        ckpt = torch.load(mock_v1_ckpt, map_location='cpu', weights_only=False)
        expected = ckpt['net']['length_regulator']['module.f0_embedding.weight']
        encoder = PretrainedF0Encoder(mock_v1_ckpt)
        assert torch.equal(encoder.f0_embedding.weight.data, expected)

    def test_zero_f0_handling(self, mock_v1_ckpt):
        """F0=0 (無声区間) の入力でもエラーなく動作"""
        encoder = PretrainedF0Encoder(mock_v1_ckpt)
        f0 = torch.zeros(1, 20)
        out = encoder(f0)
        assert out.shape == (1, 20, 512)
        assert not torch.isnan(out).any()

    def test_gradient_flows_through_proj(self, mock_v1_ckpt):
        """proj を通じて勾配が流れることを確認"""
        encoder = PretrainedF0Encoder(mock_v1_ckpt, freeze_embedding=True)
        f0 = torch.rand(2, 50) * 500 + 50
        out = encoder(f0)
        loss = out.sum()
        loss.backward()
        for param in encoder.proj.parameters():
            assert param.grad is not None


# ── F0CrossAttention Tests ────────────────────────────────────────


class TestF0CrossAttention:

    def test_output_shape(self):
        attn = F0CrossAttention(hidden_dim=512, num_heads=8)
        x = torch.randn(2, 100, 512)
        f0_features = torch.randn(2, 100, 512)
        out = attn(x, f0_features)
        assert out.shape == (2, 100, 512)

    def test_output_shape_different_f0_length(self):
        """F0 とメル長が異なる場合でも動作"""
        attn = F0CrossAttention(hidden_dim=512, num_heads=8)
        x = torch.randn(2, 100, 512)
        f0_features = torch.randn(2, 80, 512)  # F0の長さが異なる
        out = attn(x, f0_features)
        assert out.shape == (2, 100, 512)

    def test_with_mask(self):
        attn = F0CrossAttention(hidden_dim=512, num_heads=8)
        x = torch.randn(2, 100, 512)
        f0_features = torch.randn(2, 80, 512)
        # パディングマスク: Falseが有効、Trueがマスク
        f0_mask = torch.zeros(2, 80, dtype=torch.bool)
        f0_mask[0, 60:] = True  # バッチ0の後半をマスク
        f0_mask[1, 70:] = True
        out = attn(x, f0_features, f0_mask=f0_mask)
        assert out.shape == (2, 100, 512)
        assert not torch.isnan(out).any()

    def test_residual_connection(self):
        """初期状態でresidual_scaleが0 (sigmoid=0.5)で残差が支配的"""
        attn = F0CrossAttention(hidden_dim=512, num_heads=8, dropout=0.0)
        attn.eval()
        x = torch.randn(2, 50, 512)
        f0_features = torch.randn(2, 50, 512)
        out = attn(x, f0_features)
        # 出力が入力と大きく離れていない (残差接続が効いている)
        diff = (out - x).abs().mean()
        x_scale = x.abs().mean()
        assert diff < x_scale * 2  # 残差接続により出力は入力のオーダー内

    def test_gradient_flow(self):
        """全パラメータに勾配が流れることを確認"""
        attn = F0CrossAttention(hidden_dim=512, num_heads=8)
        x = torch.randn(2, 50, 512, requires_grad=True)
        f0_features = torch.randn(2, 50, 512, requires_grad=True)
        out = attn(x, f0_features)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert f0_features.grad is not None
        for param in attn.parameters():
            assert param.grad is not None

    def test_different_hidden_dim(self):
        attn = F0CrossAttention(hidden_dim=256, num_heads=4)
        x = torch.randn(1, 30, 256)
        f0_features = torch.randn(1, 30, 256)
        out = attn(x, f0_features)
        assert out.shape == (1, 30, 256)
