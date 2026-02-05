import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from modules.v2.dit_wrapper import DiT
from modules.v2.seedsvc_v1 import SeedSVC_V1
from modules.v2.cfm import CFM


# ── Fixtures ──────────────────────────────────────────────────────


HIDDEN_DIM = 64
NUM_HEADS = 4
DEPTH = 3
IN_CHANNELS = 16
CONTENT_DIM = 32
STYLE_DIM = 16
BLOCK_SIZE = 256


@pytest.fixture
def mock_v1_ckpt(tmp_path):
    """V1 チェックポイントの mock"""
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


@pytest.fixture
def v2_dit():
    """小さい DiT モデル (テスト用)"""
    model = DiT(
        time_as_token=True,
        style_as_token=True,
        uvit_skip_connection=False,
        block_size=BLOCK_SIZE,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        in_channels=IN_CHANNELS,
        content_dim=CONTENT_DIM,
        style_encoder_dim=STYLE_DIM,
        class_dropout_prob=0.0,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
    )
    return model


@pytest.fixture
def seedsvc_v1(mock_v1_ckpt, v2_dit):
    """SeedSVC_V1 モデル"""
    return SeedSVC_V1(
        v1_ckpt_path=mock_v1_ckpt,
        v2_model=v2_dit,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        num_layers=DEPTH,
        dropout=0.0,
        freeze_v2=True,
    )


def make_inputs(batch_size=2, mel_len=50, f0_len=50, device='cpu'):
    """テスト用ダミー入力を生成"""
    return dict(
        x=torch.randn(batch_size, IN_CHANNELS, mel_len, device=device),
        prompt_x=torch.randn(batch_size, IN_CHANNELS, mel_len, device=device),
        x_lens=torch.full((batch_size,), mel_len, dtype=torch.long, device=device),
        t=torch.rand(batch_size, device=device),
        style=torch.randn(batch_size, STYLE_DIM, device=device),
        cond=torch.randn(batch_size, mel_len, CONTENT_DIM, device=device),
        f0=torch.rand(batch_size, f0_len, device=device) * 500 + 50,
        f0_lens=torch.full((batch_size,), f0_len, dtype=torch.long, device=device),
    )


# ── SeedSVC_V1 Tests ─────────────────────────────────────────────


class TestSeedSVC_V1:

    def test_output_shape_with_f0(self, seedsvc_v1):
        """F0 付きの出力形状が DiT と同一 (B, in_channels, T)"""
        inputs = make_inputs()
        out = seedsvc_v1(**inputs)
        assert out.shape == (2, IN_CHANNELS, 50)

    def test_output_shape_without_f0(self, seedsvc_v1):
        """F0 なしで通常 VC として動作"""
        inputs = make_inputs()
        del inputs['f0']
        del inputs['f0_lens']
        out = seedsvc_v1(**inputs)
        assert out.shape == (2, IN_CHANNELS, 50)

    def test_v2_dit_frozen(self, seedsvc_v1):
        """V2 DiT の全パラメータが凍結されている"""
        for param in seedsvc_v1.v2_dit.parameters():
            assert not param.requires_grad

    def test_trainable_params_only(self, seedsvc_v1):
        """get_trainable_params が proj + cross_attn のみ返す"""
        trainable = list(seedsvc_v1.get_trainable_params())
        total_trainable = sum(p.numel() for p in trainable)

        # proj: Linear(768, 64) + Linear(64, 64) = 4 params
        # cross_attn × 3 layers: 各層 q/k/v/out proj + layer_norm + residual_scale
        assert total_trainable > 0

        # V2 DiT のパラメータが含まれていないことを確認
        v2_params = set(id(p) for p in seedsvc_v1.v2_dit.parameters())
        for p in trainable:
            assert id(p) not in v2_params

    def test_in_channels_property(self, seedsvc_v1):
        """in_channels が DiT から正しく委譲される"""
        assert seedsvc_v1.in_channels == IN_CHANNELS

    def test_gradient_flow(self, seedsvc_v1):
        """loss.backward() で cross_attn/proj に勾配が到達"""
        inputs = make_inputs()
        out = seedsvc_v1(**inputs)
        loss = out.sum()
        loss.backward()

        # proj に勾配あり
        for param in seedsvc_v1.f0_encoder.proj.parameters():
            assert param.grad is not None

        # cross_attn に勾配あり
        for layer in seedsvc_v1.f0_cross_attn:
            for param in layer.parameters():
                assert param.grad is not None

        # V2 DiT には勾配なし (凍結)
        for param in seedsvc_v1.v2_dit.parameters():
            assert param.grad is None

    def test_f0_embedding_frozen(self, seedsvc_v1):
        """F0 埋め込みが凍結されている"""
        assert not seedsvc_v1.f0_encoder.f0_embedding.weight.requires_grad


# ── CFM Integration Tests ────────────────────────────────────────


class TestCFMIntegration:

    def test_cfm_forward_with_f0(self, seedsvc_v1):
        """CFM.forward に f0 を渡して loss が計算される"""
        cfm = CFM(estimator=seedsvc_v1)
        B, T = 2, 50
        loss = cfm(
            x1=torch.randn(B, IN_CHANNELS, T),
            x_lens=torch.full((B,), T, dtype=torch.long),
            prompt_lens=torch.full((B,), 10, dtype=torch.long),
            mu=torch.randn(B, T, CONTENT_DIM),
            style=torch.randn(B, STYLE_DIM),
            f0=torch.rand(B, T) * 500 + 50,
            f0_lens=torch.full((B,), T, dtype=torch.long),
        )
        assert loss.dim() == 0  # scalar
        assert not torch.isnan(loss)

    def test_cfm_forward_without_f0(self, seedsvc_v1):
        """CFM.forward に f0=None で通常動作 (後方互換性)"""
        cfm = CFM(estimator=seedsvc_v1)
        B, T = 2, 50
        loss = cfm(
            x1=torch.randn(B, IN_CHANNELS, T),
            x_lens=torch.full((B,), T, dtype=torch.long),
            prompt_lens=torch.full((B,), 10, dtype=torch.long),
            mu=torch.randn(B, T, CONTENT_DIM),
            style=torch.randn(B, STYLE_DIM),
        )
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_cfm_backward_with_f0(self, seedsvc_v1):
        """CFM + SeedSVC_V1 で勾配計算が正常に完了"""
        cfm = CFM(estimator=seedsvc_v1)
        B, T = 2, 50
        loss = cfm(
            x1=torch.randn(B, IN_CHANNELS, T),
            x_lens=torch.full((B,), T, dtype=torch.long),
            prompt_lens=torch.full((B,), 10, dtype=torch.long),
            mu=torch.randn(B, T, CONTENT_DIM),
            style=torch.randn(B, STYLE_DIM),
            f0=torch.rand(B, T) * 500 + 50,
            f0_lens=torch.full((B,), T, dtype=torch.long),
        )
        loss.backward()

        # trainable params に勾配が到達
        for param in seedsvc_v1.get_trainable_params():
            assert param.grad is not None
