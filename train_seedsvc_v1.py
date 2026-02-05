import os
import sys
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import torch
import torchaudio
import yaml
import argparse
import glob
import time
import shutil
from tqdm import tqdm
import hydra
from omegaconf import DictConfig

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from data.ft_dataset import build_ft_dataloader
from data.streaming_dataset import build_streaming_dataloader
from modules.v2.seedsvc_v1 import SeedSVC_V1
from modules.rmvpe import RMVPE


class Trainer:
    def __init__(
            self,
            config_path,
            pretrained_cfm_ckpt_path,
            data_dir,
            run_name,
            batch_size=2,
            num_workers=4,
            steps=50000,
            save_interval=1000,
            max_epochs=1000,
            phase=1,
            mixed_precision=None,
            streaming=False,
            hf_repo_id=None,
            hf_token=None,
        ):
        self.config_path = config_path
        self.config = yaml.safe_load(open(config_path, encoding='utf-8'))
        self.phase = phase
        self.streaming = streaming
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token

        # Setup logging directory
        self.log_dir = os.path.join("runs", run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))

        # Setup accelerator
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
        self.accelerator = Accelerator(
            project_dir=self.log_dir,
            split_batches=True,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=mixed_precision,
        )
        self.device = self.accelerator.device

        # Dataloader
        model_cfg = self.config['model']
        self._init_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            spect_params=model_cfg['mel_fn'],
            sr=model_cfg['sr'],
        )

        # Models
        self._init_models(pretrained_cfm_ckpt_path)

        # Training params
        self.iters = 0
        self.start_epoch = 0
        self.log_interval = 10
        self.max_steps = steps
        self.save_interval = save_interval
        self.max_epochs = max_epochs

    def _init_dataloader(self, data_dir, batch_size, num_workers, spect_params, sr):
        self.spect_params = spect_params
        self.sr = sr

        if self.streaming:
            # HuggingFace ストリーミングモード
            if self.hf_repo_id is None:
                raise ValueError("--hf-repo-id is required when using --streaming")
            print(f"Using streaming mode with HuggingFace dataset: {self.hf_repo_id}")
            self.train_dataloader = build_streaming_dataloader(
                repo_id=self.hf_repo_id,
                spect_params=spect_params,
                sr=sr,
                batch_size=batch_size,
                num_workers=0,  # ストリーミングでは 0 推奨
                token=self.hf_token,
            )
        else:
            # ローカルファイルモード
            self.train_dataloader = build_ft_dataloader(
                data_dir,
                spect_params,
                sr,
                batch_size=batch_size,
                num_workers=num_workers,
            )

    def _init_models(self, pretrained_cfm_ckpt_path):
        model_cfg = self.config['model']
        seedsvc_cfg = self.config['seedsvc_v1']
        training_cfg = self.config.get('training', {})

        # 1. Instantiate VoiceConversionWrapper via Hydra
        with self.accelerator.main_process_first():
            cfg = DictConfig(model_cfg)
            self.model = hydra.utils.instantiate(cfg).to(self.device)

            # 2. Load pretrained V2 checkpoints
            self.model.load_checkpoints(
                cfm_checkpoint_path=pretrained_cfm_ckpt_path,
                ar_checkpoint_path=None,
            )

            # 3. Freeze everything
            for p in self.model.parameters():
                p.requires_grad = False

            # 4. Wrap DiT with SeedSVC_V1
            freeze_v2 = (self.phase == 1)
            self.seedsvc_v1 = SeedSVC_V1(
                v1_ckpt_path=seedsvc_cfg['v1_ckpt_path'],
                v2_model=self.model.cfm.estimator,
                hidden_dim=seedsvc_cfg['hidden_dim'],
                num_heads=seedsvc_cfg['num_heads'],
                num_layers=seedsvc_cfg['num_layers'],
                dropout=seedsvc_cfg['dropout'],
                freeze_v2=freeze_v2,
            ).to(self.device)

            # 5. Replace estimator in CFM
            self.model.cfm.estimator = self.seedsvc_v1

        # RMVPE for F0 extraction (frozen, not part of training)
        from hf_utils import load_custom_model_from_hf
        rmvpe_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)

        # Optimizer (trainable params only)
        self._init_optimizers()

        # Load SeedSVC V1 checkpoint if resuming
        self._load_seedsvc_checkpoint()

        # Prepare model with accelerator
        self.model = self.accelerator.prepare(self.model)

    def _init_optimizers(self):
        training_cfg = self.config.get('training', {})
        from optimizers import MinLRExponentialLR
        from torch.optim import AdamW

        if self.phase == 1:
            lr = training_cfg.get('lr', 1e-4)
            params = list(self.seedsvc_v1.get_trainable_params())
        else:
            lr = training_cfg.get('phase2', {}).get('lr', 1e-5)
            params = [p for p in self.seedsvc_v1.parameters() if p.requires_grad]

        self.optimizer = AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-6,
            weight_decay=0.01,
        )
        self.scheduler = MinLRExponentialLR(self.optimizer, gamma=0.999996, min_lr=1e-5)
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)

    def _find_checkpoint(self, name_pattern):
        available = glob.glob(os.path.join(self.log_dir, name_pattern))
        if available:
            return max(available, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        return None

    def _load_seedsvc_checkpoint(self):
        ckpt_path = self._find_checkpoint("SeedSVC_V1_epoch_*_step_*.pth")
        if ckpt_path:
            print(f"Loading SeedSVC V1 checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            net = ckpt['net']
            self.seedsvc_v1.f0_encoder.load_state_dict(net['f0_encoder'], strict=False)
            self.seedsvc_v1.f0_cross_attn.load_state_dict(net['f0_cross_attn'], strict=False)
            self.iters = ckpt.get('iters', 0)
            self.start_epoch = ckpt.get('epoch', 0)

    def train(self):
        for epoch in range(self.start_epoch, self.start_epoch + self.max_epochs):
            epoch_start = time.time()

            try:
                self.train_dataloader.sampler.set_epoch(epoch)
            except AttributeError:
                pass

            self.model.train()

            for i, batch in enumerate(tqdm(self.train_dataloader)):
                # ストリーミングモードで無効なバッチをスキップ
                if batch is None:
                    continue
                self._process_batch(epoch, i, batch)
                if self.iters >= self.max_steps and self.accelerator.is_main_process:
                    print("Reached max steps, stopping training")
                    self._save_checkpoint(epoch)
                    return

            if self.accelerator.is_main_process:
                print(f"Epoch {epoch} completed in {time.time() - epoch_start:.2f}s")

            if epoch + 1 >= self.start_epoch + self.max_epochs:
                if self.accelerator.is_main_process:
                    print("Reached max epochs, stopping training")
                    self._save_checkpoint(epoch)
                return

    def _process_batch(self, epoch, i, batch):
        waves, mels, wave_lens, mel_lens = batch

        # Resample to 16kHz
        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lens.float() * 16000 / self.sr).long()

        with self.accelerator.autocast():
            # Content extraction (frozen)
            with torch.no_grad():
                _, content_indices, content_lens = self.model.content_extractor_wide(
                    waves_16k.to(self.device), wave_lengths_16k.to(self.device),
                )
                style = self.model.compute_style(
                    waves_16k.to(self.device), wave_lengths_16k.to(self.device),
                )

            # F0 extraction (frozen)
            with torch.no_grad():
                f0 = self.rmvpe.infer_from_audio_batch(waves_16k.to(self.device))
                # Compute F0 valid lengths from audio lengths
                # RMVPE uses hop_length=160 at 16kHz, with center=True
                f0_lens = (wave_lengths_16k.float() / 160).long() + 1
                f0_lens = f0_lens.clamp(max=f0.shape[1]).to(self.device)

            # Length regulation (frozen)
            with torch.no_grad():
                cond, _ = self.model.cfm_length_regulator(
                    content_indices, ylens=mel_lens.to(self.device),
                )

            # CFM forward with F0
            B = mels.size(0)
            prompt_len_max = mel_lens - 1
            prompt_len = (torch.rand([B], device=self.device) * prompt_len_max.to(self.device).float()).floor().long()
            prompt_len[torch.rand([B], device=self.device) < 0.1] = 0

            loss = self.model.cfm(
                mels.to(self.device), mel_lens.to(self.device),
                prompt_len, cond, style,
                f0=f0, f0_lens=f0_lens,
            )

            self.accelerator.backward(loss)

            if self.phase == 1:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.seedsvc_v1.get_trainable_params(), 1000.0,
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.seedsvc_v1.parameters() if p.requires_grad], 1000.0,
                )
            self.optimizer.step()
            self.scheduler.step(self.iters)
            self.optimizer.zero_grad()

        # Log
        if self.iters % self.log_interval == 0 and self.accelerator.is_main_process:
            with torch.no_grad():
                lr = self.scheduler.get_last_lr()[0] if i != 0 else 0
                print(
                    "Epoch %d, Iter %d, Loss: %.4f, Grad Norm: %.4f, LR: %.6f"
                    % (epoch, i, loss.item(), grad_norm, lr)
                )

        # Save checkpoint
        if self.iters != 0 and self.iters % self.save_interval == 0 and self.accelerator.is_main_process:
            self._save_checkpoint(epoch)

        self.iters += 1

    def _save_checkpoint(self, epoch):
        print('Saving checkpoint...')
        unwrapped = self.accelerator.unwrap_model(self.model)
        # Save only SeedSVC V1 components (f0_encoder + cross_attn)
        seedsvc_v1_module = unwrapped.cfm.estimator
        save_state = {
            'f0_encoder': seedsvc_v1_module.f0_encoder.state_dict(),
            'f0_cross_attn': seedsvc_v1_module.f0_cross_attn.state_dict(),
        }
        state = {
            'net': save_state,
            'iters': self.iters,
            'epoch': epoch,
            'phase': self.phase,
        }
        save_path = os.path.join(
            self.log_dir, 'SeedSVC_V1_epoch_%05d_step_%05d.pth' % (epoch, self.iters)
        )
        torch.save(state, save_path)
        print(f"Saved SeedSVC V1 checkpoint to {save_path}")

        self._remove_old_checkpoints("SeedSVC_V1_epoch_*_step_*.pth", max_keep=2)

    def _remove_old_checkpoints(self, pattern, max_keep=2):
        checkpoints = glob.glob(os.path.join(self.log_dir, pattern))
        if len(checkpoints) > max_keep:
            checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
            for cp in checkpoints[:-max_keep]:
                os.remove(cp)


def main(args):
    trainer = Trainer(
        config_path=args.config,
        pretrained_cfm_ckpt_path=args.pretrained_cfm_ckpt,
        data_dir=args.dataset_dir,
        run_name=args.run_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        steps=args.max_steps,
        save_interval=args.save_every,
        max_epochs=args.max_epochs,
        phase=args.phase,
        mixed_precision=args.mixed_precision,
        streaming=args.streaming,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SeedSVC V1.0.0 Training')
    parser.add_argument('--config', type=str, default='configs/v2/seedsvc_v1.yaml')
    parser.add_argument('--pretrained-cfm-ckpt', type=str, default=None,
                        help='Path to pretrained V2 CFM checkpoint (default: download from HF)')
    parser.add_argument('--dataset-dir', type=str, default=None,
                        help='Path to training data directory (required unless --streaming)')
    parser.add_argument('--run-name', type=str, default='seedsvc_v1')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=50000)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase: 1=frozen V2 DiT, 2=unfrozen V2 DiT fine-tune')
    parser.add_argument('--mixed-precision', type=str, default=None,
                        choices=[None, 'fp16', 'bf16'])
    # Streaming options
    parser.add_argument('--streaming', action='store_true',
                        help='Use HuggingFace streaming mode (no local download)')
    parser.add_argument('--hf-repo-id', type=str, default=None,
                        help='HuggingFace dataset repo ID (e.g., "MrDragonFox/JA_Emilia_Yodas_266h")')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace token for gated repos (optional, uses cached login if not specified)')
    args = parser.parse_args()

    # Validation
    if not args.streaming and args.dataset_dir is None:
        parser.error("--dataset-dir is required unless --streaming is used")
    if args.streaming and args.hf_repo_id is None:
        parser.error("--hf-repo-id is required when using --streaming")

    main(args)
