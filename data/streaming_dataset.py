"""
HuggingFace Datasets ストリーミング対応データローダー

Emilia などの大規模データセットをローカルにダウンロードせずに
ストリーミングで学習するためのデータセットクラス。
"""
import torch
import numpy as np
import random
from torch.utils.data import IterableDataset, DataLoader
from modules.audio import mel_spectrogram


duration_setting = {
    "min": 1.0,
    "max": 30.0,
}


def to_mel_fn(wave, mel_fn_args):
    return mel_spectrogram(wave, **mel_fn_args)


class StreamingDataset(IterableDataset):
    """
    HuggingFace Datasets のストリーミングモードに対応した IterableDataset。

    使用例:
        from datasets import load_dataset
        hf_dataset = load_dataset("amphion/Emilia-Dataset", split="train", streaming=True)
        dataset = StreamingDataset(hf_dataset, spect_params, sr=22050)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=streaming_collate)
    """

    def __init__(
        self,
        hf_dataset,
        spect_params,
        sr=22050,
        audio_column="audio",
        shuffle_buffer_size=1000,
    ):
        """
        Args:
            hf_dataset: HuggingFace datasets の IterableDataset (streaming=True)
            spect_params: メルスペクトログラムのパラメータ
            sr: 出力サンプリングレート
            audio_column: 音声データの列名
            shuffle_buffer_size: シャッフルバッファサイズ
        """
        self.hf_dataset = hf_dataset
        self.sr = sr
        self.audio_column = audio_column
        self.shuffle_buffer_size = shuffle_buffer_size

        self.mel_fn_args = {
            "n_fft": spect_params['n_fft'],
            "win_size": spect_params.get('win_length', spect_params.get('win_size', 1024)),
            "hop_size": spect_params.get('hop_length', spect_params.get('hop_size', 256)),
            "num_mels": spect_params.get('n_mels', spect_params.get('num_mels', 80)),
            "sampling_rate": sr,
            "fmin": spect_params['fmin'],
            "fmax": None if spect_params.get('fmax') == "None" else spect_params.get('fmax'),
            "center": False
        }

    def _process_sample(self, sample):
        """HuggingFace の sample を (wave, mel) に変換"""
        try:
            audio_data = sample[self.audio_column]

            # HuggingFace Audio feature の形式
            if isinstance(audio_data, dict) and 'array' in audio_data:
                array = audio_data['array']
                orig_sr = audio_data['sampling_rate']
            elif isinstance(audio_data, str):
                # URL またはパスの場合、librosa で読み込む
                import librosa
                try:
                    array, orig_sr = librosa.load(audio_data, sr=None)
                except Exception as e:
                    print(f"Failed to load audio from {audio_data}: {e}")
                    return None
            else:
                # 生の numpy array の場合
                array = audio_data
                orig_sr = self.sr

            # numpy array に変換
            if isinstance(array, list):
                array = np.array(array, dtype=np.float32)
            elif not isinstance(array, np.ndarray):
                array = np.array(array, dtype=np.float32)
            else:
                array = array.astype(np.float32)

            # ステレオをモノラルに変換
            if array.ndim > 1:
                array = array.mean(axis=0)

            # 長さチェック
            duration = len(array) / orig_sr
            if duration < duration_setting["min"] or duration > duration_setting["max"]:
                return None

            # リサンプル
            if orig_sr != self.sr:
                import librosa
                array = librosa.resample(array, orig_sr=orig_sr, target_sr=self.sr)

            # torch tensor に変換
            wave = torch.from_numpy(array).float()

            # メルスペクトログラム計算
            mel = to_mel_fn(wave.unsqueeze(0), self.mel_fn_args).squeeze(0)

            return wave, mel

        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

    def __iter__(self):
        # シャッフルバッファを使用
        if self.shuffle_buffer_size > 0:
            dataset = self.hf_dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        else:
            dataset = self.hf_dataset

        for sample in dataset:
            result = self._process_sample(sample)
            if result is not None:
                yield result


def streaming_collate(batch):
    """ストリーミングデータセット用の collate 関数"""
    # None をフィルタリング
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    batch_size = len(batch)

    # 長さでソート
    lengths = [b[1].shape[1] for b in batch]
    batch_indexes = np.argsort(lengths)[::-1]
    batch = [batch[bid] for bid in batch_indexes]

    nmels = batch[0][1].size(0)
    max_mel_length = max([b[1].shape[1] for b in batch])
    max_wave_length = max([b[0].size(0) for b in batch])

    mels = torch.zeros((batch_size, nmels, max_mel_length)).float() - 10
    waves = torch.zeros((batch_size, max_wave_length)).float()

    mel_lengths = torch.zeros(batch_size).long()
    wave_lengths = torch.zeros(batch_size).long()

    for bid, (wave, mel) in enumerate(batch):
        mel_size = mel.size(1)
        mels[bid, :, :mel_size] = mel
        waves[bid, :wave.size(0)] = wave
        mel_lengths[bid] = mel_size
        wave_lengths[bid] = wave.size(0)

    return waves, mels, wave_lengths, mel_lengths


def build_streaming_dataloader(
    repo_id,
    spect_params,
    sr,
    batch_size=2,
    num_workers=0,
    split="train",
    audio_column=None,  # None = auto-detect ("audio" or "wav")
    shuffle_buffer_size=1000,
    token=None,
):
    """
    HuggingFace データセットからストリーミングデータローダーを構築。

    Args:
        repo_id: HuggingFace データセットID (例: "amphion/Emilia-Dataset")
        spect_params: メルスペクトログラムのパラメータ
        sr: サンプリングレート
        batch_size: バッチサイズ
        num_workers: DataLoader のワーカー数 (ストリーミングでは 0 推奨)
        split: データセットの split
        audio_column: 音声データの列名
        shuffle_buffer_size: シャッフルバッファサイズ
        token: HuggingFace トークン (gated repo 用)

    Returns:
        DataLoader
    """
    from datasets import load_dataset, Audio

    print(f"Loading streaming dataset from {repo_id}...")

    # audio_column を自動検出（features から取得、イテレートせずに）
    if audio_column is None:
        # データセット情報を取得
        from datasets import get_dataset_config_names, get_dataset_infos
        try:
            infos = get_dataset_infos(repo_id, token=token)
            if infos:
                info = list(infos.values())[0]
                if info.features:
                    feature_names = list(info.features.keys())
                    if "audio" in feature_names:
                        audio_column = "audio"
                    elif "wav" in feature_names:
                        audio_column = "wav"
                    print(f"Auto-detected audio column from dataset info: '{audio_column}'")
        except Exception as e:
            print(f"Could not get dataset info: {e}")

        # info から取得できなかった場合はデフォルト
        if audio_column is None:
            audio_column = "wav"  # Emilia 系は "wav" カラムを使用
            print(f"Using default audio column: '{audio_column}'")

    # データセットをロード（Audio 型の自動デコードを無効化）
    hf_dataset = load_dataset(
        repo_id,
        split=split,
        streaming=True,
        token=token,
    )

    # Audio feature を無効化して生データとして取得
    # torchcodec 問題を回避するため、Audio 型カラムはデコードせずに読み込む
    try:
        from datasets import Value
        # wav カラムが Audio 型の場合、String に変換して自動デコードを防ぐ
        hf_dataset = hf_dataset.cast_column(audio_column, Value("string"))
        print(f"Disabled automatic audio decoding for '{audio_column}'")
    except Exception as e:
        # 既に string 型の場合は何もしない
        print(f"Audio column '{audio_column}' will be processed manually: {e}")

    dataset = StreamingDataset(
        hf_dataset,
        spect_params,
        sr=sr,
        audio_column=audio_column,
        shuffle_buffer_size=shuffle_buffer_size,
    )

    # IterableDataset では num_workers=0 が安定
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=streaming_collate,
        num_workers=num_workers,
    )

    return dataloader


if __name__ == "__main__":
    # テスト用
    spect_params = {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "fmin": 0,
        "fmax": None,
    }

    # 日本語サブセットでテスト
    dataloader = build_streaming_dataloader(
        repo_id="MrDragonFox/JA_Emilia_Yodas_266h",
        spect_params=spect_params,
        sr=22050,
        batch_size=2,
    )

    for idx, batch in enumerate(dataloader):
        if batch is None:
            continue
        waves, mels, wave_lengths, mel_lengths = batch
        print(f"Batch {idx}: waves={waves.shape}, mels={mels.shape}")
        if idx >= 5:
            break
