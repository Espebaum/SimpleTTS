import os
import torch
from torch.utils.data import Dataset
from preprocess import wav_to_mel, text_to_tensor, mel_to_tensor


class TTSDataset(Dataset):
    def __init__(self, metadata_path, wav_dir, char_to_idx, max_text_length, max_mel_length, mel_dim):
        """
        Args:
            metadata_path: Path to the metadata file (e.g., metadata.csv).
            wav_dir: Directory containing WAV files.
            char_to_idx: Dictionary for mapping characters to indices.
            max_text_length: Maximum length for text sequences.
            max_mel_length: Maximum length for mel spectrograms.
            mel_dim: Number of mel bands.
        """
        self.wav_dir = wav_dir
        self.char_to_idx = char_to_idx
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        self.mel_dim = mel_dim

        # Load metadata
        with open(metadata_path, "rt", encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 메타 데이터로부터 텍스트와 오디오 경로 추출
        text = self.metadata[idx][1]
        wav_path = os.path.join(self.wav_dir, self.metadata[idx][0] + ".wav")

        # 텍스트를 텐서로 변환
        text_tensor = text_to_tensor(
            text, self.char_to_idx, self.max_text_length)

        # wav 파일을 mel 스펙트로그램으로 변환
        mel_tensor = wav_to_mel(wav_path, n_mels=self.mel_dim)

        mel_tensor = mel_to_tensor(
            mel_tensor, self.max_mel_length, self.mel_dim)

        mel_input = torch.cat(
            [torch.zeros(1, self.mel_dim), mel_tensor[:-1]], dim=0)

        # Stop token
        stop_token = torch.tensor(
            [0] * (len(mel_tensor) - 1) + [1] *
            (self.max_mel_length - len(mel_tensor)),
            dtype=torch.float
        )

        return text_tensor, mel_tensor, mel_input, stop_token
