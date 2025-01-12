import os
import torchaudio
import numpy as np
import torch
import pandas as pd


def load_lj_speech_data(dataset_path, sr=22050):
    # 메타 데이터 로드
    metadata_path = os.path.join(dataset_path, "metadata.csv")
    wav_dir = os.path.join(dataset_path, "wavs")

    data = []
    metadata = pd.read_csv(metadata_path, sep="|", header=None, names=[
                           "wav_id", "text", "normalized_text"])

    for _, row in metadata.iterrows():
        wav_path = os.path.join(wav_dir, f"{row['wav_id']}.wav")
        wav, _ = torchaudio.load(wav_path)
        data.append({"text": row["normalized_text"], "wav": wav})

    return data, metadata_path


def collate_fn(batch):
    """Custom collate function for variable-length TTS data."""
    text_tensors = []
    mel_tensors = []
    mel_inputs = []
    stop_targets = []

    max_text_length = max(len(sample[0]) for sample in batch)
    max_mel_length = max(len(sample[1]) for sample in batch)

    for sample in batch:
        text_tensor = torch.nn.functional.pad(
            sample[0], (0, max_text_length - len(sample[0])), value=0
        )
        mel_tensor = torch.nn.functional.pad(
            sample[1], (0, 0, 0, max_mel_length - len(sample[1])), value=0
        )
        mel_input = torch.nn.functional.pad(
            sample[2], (0, 0, 0, max_mel_length - len(sample[2])), value=0
        )
        stop_target = torch.nn.functional.pad(
            sample[3], (0, max_mel_length - len(sample[3])), value=1
        )

        text_tensors.append(text_tensor)
        mel_tensors.append(mel_tensor)
        mel_inputs.append(mel_input)
        stop_targets.append(stop_target)

    return (
        torch.stack(text_tensors),
        torch.stack(mel_tensors),
        torch.stack(mel_inputs),
        torch.stack(stop_targets),
    )


def wav_to_mel(file_path, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
    try:
        # shape [1, num_samples]
        waveform, _ = torchaudio.load(file_path)
        # 모노 오디오 [1, num_samples]에서 채널 차원 제거([num_samples])
        waveform = waveform.squeeze(0)  

        # 멜 스펙트로그램 변형
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # 멜 스펙트로그램 생성
        mel_spectrogram = mel_spectrogram_transform(waveform)

        # 파워 스케일을 데시벨 스케일로 변경
        mel_db_transform = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        mel_db = mel_db_transform(mel_spectrogram)

        # 입력 형식을 맞추기 위해 (f, t)를 (t, f)로 전치
        return mel_db.T  # Shape: (time, frequency)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return torch.zeros((1, n_mels), dtype=torch.float)


def text_to_tensor(text, char_to_idx, max_length):
    """
    텍스트를 정수 텐서로 변환.
    """
    unk = len(char_to_idx)
    text_idx = [char_to_idx.get(char, unk)
                for char in text[:max_length]]  # 없는 문자는 0으로 처리
    text_idx += [char_to_idx['EOS']] * (max_length - len(text_idx))
    return torch.tensor(text_idx, dtype=torch.long)


def mel_to_tensor(mel, max_length, mel_dim):
    """
    멜 스펙트로그램을 텐서로 변환
    """
    mel = mel[:max_length]
    mel = torch.cat([mel, torch.zeros(max_length - len(mel), mel_dim)], dim=0)
    return mel
