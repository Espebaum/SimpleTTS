import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torchaudio # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import soundfile as sf  # type: ignore
import os  # type: ignore

from torch.utils.data import DataLoader  # type: ignore
from model import SimpleTTS
from dataset import TTSDataset
from train import train_tts
from train import infer_tts
from preprocess import collate_fn
from preprocess import load_lj_speech_data

"""
CUDA 12.1 환경에서 진행

pip install torch==2.5.0+cu121 torchvision==0.20.0+cu121 torchaudio==2.5.0+cu121 --index-url https://download.pytorch.org/whl/cu121
"""

def plot_mel_spectrogram(mel, title="Mel Spectrogram"):
    """Visualize the mel spectrogram."""
    plt.figure(figsize=(10, 4))
    plt.imshow(mel.cpu().numpy().T, aspect="auto",
               origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


def save_mel_to_wav(mel, sample_rate, filename):
    """
    Args:
        mel (torch.Tensor): Mel Spectrogram (shape: [n_mels, time_frames]).
        sample_rate (int): Target sample rate.
        filename (str): Path to save the WAV file.
    """
    device = mel.device  # Get the device of the input mel tensor

    # Calculate n_fft and n_stft from mel.shape[0] (n_mels)
    n_mels = mel.shape[0]
    n_fft = max(2048, 4 * (n_mels - 1))  # Ensure n_fft >= win_length
    n_stft = n_fft // 2 + 1

    # Mel spectrogram to power spectrogram
    mel_to_linear_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_stft,
        n_mels=n_mels,
        sample_rate=sample_rate
    ).to(device)  # Move to the same device as mel

    power_spec = mel_to_linear_transform(mel)

    # Power spectrogram to waveform using Griffin-Lim
    griffin_lim_transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft
    ).to(device)  # Move to the same device as mel

    waveform = griffin_lim_transform(power_spec)

    # Save waveform as WAV file
    sf.write(filename, waveform.cpu().numpy(), samplerate=sample_rate)
    print(f"WAV file saved as {filename}")


# python main.py
if __name__ == "__main__":

    characters = [
        'EOS', ' ', '!', ',', '-', '.', ';', '?',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
        'y', 'z', 'à', 'â', 'è', 'é', 'ê', 'ü',
        '’', '“', '”',
    ]

    char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
    char_to_idx['<UNK>'] = len(char_to_idx)
    embedding_dim, hidden_dim, mel_dim = 128, 256, 80
    max_text_length, max_mel_length = 50, 100
    wav_dir = "C:\projects\my\LJSpeech-1.1\wavs"

    dataset_path = "C:\projects\my\LJSpeech-1.1"
    data, metadata_path = load_lj_speech_data(dataset_path)

    # (data, dictionary, 50, 100, 80)
    dataset = TTSDataset(metadata_path, wav_dir, char_to_idx,
                         max_text_length, max_mel_length, mel_dim)

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Initialize model (28, 128, 256, 80)
    model = SimpleTTS(vocab_size=len(char_to_idx) + 1,
                      embedding_dim=128, hidden_dim=256, mel_dim=mel_dim).cuda()

    # Check if saved model exists
    model_path = "C:\projects\my\simpletts.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded from {model_path}, skipping training.")
    else:
        # Train model if no saved model exists
        print("Training model...")
        train_tts(model, dataloader, epochs=10,
                  lr=0.001, criterion=nn.MSELoss())
        # Save trained model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved as {model_path}.")

    # 추론
    test_text = "hello world"
    generated_mel = infer_tts(model, test_text, char_to_idx, mel_dim)
    print("Generated mel shape:", generated_mel.shape)

    # 멜 스펙트로그램 시각화
    plot_mel_spectrogram(generated_mel, title="Generated Mel Spectrogram")

    # mel을 wav로 바꾸어 저장
    save_mel_to_wav(generated_mel, sample_rate=22050, filename="generated.wav")
