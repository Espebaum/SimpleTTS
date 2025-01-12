# 프로젝트 결과 보고서

[참고, Build text-to-speech from scratch](https://medium.com/@tttzof351/build-text-to-speech-from-scratch-part-1-ba8b313a504f)

## (0) Requirements

- cuda 12.1 환경에서 진행되었습니다.    

```
torch==2.5.0+cu121
torchvision==0.20.0+cu121
torchaudio==2.5.0+cu121
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
soundfile>=0.11.0
```

## (1) 데이터 전처리

### 딕셔너리

- 텍스트를 정규화하기 위해, 아래의 딕셔너리를 사용합니다. 이는 문자를 사용하여 정규화하는 것으로 간단하고 범용적인 방법입니다. 

```python
# 사용할 문자들의 리스트
characters = [
        'EOS', ' ', '!', ',', '-', '.', ';', '?',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
        'y', 'z', 'à', 'â', 'è', 'é', 'ê', 'ü',
        '’', '“', '”', 
    ]


char_to_idx = {char: idx + 1 for idx, char in enumerate(characters)}
```

### LJSpeech

- 데이터 셋으로 LJSpeech를 사용했습니다. 데이터셋을 로드합니다.

```python
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
```

- 위 로드 함수는 아래의 `metadata.csv`를 파싱하며, 해당하는 wav 파일로부터 ndarray를 추출하여 문장과 wav ndarray를 짝지어 반환합니다.

```
LJ001-0001|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition|Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition
LJ001-0002|in being comparatively modern.|in being comparatively modern.

.
.
.

```

```python
data[0] = {
    "text": "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition",
    "wav": numpy.array([...])  # LJ001-0001.wav의 오디오 데이터
}
```

### 커스텀 데이터셋

- 앞서 만들어둔 딕셔너리, metadata의 경로와 임의의 변수들을 사용해 커스텀 데이터셋을 생성합니다.

```python
dataset = TTSDataset(metadata_path, wav_dir, max_text_length, max_mel_length, mel_dim)

class TTSDataset(Dataset):
    def __init__(self, metadata_path, wav_dir, char_to_idx, max_text_length, max_mel_length, mel_dim):
        # 변수 할당
		self.wav_dir = wav_dir
        self.char_to_idx = char_to_idx
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        self.mel_dim = mel_dim
        
        # 메타 데이터 로드
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = [line.strip().split("|") for line in f]
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Extract text and audio path from metadata
        text = self.metadata[idx][1]
		"""
		self.metadata = [
			["LJ001-0001", 
			"Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", 
			"Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition"],
			... 
		]
		"""

        wav_path = os.path.join(self.wav_dir, self.metadata[idx][0] + ".wav") # LJSpeech-1.1\wavs\LJ001-0001.wav
        
        # 텍스트를 텐서로 변환
        text_tensor = text_to_tensor(text, self.char_to_idx, self.max_text_length)
		"""
		"Hello, world" = tensor([15, 12, 19, 19, 22, 3, 1, 30, 22, 25, 19, 11, 0, 0, 0, ... ], dtype=torch.int32)
		"""
        
        # wav를 mel spectrogram으로 변환
        mel_tensor = wav_to_mel(wav_path, n_mels=self.mel_dim)
        mel_tensor = mel_to_tensor(mel_tensor, self.max_mel_length, self.mel_dim)
		mel_input = torch.cat([torch.zeros(1, self.mel_dim), mel_tensor[:-1]], dim=0)
        
        # Stop token
        stop_token = torch.tensor(
            [0] * (len(mel_tensor) - 1) + [1] * (self.max_mel_length - len(mel_tensor)),
            dtype=torch.float
        )
        
        return text_tensor, mel_tensor, mel_input, stop_token

dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
```

- 데이터셋의 메타 데이터의 각 인덱스에는 0번째로 텍스트가, 1번째로 wav ndarray가 포함되는데, 그 요소들을 각각 텐서로 변환하기 위한 함수들입니다.

```python
def text_to_tensor(text, char_to_idx, max_length):
    """
    텍스트를 정수 텐서로 변환.
    """
    unk = len(char_to_idx)
    text_idx = [char_to_idx.get(char, unk)
                for char in text[:max_length]]  # 없는 문자는 0으로 처리
    text_idx += [char_to_idx['EOS']] * (max_length - len(text_idx))
    return torch.tensor(text_idx, dtype=torch.long)

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

def mel_to_tensor(mel, max_length, mel_dim):
    """
    멜 스펙트로그램을 텐서로 변환
    """
    mel = mel[:max_length]
    mel = torch.cat([mel, torch.zeros(max_length - len(mel), mel_dim)], dim=0)
    return mel
```

## (2) 모델

- 참고했던 [Build text-to-speech from scratch](https://medium.com/@tttzof351/build-text-to-speech-from-scratch-part-1-ba8b313a504f)에서는 Transformer 기반의 TTS Model을 만들었지만, 이 프로젝트에서는 학습을 위해 아주 간단한 하나의 인코더-디코더 구조로 된 GRU 모델을 사용했습니다.

```python
class SimpleTTS(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, mel_dim):
		super(SimpleTTS, self).__init__()
		# Text Encoder
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.encoder_rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)

		# Mel Decoder
		self.decoder_rnn = nn.GRU(mel_dim, hidden_dim, batch_first=True)
		self.mel_linear = nn.Linear(hidden_dim, mel_dim) # (256, 100)

		# Stop Token Predictor
		self.stop_token = nn.Linear(hidden_dim, 1)

	def	forward(self, text, mel_input):
		"""
		Encode Text : 
			-> text.shape ([2, 50]) 
		Decode Mel Spectrogram : 
			-> mel_input.shape ([2, 100, 80])
		"""

		# Encode text ([2, 50])
		text_embedded = self.embedding(text) # (batch, text_len, embedding_dim) 
		# ([2, 50, 128])
		encoder_outputs, _ = self.encoder_rnn(text_embedded) # (batch, text_len, hidden_dim)
		# ([2, 50, 256])

		# Decode mel Spectrogram ([2, 100, 80])
		decoder_outputs, _ = self.decoder_rnn(mel_input) # (batch, mel_len, embedding_dim)
		# ([2, 100, 256])

		mel_output = self.mel_linear(decoder_outputs) # ([2, 100, 80])

		# Stop token prediction
		stop_output = self.stop_token(decoder_outputs) # ([2, 100, 1])
		# print(stop_output.shape) 
		stop_output = stop_output.squeeze(-1) # ([2, 100])
		
		return mel_output, stop_output

```

- 모델이 너무 간단해서 encoder_outputs은 사용되지 않는데, 결과를 좋게 하려면 당연히 decoder에서 encoder를 사용해야 할 것입니다.

## (3) 학습

- 모델을 학습하고(`train_tts`), 추론하여(`infer_tts`) 최종적으로 텍스트에서 mel을 추출합니다. 그리고 mel을 wav로 바꾸어 저장합니다.

```python

if __name__ == "__main__":

	# ... 전처리

	# (28, 128, 256, 80)
    model = SimpleTTS(vocab_size=len(char_to_idx) + 1,
                      embedding_dim=128, hidden_dim=256, mel_dim=mel_dim).cuda()

    # 모델이 저장되어 있으면 그것을 사용
    model_path = "C:\projects\my\simpletts.pth"

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print(f"Model loaded from {model_path}, skipping training.")
    else:
        # 모델이 없으면 새로 훈련
        print("Training model...")
        train_tts(model, dataloader, epochs=10,
                  lr=0.001, criterion=nn.MSELoss())
        # 저장된 모델 사용
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

```

### train_tts

```python
def train_tts(model, dataloader, epochs, lr, criterion):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            text, mel_target, mel_input, stop_target = batch
            
			text, mel_target, mel_input, stop_target = (
                text.cuda(), 
                mel_target.cuda(), 
                mel_input.cuda(), 
                stop_target.cuda()
            )
            
            # 순전파
            mel_output, stop_output = model(text, mel_input)
            
            mel_output = mel_output[:, :mel_target.size(1), :]
            stop_output = stop_output[:, :stop_target.size(1)]
            
            # 손실 함수, nn.MSELoss()
            mel_loss = criterion(mel_output, mel_target)
            stop_loss = criterion(stop_output, stop_target)
            loss = mel_loss + stop_loss
            
            # 역전파
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

		avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
```

```python
@torch.no_grad()
def infer_tts(model, text, char_to_idx, mel_dim, max_length=100):
    model.eval()
    
    # 텍스트를 텐서로 변환 (1, text_length)
    text_tensor = text_to_tensor(text, char_to_idx, max_length).unsqueeze(0).cuda()

	# 초기 입력으로 빈 멜 스펙트로그램 생성 (1, 1, 80)
    mel_input = torch.zeros(1, 1, mel_dim).cuda()
    
    generated_mel = []
    for _ in range(max_length):
		# mel_input 지금까지 생성된 멜 스펙트로그램
        mel_output, stop_output = model(text_tensor, mel_input)
		# mel_step, 다음 타임스텝의 멜 스펙트로그램
        mel_step = mel_output[:, -1, :]
        generated_mel.append(mel_step)
        
        if torch.sigmoid(stop_output[:, -1]) > 0.9:
            break

		# 생성된 mel_step을 mel_input에 추가하여 다음 생성에 활용
        # 자가 회귀 방식으로 멜 스펙트로그램을 점진적으로 생성
        mel_input = torch.cat([mel_input, mel_step.unsqueeze(1)], dim=1)
    
    if len(generated_mel) == 0:
        print("Warning: No mel spectrogram was generated!")
        return torch.zeros(1, mel_dim)
    
    return torch.cat(generated_mel, dim=0)
```

## (4) 결과 확인

- 생성된 mel을 wav로 바꾸어 루트 디렉토리에 저장합니다.

```python
def save_mel_to_wav(mel, sample_rate, filename):
    device = mel.device

    # FFT 관련 파라미터 계산
    n_mels = mel.shape[0]
    n_fft = max(2048, 4 * (n_mels - 1))
    n_stft = n_fft // 2 + 1

    # 멜 스펙트로그램을 파워 스펙트로그램으로 변환
    mel_to_linear_transform = torchaudio.transforms.InverseMelScale(
        n_stft=n_stft,
        n_mels=n_mels,
        sample_rate=sample_rate
    ).to(device)  # Move to the same device as mel

    power_spec = mel_to_linear_transform(mel)

    # 파워 스펙트로그램을 음성 신호로 변환 (Griffin-Lim)
    griffin_lim_transform = torchaudio.transforms.GriffinLim(
        n_fft=n_fft, hop_length=n_fft // 4, win_length=n_fft
    ).to(device)  # Move to the same device as mel

    waveform = griffin_lim_transform(power_spec)

    # 음성 신호를 WAV 파일로 저장
    sf.write(filename, waveform.cpu().numpy(), samplerate=sample_rate)
    print(f"WAV file saved as {filename}")
```

<img src=".\generated_mel.PNG">

## (5) 결과

- 시험으로 생성해본 wav의 경우, 에포크를 10번으로 설정했는데 잡음밖에 들리지 않았다. 아마도 더 많은 훈련이 필요한 것 같다.

- 구현의 편의성을 위해 GRU를 사용했는데, LSTM이나 Transformer를 사용해볼 수도 있을 것 같다.
