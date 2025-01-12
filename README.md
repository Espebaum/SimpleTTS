# 프로젝트 결과 보고서

![link](https://github.com/Espebaum/SimpleTTS)

## (1) 데이터 전처리 방법

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
dataset = TTSDataset(metadata_path, wav_dir, char_to_idx, \
                         max_text_length, max_mel_length, mel_dim)

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
```