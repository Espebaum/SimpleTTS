import torch
import torch.optim as optim
from tqdm import tqdm

from preprocess import text_to_tensor

@torch.no_grad()
def infer_tts(model, text, char_to_idx, mel_dim, max_length=100):
    model.eval()
    
    # Convert text to tensor
    text_tensor = text_to_tensor(text, char_to_idx, max_length).unsqueeze(0).cuda()
    mel_input = torch.zeros(1, 1, mel_dim).cuda()  # Start of sequence (SOS)
    
    generated_mel = []
    for _ in range(max_length):
        mel_output, stop_output = model(text_tensor, mel_input)
        mel_step = mel_output[:, -1, :]  # Last time step
        generated_mel.append(mel_step)
        
        # Check stop token with a reasonable threshold
        if torch.sigmoid(stop_output[:, -1]) > 0.9:  # Adjust threshold if necessary
            break
        
        mel_input = torch.cat([mel_input, mel_step.unsqueeze(1)], dim=1)
    
    # Ensure generated_mel is not empty
    if len(generated_mel) == 0:
        print("Warning: No mel spectrogram was generated!")
        return torch.zeros(1, mel_dim)  # Return a dummy mel spectrogram
    
    return torch.cat(generated_mel, dim=0)

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
            
            # Calculate losses
            mel_loss = criterion(mel_output, mel_target)
            stop_loss = criterion(stop_output, stop_target)
            loss = mel_loss + stop_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")