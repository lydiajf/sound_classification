from pathlib import Path
import pandas as pd
import torch 
import torchaudio
import torch.nn as nn
from tqdm import tqdm
# from transformers import GPT2Config, GPT2Tokenizer
import whisper
from models.models import Transformer
import numpy as np

# i want it to be split into batches
# audio class

from datasets import load_dataset

ds = load_dataset("danavery/urbansound8K", cache_dir="./data")['train']


# ds_train = ds[:70%]
# ds_val = ds[70%:90%]
# ds_test = ds[90%:100%]

# Preprocess and Save Function
def preprocess(dataset, save_dir, target_sample_rate=16000, max_duration=10):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    spectrograms = []
    labels = []
    
    # Initialize Whisper's preprocessing
    mel_spectrogram_transform = whisper.log_mel_spectrogram
    
    max_length = target_sample_rate * max_duration  # Max length in samples
    
    for example in tqdm(dataset, desc="Preprocessing Audio"):
        audio_array = example['audio']['array']
        label = example['classID'] 
        
        audio_array = torch.tensor(audio_array).float()  # Shape: [1, num_samples]

        # if sample_rate != target_sample_rate:
        #     resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate, dtype=waveform.dtype)
        #     waveform = resampler(waveform)
        
        # # Trim or pad waveform
        # if waveform.size(1) < max_length:
        #     padding = max_length - waveform.size(1)
        #     waveform = torch.nn.functional.pad(waveform, (0, padding))
        # else:
        #     waveform = waveform[:, :max_length]
        
        # # Generate log-Mel spectrogram using Whisper's method
        
        # mel_spectrogram_transform = mel_spectrogram_transform.to(dtype=waveform.dtype)

        audio_array = whisper.pad_or_trim(audio_array)
        mel = mel_spectrogram_transform(audio_array)
        
        # Optionally normalize spectrogram
        # mel = (mel - mel.mean()) / mel.std()
        
        spectrograms.append(mel)
        labels.append(label)
    
    return spectrograms, labels
    # Save preprocessed data
    # torch.save({'spectrograms': spectrograms, 'labels': labels}, save_dir / 'preprocessed_data.pt')


# # Load Preprocessed Data Function
# def load_preprocessed_data(save_dir):
#     data = torch.load(Path(save_dir) / 'preprocessed_data.pt')
#     return data['spectrograms'], data['labels']

 
if __name__ == '__main__':
    # Parameters
    
    save_dir = 'preprocessed_data'
    batch_size = 32
    learning_rate = 1e-4
    epochs = 10
    num_classes = 10  # Number of classes
    emb_dim = 256  # Embedding dimension
    num_heads = 4
    hidden_dim_ff = 128
    num_encoder_layers = 6  # Number of encoder layers
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device being used',device)

    # Load the preprocessed data
    spectrograms, labels = preprocess(ds, save_dir)
    
    # Prepare data for DataLoader
    class AudioDataset(torch.utils.data.Dataset):
        def __init__(self, spectrograms, labels):
            self.spectrograms = spectrograms
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            mel = self.spectrograms[idx]
            label = self.labels[idx]
            # Adjust mel shape if necessary
            # Current shape: [80, time_steps]
            # For Conv1d, we need shape: [channels, time_steps]
            return mel, label
    
    # Create Dataset and DataLoader
    dataset = AudioDataset(spectrograms, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model
    model = Transformer(
        emb_dim=emb_dim,
        num_heads=num_heads,
        hidden_dim_ff=hidden_dim_ff,
        num_encoder_layers=num_encoder_layers,
        num_classes=num_classes
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for mel, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            mel = mel.to(device)
            labels = labels.to(device)
            
            # Adjust mel shape for Conv1d: [batch_size, channels, time_steps]
            mel = mel.squeeze(1)  # Remove singleton dimension if present
            # mel = mel.permute(0, 2, 1)  # From [batch_size, n_mels, time_steps] to [batch_size, time_steps, n_mels]
            # mel = mel.transpose(1, 2)  # Now [batch_size, n_mels, time_steps]
            
            optimizer.zero_grad()

            # outputs = model(mel)

            # Forward pass
            class_logits = model(mel)

            loss = criterion(class_logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'models/urban_sound_model.pth')

model.eval()