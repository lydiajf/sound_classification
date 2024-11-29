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
import wandb

# i want it to be split into batches
# audio class

from datasets import load_dataset

ds = load_dataset("danavery/urbansound8K", cache_dir="./data")['train']

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    
    # Splits the dataset into training, validation, and test sets.

    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    ds_train = dataset[:train_size]
    ds_val = dataset[train_size:train_size + val_size]
    ds_test = dataset[train_size + val_size:]

    return ds_train, ds_val, ds_test

# Preprocess and Save Function
def preprocess(dataset, save_dir, target_sample_rate=16000, max_duration=10):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    spectrograms = []
    labels = []
    
    # Initialize Whisper's preprocessing
    mel_spectrogram_transform = whisper.log_mel_spectrogram
    
    max_length = target_sample_rate * max_duration  # Max length in samples
    
    audio = dataset['audio']
    classID = dataset['classID']

    for i, record in tqdm(enumerate(audio), desc="Preprocessing Audio"):
        # audio_array = example['audio']['array']
        # label = example['classID']
        audio_array = record['array']
        label = classID[i]
        
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

    ds_train, ds_val, ds_test = split_dataset(ds)

    # Load the preprocessed data
    spectrograms, labels = preprocess(ds_train, save_dir)
    
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

    wandb.init(project='urban-sound-classification', config={
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_classes": num_classes,
    "emb_dim": emb_dim,
    "num_heads": num_heads,
    "hidden_dim_ff": hidden_dim_ff,
    "num_encoder_layers": num_encoder_layers,
    })
    
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

            wandb.log({'batch_loss': loss.item(), 'epoch': epoch+1})
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'models/urban_sound_model_with_splits.pth')


    spectrograms_val, labels_val = preprocess(ds_val, save_dir)

    model.eval()

    with torch.no_grad():  # Disable gradient calculation for evaluation
        correct_predictions = 0
        total_samples = 0
        all_preds = []
        all_labels = []

        for mel, label in zip(spectrograms_val, labels_val):
            mel = mel.unsqueeze(0)  # Add batch dimension if necessary
            output = model(mel)  # Forward pass

            # Assuming you have a way to get predicted class from output
            predicted = torch.argmax(output, dim=1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

        accuracy = correct_predictions / total_samples
        print(f"Validation Accuracy: {accuracy:.4f}")

        # Log validation loss and accuracy to wandb
        wandb.log({'val_accuracy': accuracy, 'epoch': epoch+1})

        # Create a table of predictions and true labels
        table = wandb.Table(columns=['Predicted', 'True'])
        for pred, true_label in zip(all_preds, all_labels):
            table.add_data(pred, true_label)
        # Log the table to wandb
        wandb.log({'predictions': table})