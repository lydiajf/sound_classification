import torch
from models.models import Transformer

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


model = Transformer(
    emb_dim=emb_dim,
    num_heads=num_heads,
    hidden_dim_ff=hidden_dim_ff,
    num_encoder_layers=num_encoder_layers,
    num_classes=num_classes
).to(device)

model.load_state_dict(torch.load('urban_sound_model.pth', map_location=torch.device('cpu')))

print(model)