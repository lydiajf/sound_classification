import torch
import numpy as np

from models.encoder import Encoder
from models.decoder import Decoder2, MaskedAttention, CrossAttention, getPositionEncoding

# todo :
# normalisation of layers
# batching

class AudioFeatureExtractor(torch.nn.Module):
    def __init__(self, emb_dim):
        super(AudioFeatureExtractor, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=80, out_channels=emb_dim, kernel_size=3, padding=1)
        self.gelu1 = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(in_channels=emb_dim, out_channels=emb_dim, kernel_size=3, padding=1,stride=2)
        self.gelu2 = torch.nn.GELU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.gelu2(x)
        return x

class Transformer(torch.nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        hidden_dim_ff,
        # Wemb_dim,
        # Pemb_dim,
        # new_dim,
        # voc_size,
        num_encoder_layers=6,
        # num_decoder_layers=6,
        num_classes=10
    ):
        super().__init__()
        self.feature_extractor = AudioFeatureExtractor(emb_dim)

        # self.word_embedding = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)
        # Stacking encoder layers
        self.encoders = torch.nn.ModuleList(
            [Encoder(emb_dim, num_heads, hidden_dim_ff) for _ in range(num_encoder_layers)]
        )

        # Stacking decoder layers
        # self.decoders = torch.nn.ModuleList(
        #     [Decoder2(voc_size, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff) for _ in range(num_decoder_layers)]
        # )

        # self.project = torch.nn.Linear(Wemb_dim, voc_size)
        self.classifier = torch.nn.Linear(emb_dim, num_classes)
    
    
    def forward(self, audio_input):
        # Pass through all encoder layers
        #  = torch.nn.Linear(pxl_size,emb_dim)
        # encoder_output = self.linear(encoder_output)
        x = self.feature_extractor(audio_input)  # x shape: [batch_size, emb_dim, time_steps]
        x = x.permute(0, 2, 1)  # x shape: [batch_size, time_steps, emb_dim]

        # Positional Encoding
        batch_size, seq_len, emb_dim = x.size()
        pos_enc = getPositionEncoding(batch_size, seq_len, emb_dim)
        encoder_input = x + pos_enc

        # Pass through encoder layers
        encoder_output = encoder_input
        for i, encoder in enumerate(self.encoders):
            encoder_output = encoder(encoder_output)
            print(f"Encoder Layer {i + 1} output shape:", encoder_output.shape)

        # # Pass through all decoder layers
        # wemb = self.word_embedding(word_inpt)
        # # Positional encoding for word embeddings
        # batch_size, Wseq_len, Wd = wemb.size(0), wemb.size(1), wemb.size(2)
        # Wsin_emb = getPositionEncoding(batch_size, Wseq_len, Wd)
        # print('The Wemb after adding positional encoding:', Wsin_emb.shape)
        # wemb = wemb + Wsin_emb
        # print('the output of wemb after embedding',wemb.shape)
        
        # for i, decoder in enumerate(self.decoders):
        #     wemb = decoder(wemb, encoder_output)
        #     print(f"Decoder Layer {i + 1} output shape:", wemb.shape)

        # prediction = self.project(wemb)
        pooled_output = encoder_output.mean(dim=1)
        class_logits = self.classifier(pooled_output)


        return class_logits



# omars test

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define dimensions
    # if emb_dim chnaged you need to change Pemb_dim aswell
    pxl_size = 784
    emb_dim = 256
    num_heads = 4
    hidden_dim_ff = 512
    Wemb_dim = 128
    Pemb_dim = 256
    new_dim = 256
    voc_size = 1000
    num_encoder_layers = 20
    num_decoder_layers = 20

    # Create model
    model = Transformer(
        pxl_size,
        emb_dim,
        num_heads,
        hidden_dim_ff,
        Wemb_dim,
        Pemb_dim,
        new_dim,
        voc_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    # # Dummy input data before batching
    # pxl_input = torch.rand((600, 784))  # Batch of 32 pixel inputs
    # print(pxl_input.shape)
    # wemb = torch.randint(0, voc_size, (32,))  # Batch of 32 word indices

    # Dummy input data
    pxl_input = torch.rand((2, 600, 784)).to(device)  # Batch of 32 pixel inputs
    print(pxl_input.shape)
    wemb = torch.rand(2, 600).to(device) # Batch of 32 word indices

    # Forward pass
    output = model(pxl_input, wemb)
    print("Final Transformer output shape:", output)