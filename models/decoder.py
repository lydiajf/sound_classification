import torch
import numpy as np
from transformers import GPT2Tokenizer

GPT_CONFIG_124M = {
    "vocab_size": 50300,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

def getPositionEncoding(batch_size, seq_len, d, n=10000):
    P = np.zeros((batch_size, seq_len, d))  # Adjusted to include batch size
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            batch_indices = np.arange(batch_size)  # Shape (batch_size,)
            P[:, k, 2 * i] = np.sin(batch_indices / denominator)  # Broadcasting for batch size
            P[:, k, 2 * i + 1] = np.cos(batch_indices / denominator)
    return torch.tensor(P, dtype=torch.float32)  # Convert to PyTorch tensor

    
class Decoder2(torch.nn.Module):
    def __init__(self, vocab_size, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.masked_attn = MaskedAttention(Wemb_dim, num_heads, hidden_dim_ff)
        self.cross_attn = CrossAttention(Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, vocab_size)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(Wemb_dim, hidden_dim_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_ff, Wemb_dim)
        )
    
    def forward(self, wemb, pemb):

        # No positional encoding needed for image embeddings (Pemb)
        print("The Pemb shape:", pemb.shape) 

        word_emb = self.masked_attn(wemb)
        cross_emb = self.cross_attn(word_emb, pemb)

        out = self.ff(cross_emb)
        # add & normalize
        return out
    
class MaskedAttention(torch.nn.Module):
    def __init__(self, emb_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        # print(self.head_dim)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.linear_q = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_k = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_v = torch.nn.Linear(emb_dim, emb_dim)
        
        self.linear_concat = torch.nn.Linear(emb_dim, emb_dim)

        self.norm = torch.nn.LayerNorm(emb_dim)
        # Learnable bias for attention
        # self.attn_embedding_bias = torch.nn.Parameter(torch.zeros(emb_dim))
        

    def forward(self, emb):

        # Fix: Get dimensions correctly using size()
        # seq_len, embed_dim
        batch_size = emb.size(0)
        seq_len = emb.size(1)
        
        # Transform embeddings for query, key, and value
        query = self.linear_q(emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(emb).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores and apply softmax
        scaling_factor = self.head_dim ** 0.5
        similarity_matrix = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply upper triangular mask (if required for causality)
        mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1) * -1e9
        similarity_matrix = similarity_matrix + mask 

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(similarity_matrix, dim=-1)
    
        # Apply attention weights to values and reshape back
        attention = torch.matmul(soft_matrix, value)
        attention = attention.transpose(1, 2).contiguous()
        attn_emb = attention.view(batch_size,seq_len, -1)  # Reshape

        attn_emb = self.linear_concat(attn_emb)

        attn_emb = self.norm(attn_emb + emb)

        # add residual and normalisation
        
        return attn_emb

class CrossAttention(torch.nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size):
        super().__init__()
        self.num_heads = num_heads
        self.Whead_dim = new_dim // num_heads  # Embedding dimension for words per head
        self.Phead_dim = new_dim // num_heads  # Embedding dimension for images per head

        assert Wemb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert Pemb_dim % num_heads == 0, "Embedding dimension for images must be divisible by the number of heads"

        self.embeddings = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)

        # Linear layers for query, key, and value transformations
        self.linear_q = torch.nn.Linear(Wemb_dim, new_dim)
        self.linear_k = torch.nn.Linear(Pemb_dim, new_dim)
        self.linear_v = torch.nn.Linear(Pemb_dim, new_dim)

        # Linear layer for the concatenated output
        self.linear_concat = torch.nn.Linear(new_dim, Wemb_dim)

        self.norm = torch.nn.LayerNorm(Wemb_dim)

    def forward(self, wemb, pemb):
        # wemb: [batch_size, seq_len_w, Wemb_dim]
        # pemb: [batch_size, seq_len_p, Pemb_dim]
        # batch_size,seq_len, wemb_dim
        # No positional encoding needed for image embeddings (Pemb)
        batch_size = wemb.size(0)
        batch_size = pemb.size(0)
        Wseq_len = wemb.size(1)
        Pseq_len = pemb.size(1)
        
     
        print("The Pemb shape:", pemb.shape) 

        # Transform embeddings for query, key, and value
        query = self.linear_q(wemb).view(batch_size, Wseq_len, self.num_heads, self.Whead_dim).transpose(1, 2)
        key = self.linear_k(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)
        value = self.linear_v(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)

        print("Query shape after linear transformation:", query.shape)
        print("Key shape after linear transformation:", key.shape)
        print("Value shape after linear transformation:", value.shape)

        # Attention computation: query * key^T
        scaling_factor = self.Whead_dim ** 0.5  # or use self.Phead_dim if necessary
        attention = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(attention, dim=-1)

        # Attention output
        # matmul replacement
        sim_mat = soft_matrix @ value  
        sim_mat = sim_mat.transpose(1, 2).contiguous()
        final_emb = sim_mat.view(batch_size, Wseq_len, -1)  # Reshape to (Wseq_len, num_heads * Whead_dim)

        # Pass through the linear layer after concatenation
        final_emb = self.linear_concat(final_emb)

        final_emb = self.norm(final_emb + wemb)
        # add residual and normalization

        return final_emb


#%%
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    # Testing
    args = (4000, 64,128, 128, 4,6000)  # (vocab size, embedding size, num heads, FFN hidden dim)
    # self, vocabsize, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size)
    model = Decoder2(*args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Random input tensor with integer values within embedding vocab range
    pemb = torch.rand((2,600,128)).to(device)  # Integer values for embedding lookup
    wemb = torch.rand(2,10, 64).to(device)


    print(pemb.shape)
    print(wemb.shape)
    output = model(wemb,pemb)
    print("Output shape:", output.shape)