import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" ##an assertion that checks if d_model is divisible by num_heads without any remainder.

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads ## calculates the value of d_k, which is the dimension of each head

        self.W_q = nn.Linear(d_model, d_model) ##creates a linear transformation (a fully connected layer) for the query vectors (Q)
        self.W_k = nn.Linear(d_model, d_model) ## creates a linear transformation for the key vectors (K)
        self.W_v = nn.Linear(d_model, d_model) ## creates a linear transformation for the value vectors (V)
        self.W_o = nn.Linear(d_model, d_model) ##creates a linear transformation for the output of the multi-head attention mechanism

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) ## control the magnitude of the attention scores.
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9) ##apply mask, to zero out attention scores for positions that should be masked
        attn_probs = torch.softmax(attn_scores, dim=-1) ## passed through a softmax function to obtain the attention probabilities, ensuring that the probabilities sum to 1 for each query
        output = torch.matmul(attn_probs, V) ##compute a weighted sum of the values V
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) ## reshapes the input tensor x to have an additional two dimensions and swaps the second and third dimensions

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model) ##transposes the dimensions of the tensor, reshapes the tensor into a new shape

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V)) ##apply linear transformations to the input queries, keys, and values.

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask) ##computes the scaled dot-product attention between the transformed queries, keys, and values
        output = self.W_o(self.combine_heads(attn_output)) ## concatenated the attention outputs from different heads and then linearly transformed again
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff) ##takes the input of dimension d_model and maps it to an intermediate dimension of d_ff
        self.fc2 = nn.Linear(d_ff, d_model) ##maps the intermediate dimension d_ff back to the original dimension d_model
        self.relu = nn.ReLU() ##the activation function after the first linear layer

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x))) ##applies the first linear transformation, mapping the input x from d_model to d_ff, then  applies the Rectified Linear Unit (ReLU) activation function element-wise to the output of the first linear layer
                                                ## Finally, the output of the ReLU activation is passed through the second linear layer, mapping it back to the original dimension d_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model) ##creates a matrix pe of shape (max_seq_length, d_model) initialized with zeros
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1) ##creates a tensor position representing the positions of elements in the sequence, ranging from 0 to max_seq_length - 1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)) ##  calculates a div_term tensor used for the positional encoding calculations

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) ##calculates and assigns the sine and cosine values for each position in the pe matrix

        self.register_buffer('pe', pe.unsqueeze(0)) ## registers the pe matrix as a buffer

    def forward(self, x):
        return x + self.pe[:, :x.size(1)] ## retrieves the pe buffer and selects the relevant portion of positional encodings for the input tensor x based on its length (up to x.size(1)), then adds the selected positional encodings to the input tensor x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) ## layer normalization modules, are applied before and after the self-attention and feed-forward sub-layers, respectively.
        self.dropout = nn.Dropout(dropout) ## an instance of a dropout layer, which is applied after each sub-layer.

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask) ##calculates the self-attention output using the multi-head self-attention mechanis
        x = self.norm1(x + self.dropout(attn_output)) ## updated by adding the output of self-attention after layer normalization and dropout
        ff_output = self.feed_forward(x) ##calculates the output of the position-wise feed-forward network
        x = self.norm2(x + self.dropout(ff_output)) ##updated by adding the feed-forward output after layer normalization and dropout.
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask) ##computes self-attention for the input x with the target mask applied
        x = self.norm1(x + self.dropout(attn_output)) ## normalized using Layer Normalization and residual connection
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask) ##computes cross-attention between the input x and the encoder's output
        x = self.norm2(x + self.dropout(attn_output)) ##normalized and added to the previous output using a residual connection
        ff_output = self.feed_forward(x) ##x goes through a position-wise feed-forward network
        x = self.norm3(x + self.dropout(ff_output)) ##output is normalized and added to the previous output using a residual connection.
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt): ##generates masks for the source and target sequences
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2) ## mask for the source sequence to mark padding tokens (zeros)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3) ## mask for the target sequence to prevent future information leakage during self-attention
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt) ##generates source and target masks using the generate_mask method
        ##applies embedding layers to the source and target sequences and adds positional encoding
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask) ##source sequence is processed through the encoder layers, used multi-head self-attention and feed-forward sub-layers

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask) ## target sequence is processed through the decoder layers, used multi-head self-attention and feed-forward sub-layers

        output = self.fc(dec_output) ## passed through a linear layer to predict the target sequence
        return output

src_vocab_size = 5000
tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 100
dropout = 0.1

transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)

# Generate random sample data
src_data = torch.randint(1, src_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)
tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

for epoch in range(100):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")