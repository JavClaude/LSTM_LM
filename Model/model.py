import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class LSTMModel(torch.nn.Module):
    def __init__(self,
                vocab_size: int,
                embedding_dim: int,
                hidden_units: int,
                n_layers: int,
                bidirectional: bool,
                dropout_rnn: float,
                dropout: float,
                **kwargs):
        super(LSTMModel, self).__init__()

        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.total_hidden = hidden_units*2 if bidirectional else hidden_units
        self.bidirectional = bidirectional

        self.Embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.D1 = torch.nn.Dropout(dropout)
        self.LN = torch.nn.LayerNorm(embedding_dim)
        self.Encoder = torch.nn.LSTM(embedding_dim, hidden_units, num_layers=n_layers, batch_first=True, dropout=dropout_rnn, bidirectional=bidirectional)
        if embedding_dim != self.total_hidden:
            self.linear_proj = torch.nn.Linear(embedding_dim, self.total_hidden)
        self.relu = torch.nn.ReLU()
        self.LN2 = torch.nn.LayerNorm(self.total_hidden)
        self.D2 = torch.nn.Dropout(dropout)
        self.Decoder = torch.nn.Linear(self.total_hidden, vocab_size)

    def forward(self, sequence, hiddens_states):
        x_embedded = self.Embedding(sequence)
        x = self.D1(x_embedded)
        x = self.LN(x)
        hiddens, hiddens_states = self.Encoder(x, hiddens_states)
        ## Residual connection ##
        if self.embedding_dim != self.hidden_units:
            x = torch.add(
                self.relu(self.linear_proj(x_embedded)), hiddens
            )
        else:
            x = torch.add(self.relu(x_embedded), hiddens)
        x = self.LN2(x)
        logits = self.D2(x)
        logits = self.Decoder(logits)
        return logits, hiddens_states
    
    def init_hiddens(self, batch_size):
        return (
            torch.zeros(size=(self.n_layers * 2 if self.bidirectional else self.n_layers, batch_size, self.hidden_units), device=device),
            torch.zeros(size=(self.n_layers * 2 if self.bidirectional else self.n_layers, batch_size, self.hidden_units), device=device)
        )
