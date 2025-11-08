import torch
import torch.nn as nn

activation_funcs = {
    'sigmoid': nn.Sigmoid(),
    'relu': nn.ReLU(),
    'tanh': nn.Tanh()
}

class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=64,
                 num_layers=2, dropout=0.5, cell_type="RNN",
                 bidirectional=False, activation='sigmoid'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        cell_map = {
            'RNN': nn.RNN,
            'LSTM': nn.LSTM,
            'Bidirectional LSTM': nn.LSTM
        }
        self.cell_type = cell_type
        self.bidirectional = (cell_type == 'Bidirectional LSTM')
        self.rnn = cell_map[cell_type](embed_size, hidden_size,
                                      num_layers=num_layers,
                                      dropout=dropout,
                                      batch_first=True,
                                      bidirectional=self.bidirectional)
        direction_mult = 2 if self.bidirectional else 1
        self.fc = nn.Linear(hidden_size * direction_mult, 1)
        self.output_activation = nn.Sigmoid()
        self.hidden_activation = activation_funcs[activation]

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        # Apply hidden activation to last output step before fc
        out = self.hidden_activation(out[:, -1, :])
        out = self.fc(out)
        out = self.output_activation(out).squeeze(1)
        return out

def get_model(config):
    return SentimentRNN(
        vocab_size=config['vocab_size'],
        embed_size=config['embed_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        cell_type=config['cell_type'],
        bidirectional=config.get('bidirectional', False),
        activation=config.get('activation', 'sigmoid')
    )