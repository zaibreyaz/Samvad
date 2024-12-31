import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim + hidden_size, hidden_size, batch_first=True)  # Adjust input size
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_token, hidden, context):
        # Ensure input_token is shaped correctly
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)  # (batch_size, 1)

        # Embed the input token
        embedded = self.embedding(input_token)  # (batch_size, 1, embedding_dim)

        # Adjust context shape if necessary
        if context.dim() == 1:  # If context is a vector of shape (hidden_size,)
            context = context.unsqueeze(0)  # Add batch dimension: (1, hidden_size)

        context = context.unsqueeze(1)  # Add time dimension: (batch_size, 1, hidden_size)

        # Concatenate embedded token and context
        lstm_input = torch.cat([embedded, context], dim=2)  # (batch_size, 1, embedding_dim + hidden_size)

        # LSTM forward pass
        output, hidden = self.lstm(lstm_input, hidden)  # Output: (batch_size, 1, hidden_size)

        # Project to vocabulary size
        output = self.fc(output)  # (batch_size, 1, vocab_size)

        return output, hidden