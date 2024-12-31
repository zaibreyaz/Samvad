import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, encoder_outputs, hidden_state, alignment=None):
        batch_size, num_frames, hidden_size = encoder_outputs.size()

        # If alignment is provided, mask attention to focus only on aligned frames
        if alignment is not None:
            mask = alignment.unsqueeze(-1).float()  # Shape: (batch_size, num_frames, 1)
            masked_outputs = encoder_outputs * mask
        else:
            masked_outputs = encoder_outputs

        # Compute attention weights
        hidden_state = hidden_state.unsqueeze(1).expand(-1, num_frames, -1)
        energy = self.attention(torch.tanh(masked_outputs + hidden_state))
        attention_weights = F.softmax(energy, dim=1)

        context = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context, attention_weights