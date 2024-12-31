import torch
import torch.nn as nn
from src.CSLR.Encoder import Encoder
from src.CSLR.Attention import Attention
from src.CSLR.Decoder import Decoder
from src.CSLR.Landmark import Landmark

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size, embedding_dim):
        super(Seq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_cnn = Encoder(hidden_size)
        self.encoder_mlp = Landmark(hidden_size=hidden_size)
        self.fusion = nn.Linear(2 * hidden_size, hidden_size)
        self.attention = Attention(hidden_size)
        self.decoder = Decoder(vocab_size, hidden_size, embedding_dim)

    def forward(self, images, landmarks, target_tokens, alignment=None):
        batch_size, num_frames, _, _, _ = images.size()
        target_len = target_tokens.size(1)

        # Encode images and landmarks
        image_features = self.encoder_cnn(images)  # (batch_size, num_frames, hidden_size)
        landmark_features = self.encoder_mlp(landmarks)  # (batch_size, num_frames, hidden_size)

        # Fuse features
        encoder_outputs = self.fusion(torch.cat([image_features, landmark_features], dim=2))  # (batch_size, num_frames, hidden_size)

        # Decoder initialization
        decoder_input = target_tokens[:, 0]  # Start token
        decoder_hidden = (torch.zeros(1, batch_size, self.hidden_size).to(images.device),
                          torch.zeros(1, batch_size, self.hidden_size).to(images.device))

        outputs = torch.zeros(batch_size, target_len, self.decoder.fc.out_features).to(images.device)

        for t in range(1, target_len):
            # Pass alignment to attention
            context, _ = self.attention(encoder_outputs, decoder_hidden[0].squeeze(0), alignment)
            output, decoder_hidden = self.decoder(decoder_input.unsqueeze(1), decoder_hidden, context)
            outputs[:, t] = output.squeeze(1)

            # Teacher forcing
            decoder_input = target_tokens[:, t]

        return outputs