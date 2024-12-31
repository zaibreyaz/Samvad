import torch.nn as nn

class Landmark(nn.Module):
    def __init__(self, input_size=63, hidden_size=256):
        super(Landmark, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
        )

    def forward(self, landmarks):
        # Flatten landmarks: (batch_size, num_frames, 21, 3) -> (batch_size, num_frames, 63)
        batch_size, num_frames, _, _ = landmarks.size()
        landmarks = landmarks.view(batch_size, num_frames, -1)
        return self.mlp(landmarks)