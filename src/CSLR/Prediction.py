import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
import mediapipe as mp
import torch.nn.functional as F
from src.CSLR.Seq2Seq import Seq2Seq


class VideoPrediction:
    def __init__(self, vocab, model_checkpoint):
        """
        Initialize the VideoPrediction class.

        Args:
            vocab (dict): Vocabulary mapping.
            model_checkpoint (str): Path to trained model checkpoint.
        """
        self.vocab = vocab
        self.model_checkpoint = model_checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.image_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.hands_model = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2, min_detection_confidence=0.25
        )
        self.no_hand_landmarks = [{"x": -1.0, "y": -1.0, "z": -1.0} for _ in range(21)]

    def extract_frames(self, video_path, output_folder):
        """
        Extract frames from a video and save them as images in the output folder.

        Args:
            video_path (str): Path to the input video file.
            output_folder (str): Path to the folder where frames will be saved.
        """
        os.makedirs(output_folder, exist_ok=True)
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        frame_count = 0
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
        video_capture.release()
        print(f"Extracted {frame_count} frames to '{output_folder}'")

    def extract_landmarks(self, image_path):
        """Extract hand landmarks from an image."""
        image = cv2.imread(image_path)
        if image is None:
            return self.no_hand_landmarks

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.hands_model.process(rgb_image)
        if result.multi_hand_landmarks:
            return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in result.multi_hand_landmarks[0].landmark]

        return self.no_hand_landmarks

    def preprocess_frames_and_landmarks(self, frames_folder):
        """
        Preprocess frames and extract landmarks.

        Args:
            frames_folder (str): Path to folder containing video frames.

        Returns:
            tuple: (preprocessed images tensor, preprocessed landmarks tensor)
        """
        image_tensors = []
        landmark_tensors = []
        for filename in sorted(os.listdir(frames_folder)):
            if filename.endswith((".jpg", ".png")):
                image_path = os.path.join(frames_folder, filename)
                # Process image
                image = Image.open(image_path).convert("RGB")
                image_tensor = self.image_transforms(image)
                image_tensors.append(image_tensor)
                # Extract landmarks
                landmarks = self.extract_landmarks(image_path)
                landmarks_tensor = torch.tensor(
                    [[lm["x"], lm["y"], lm["z"]] for lm in landmarks], dtype=torch.float32
                )
                landmark_tensors.append(landmarks_tensor)

        return (
            torch.stack(image_tensors).unsqueeze(0),
            torch.stack(landmark_tensors).unsqueeze(0),
        )

    def decode_sentence(self, predicted_tokens):
        """Convert a list of token indices into a readable sentence."""
        index_to_token = {idx: token for token, idx in self.vocab.items()}
        sentence = [index_to_token.get(token, "<UNK>") for token in predicted_tokens]
        return " ".join(sentence).replace("<SOS>", "").replace("<EOS>", "").strip()

    def infer(self, frames_folder):
        """
        Perform inference on a sequence of video frames.

        Args:
            frames_folder (str): Path to folder containing video frames.

        Returns:
            str: Decoded sentence.
        """
        # Preprocess frames and landmarks
        images, landmarks = self.preprocess_frames_and_landmarks(frames_folder)
        images, landmarks = images.to(self.device), landmarks.to(self.device)

        # Load the model
        model = Seq2Seq(vocab_size=len(self.vocab), hidden_size=256, embedding_dim=256)
        model.load_state_dict(torch.load(self.model_checkpoint, map_location=self.device))
        model = model.to(self.device)
        model.eval()

        # Initialize inference variables
        sos_token = self.vocab["<SOS>"]
        eos_token = self.vocab["<EOS>"]
        max_length = 4
        predicted_sentence = []

        with torch.no_grad():
            decoder_input = torch.tensor([[sos_token]], device=self.device)
            decoder_hidden = (torch.zeros(1, 1, model.hidden_size, device=self.device),
                              torch.zeros(1, 1, model.hidden_size, device=self.device))

            # Encode inputs
            image_features = model.encoder_cnn(images)
            landmark_features = model.encoder_mlp(landmarks)
            encoder_outputs = model.fusion(torch.cat([image_features, landmark_features], dim=2))

            for _ in range(max_length):
                context, _ = model.attention(encoder_outputs, decoder_hidden[0].squeeze(0))
                output, decoder_hidden = model.decoder(decoder_input, decoder_hidden, context)
                output = F.softmax(output, dim=2)
                token = output.argmax(2).item()
                predicted_sentence.append(token)
                if token == eos_token:
                    break
                decoder_input = torch.tensor([[token]], device=self.device)

        return self.decode_sentence(predicted_sentence)

    def predict(self, video_path):
        """
        Predict the sentence from a video.

        Args:
            video_path (str): Path to the input video.

        Returns:
            str: Predicted sentence.
        """
        frames_folder = "static/extracted_frames"
        self.extract_frames(video_path, frames_folder)
        return self.infer(frames_folder)