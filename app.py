from flask import Flask, request, jsonify, render_template
import os
from werkzeug.utils import secure_filename
from src.CSLR.Prediction import VideoPrediction
      
app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload-video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        vocab = {
            '<SOS>': 0, '<EOS>': 1, '<PAD>': 2, '<UNK>': 3, 'you': 4, 'hiding': 5, 
            'something': 6, 'water': 7, 'bring': 8, 'me': 9, 'are': 10, 'for': 11, 
            'do': 12, 'a': 13, 'favour': 14
        }
        
        model_checkpoint = "src/models/epoch_25.pth"

        video_predictor = VideoPrediction(vocab, model_checkpoint)
        predicted_sentence = video_predictor.predict(filepath)
        print("Predicted Sentence:", predicted_sentence)
        
        return jsonify({"message": predicted_sentence}), 200

    return jsonify({"error": "Invalid file type"}), 400

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
