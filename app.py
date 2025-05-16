import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import base64
from io import BytesIO

# Initialize Flask app
app = Flask(__name__)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Directories
EMBEDDING_DIR = 'embeddings'
MODEL_PATH = 'best_facenet_model_manhattan_9830.pth'
if not os.path.exists(EMBEDDING_DIR):
    os.makedirs(EMBEDDING_DIR)

# Define embeddings model
class FaceNetEmbedding(nn.Module):
    def __init__(self):
        super(FaceNetEmbedding, self).__init__()
        self.model = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.projector = nn.Linear(512, 128)
        self.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    def forward(self, x):
        x = self.model(x)          # [batch_size, 512]
        x = self.projector(x)      # [batch_size, embedding_dim]
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x

# Load mtcnn model
mtcnn = MTCNN(image_size=160, margin=0)
# Load the embeddings model
facenet = FaceNetEmbedding()
# Load the saved weights
facenet.eval()

# Helper: Convert base64 to PIL image
def readb64(base64_string):
    img_data = base64.b64decode(base64_string.split(',')[1])
    img = Image.open(BytesIO(img_data)).convert('RGB')
    return img

# Helper: Calculate manhattan similarity
def manhattan_similarity(a, b):
    a = a.numpy()
    b = b.numpy()
    distance = np.sum(np.abs(a - b))
    # sigmoid function to convert distance to similarity
    similarity = 1 / (1 + np.exp(8 * (distance - 0.7)))
    return distance, similarity

# Helper: Get all embeddings
def load_embeddings():
    database = {}
    for file in os.listdir(EMBEDDING_DIR):
        name = file.split('.')[0]
        embedding = torch.load(os.path.join(EMBEDDING_DIR, file))
        database[name] = embedding
    return database

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/save_face', methods=['POST'])
def save_face():
    name = request.form['name']
    img_data = request.form['image']
    img = readb64(img_data)
    face = mtcnn(img)
    if face is not None:
        embedding = facenet(face.unsqueeze(0)).detach().squeeze()
        embedding = F.normalize(embedding, p=1, dim=0)
        torch.save(embedding, os.path.join(EMBEDDING_DIR, f"{name}.pt"))
        return jsonify({"status": "success"})
    return jsonify({"status": "fail", "reason": "No face detected"})

@app.route('/recognize', methods=['POST'])
def recognize():
    img_data = request.form['image']
    img = readb64(img_data)
    face = mtcnn(img)
    if face is not None:
        embedding = facenet(face.unsqueeze(0)).detach().squeeze()
        embedding = F.normalize(embedding, p=1, dim=0)
        database = load_embeddings()
        max_sim = -1
        identity = "Unknown"
        for name, db_emb in database.items():
            dist, sim = manhattan_similarity(embedding, db_emb)
            print(f"Comparing with {name}: dist={dist:.4f}, sim={sim:.4f}")
            if sim > max_sim:
                max_sim = sim
                identity = name if sim > 0.7 else "Unknown"
        return jsonify({"name": identity, "similarity": float(max_sim)})
    return jsonify({"name": "No face", "similarity": 0})

if __name__ == '__main__':
    app.run(debug=True)