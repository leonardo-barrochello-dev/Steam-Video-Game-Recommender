from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from model import build_and_compile_model
import json
import os
import kagglehub

app = FastAPI(title="Steam Recommender ML API")

KAGGLE_DATASET = "antonkozyriev/game-recommendations-on-steam"

# --- Startup Data Loading & Caching ---
vocab_path = os.path.join(os.path.dirname(__file__), 'tag_vocabulary.json')

# Load Vocabulary
if not os.path.exists(vocab_path):
    raise RuntimeError(f"Vocabulary file not found at {vocab_path}. Please run train.py first.")

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
tag_to_idx = {tag: idx for idx, tag in enumerate(vocab)}

# Load Game Metadata from Kaggle Hub (uses local cache after first download)
print("Caching game tag vectors for real-time inference...")
dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
meta_path = os.path.join(dataset_path, 'games_metadata.json')

app_tags = {}
if os.path.exists(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                app_id = item.get('app_id')
                tags = item.get('tags', [])

                # Create multi-hot vector
                vec = np.zeros(len(vocab), dtype=np.float32)
                for t in tags:
                    if t in tag_to_idx:
                        vec[tag_to_idx[t]] = 1.0
                app_tags[app_id] = vec
print(f"Cached {len(app_tags)} games' tag vectors.")

# --- Model Instantiation & Build ---
model = build_and_compile_model()
# Call model with dummy inputs (51 features) to build graph
_ = model({'user_features': tf.zeros((1, 51)), 'item_features': tf.zeros((1, 51))})

weights_path = os.path.join(os.path.dirname(__file__), 'two_tower_weights.weights.h5')
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Successfully loaded trained model weights.")
else:
    print("Warning: Trained weights not found. Server running with random weights.")

# --- API Endpoints ---
class SteamGameItem(BaseModel):
    appid: int
    playtime_forever: int

class EmbeddingRequest(BaseModel):
    owned_games: list[SteamGameItem]

@app.post("/embed-user")
def embed_user(request: EmbeddingRequest):
    try:
        # Calculate total playtime and tag-weighted preference vector
        pref_vector = np.zeros(len(vocab), dtype=np.float32)
        total_hours = 0.0
        
        for game in request.owned_games:
            game_hours = game.playtime_forever / 60.0
            total_hours += game_hours
            
            # Weighted tag distribution
            game_vec = app_tags.get(game.appid, np.zeros(len(vocab), dtype=np.float32))
            pref_vector += game_vec * game_hours
            
        # Normalize tag preferences to sum to 1
        sum_pref = pref_vector.sum()
        if sum_pref > 0:
            pref_vector = pref_vector / sum_pref
            
        total_playtime_norm = np.log1p(total_hours)
        
        # Combine playtime (1) + tag distribution (50) = 51 features
        user_features = np.concatenate([[total_playtime_norm], pref_vector]).astype(np.float32)
        
        # Pass to the User Tower
        input_tensor = tf.constant([user_features], dtype=tf.float32)
        user_embedding = model.user_tower(input_tensor)
        
        # Return embedding vector as list of floats
        return {"user_embedding": user_embedding.numpy()[0].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
