from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from model import build_and_compile_model
from qdrant_manager import QdrantManager
import json
import os

app = FastAPI(title="Steam Recommender ML API")

ML_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# --- Startup Data Loading & Caching ---
vocab_path = os.path.join(ML_DIR, 'tag_vocabulary.json')

if not os.path.exists(vocab_path):
    raise RuntimeError(f"Vocabulary file not found at {vocab_path}. Please run train.py first.")

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)
tag_to_idx = {tag: idx for idx, tag in enumerate(vocab)}

# Load Game Metadata from local data/ directory
print("Caching game tag vectors for real-time inference...")
meta_path = os.path.join(DATA_DIR, 'games_metadata.json')

app_tags = {}
if os.path.exists(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                app_id = item.get('app_id')
                tags = item.get('tags', [])

                vec = np.zeros(len(vocab), dtype=np.float32)
                for t in tags:
                    if t in tag_to_idx:
                        vec[tag_to_idx[t]] = 1.0
                app_tags[app_id] = vec
print(f"Cached {len(app_tags)} games' tag vectors.")

# --- Model Instantiation & Build ---
model = build_and_compile_model()
_ = model([tf.zeros((1, 51)), tf.zeros((1, 51))])

weights_path = os.path.join(ML_DIR, 'two_tower_weights.weights.h5')
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Successfully loaded trained model weights.")
else:
    print("Warning: Trained weights not found. Server running with random weights.")

# --- Qdrant client ---
qdrant = None
try:
    qdrant = QdrantManager()
    if qdrant.collection_exists():
        print("Connected to Qdrant. Collection 'games' found.")
    else:
        print("Warning: Qdrant collection 'games' not found. Run generate_embeddings.py first.")
except Exception as e:
    print(f"Warning: Could not connect to Qdrant: {e}")


# --- API Endpoints ---
class SteamGameItem(BaseModel):
    appid: int
    playtime_forever: int

class EmbeddingRequest(BaseModel):
    owned_games: list[SteamGameItem]

class RecommendRequest(BaseModel):
    owned_games: list[SteamGameItem]
    top_k: int = 10

class RecommendCandidate(BaseModel):
    app_id: int
    name: str
    score: float

class RecommendResponse(BaseModel):
    recommendations: list[RecommendCandidate]


def compute_user_embedding(owned_games: list[SteamGameItem]) -> np.ndarray:
    pref_vector = np.zeros(len(vocab), dtype=np.float32)
    total_hours = 0.0

    for game in owned_games:
        game_hours = game.playtime_forever / 60.0
        total_hours += game_hours

        game_vec = app_tags.get(game.appid, np.zeros(len(vocab), dtype=np.float32))
        pref_vector += game_vec * game_hours

    sum_pref = pref_vector.sum()
    if sum_pref > 0:
        pref_vector = pref_vector / sum_pref

    total_playtime_norm = np.log1p(total_hours)

    user_features = np.concatenate([[total_playtime_norm], pref_vector]).astype(np.float32)

    input_tensor = tf.constant([user_features], dtype=tf.float32)
    user_emb = model.get_layer('user_tower')(input_tensor)
    return user_emb.numpy()[0]


@ app.post("/embed-user")
def embed_user(request: EmbeddingRequest):
    try:
        user_embedding = compute_user_embedding(request.owned_games)
        return {"user_embedding": user_embedding.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@ app.post("/recommend")
def recommend(request: RecommendRequest):
    if qdrant is None:
        raise HTTPException(status_code=503, detail="Qdrant not connected. Ensure Qdrant is running.")

    try:
        user_emb = compute_user_embedding(request.owned_games)

        owned_set = {g.appid for g in request.owned_games}
        RETRIEVAL_SIZE = 100  # retrieve more than needed, then filter + trim

        hits = qdrant.search(user_emb, top_k=RETRIEVAL_SIZE)

        results = [
            RecommendCandidate(app_id=h["app_id"], name=h["name"], score=h["score"])
            for h in hits
            if h["app_id"] not in owned_set
        ][:request.top_k]

        return RecommendResponse(recommendations=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
