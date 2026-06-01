import os
import json
import numpy as np
import tensorflow as tf
from model import build_and_compile_model, USER_FEATURE_DIM, ITEM_FEATURE_DIM
from preprocess import load_data, engineer_features, get_or_create_tag_vocabulary
from qdrant_manager import QdrantManager

ML_DIR = os.path.dirname(__file__)


def generate_and_save_item_embeddings():
    print("Loading data for embedding generation...")
    games_df, recs_df, meta_df = load_data(sample_frac=1.0)
    vocab = get_or_create_tag_vocabulary(meta_df, vocab_size=50)

    item_features_df, _ = engineer_features(games_df, recs_df, meta_df, vocab)
    unique_items = item_features_df.drop_duplicates(subset=['app_id']).copy()

    item_features_list = []
    for _, row in unique_items.iterrows():
        feat = np.concatenate([[row['price_norm']], row['tags_vector']])
        item_features_list.append(feat)
    item_input_features = np.stack(item_features_list).astype('float32')

    print("Loading model weights...")
    model = build_and_compile_model()
    _ = model([tf.zeros((1, USER_FEATURE_DIM)), tf.zeros((1, ITEM_FEATURE_DIM))])

    weights_path = os.path.join(ML_DIR, 'two_tower_weights.weights.h5')
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Successfully loaded trained weights.")
    else:
        print("Warning: Trained weights not found. Generating embeddings with random initialization.")

    print("Generating item embeddings...")
    item_embeddings = model.get_layer('item_tower')(item_input_features).numpy()

    print("Uploading embeddings to Qdrant...")
    qdrant = QdrantManager()
    qdrant.create_collection()

    payloads = [
        {
            "app_id": int(row['app_id']),
            "name": str(row['title']),
            "genres": str(row['tags']),
        }
        for _, row in unique_items.iterrows()
    ]
    qdrant.upsert_many(item_embeddings, payloads)
    print(f"Uploaded {len(payloads)} item embeddings to Qdrant.")


if __name__ == "__main__":
    generate_and_save_item_embeddings()
