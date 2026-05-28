import os
import sqlite3
import numpy as np
import tensorflow as tf
from model import build_and_compile_model
from preprocess import load_data, engineer_features, get_or_create_tag_vocabulary

def generate_and_save_item_embeddings():
    print("Loading data for embedding generation...")
    games_df, recs_df, meta_df = load_data(sample_frac=1.0) # Generate for all games
    vocab = get_or_create_tag_vocabulary(meta_df, vocab_size=50)
    
    # Generate features exactly as used in training
    item_features_df, _ = engineer_features(games_df, recs_df, meta_df, vocab)
    
    # We only need unique items for the embeddings database
    unique_items = item_features_df.drop_duplicates(subset=['app_id']).copy()
    
    # Prepare features for the model (price_norm + multi-hot tags = 51 dims)
    item_features_list = []
    for _, row in unique_items.iterrows():
        feat = np.concatenate([[row['price_norm']], row['tags_vector']])
        item_features_list.append(feat)
    item_input_features = np.stack(item_features_list).astype('float32')
    
    print("Loading model weights...")
    model = build_and_compile_model()
    # Call model on dummy inputs of shape (1, 51) to build the weights/layers
    _ = model([tf.zeros((1, 51)), tf.zeros((1, 51))])
    
    weights_path = os.path.join(os.path.dirname(__file__), 'two_tower_weights.weights.h5')
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Successfully loaded trained weights.")
    else:
        print("Warning: Trained weights not found. Generating embeddings with random initialization.")
    
    # Create the item tower independently or use model.item_tower
    print("Generating item embeddings...")
    # tf.function for speed if dealing with batches, but here direct call
    item_embeddings = model.get_layer('item_tower')(item_input_features).numpy()
    
    # Connect to SQLite and save
    db_path = os.path.join(os.path.dirname(__file__), '..', 'database', 'recommender.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS game_embeddings (
        app_id INTEGER PRIMARY KEY,
        embedding BLOB,
        name TEXT,
        genres TEXT
    )
    ''')
    
    print("Saving to SQLite...")
    # Clear old
    cursor.execute('DELETE FROM game_embeddings')
    
    records = []
    for idx, (_, row) in enumerate(unique_items.iterrows()):
        app_id = int(row['app_id'])
        name = str(row['title'])
        tags = str(row['tags'])
        emb_blob = item_embeddings[idx].tobytes()
        records.append((app_id, emb_blob, name, tags))
        
    cursor.executemany('''
    INSERT INTO game_embeddings (app_id, embedding, name, genres)
    VALUES (?, ?, ?, ?)
    ''', records)
    
    conn.commit()
    conn.close()
    
    print(f"Successfully saved {len(records)} game embeddings to SQLite.")

if __name__ == "__main__":
    generate_and_save_item_embeddings()
