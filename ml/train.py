import os
import tensorflow as tf
from preprocess import load_data, engineer_features, prepare_training_data
from model import build_and_compile_model


def train_model():
    print("--- Starting Training Pipeline ---")

    # Load a smaller fraction of the dataset to train quickly and avoid memory limit (recommendations.csv is 2GB)
    # You can increase this fraction if you have enough RAM/GPU power.
    sample_fraction = 0.1
    print(f"Loading data (using {sample_fraction*100}% sample)...")

    import numpy as np
    from preprocess import get_or_create_tag_vocabulary

    games_df, recs_df, meta_df = load_data(sample_frac=sample_fraction)
    vocab = get_or_create_tag_vocabulary(meta_df, vocab_size=50)

    items, recs = engineer_features(games_df, recs_df, meta_df, vocab)
    train_df, items_features, user_stats = prepare_training_data(items, recs, vocab)

    print(f"Dataset prepared. Total training samples: {len(train_df)}")

    # Prepare TensorFlow inputs from stacked feature vectors
    user_inputs = np.stack(train_df["user_features"].values).astype("float32")
    item_inputs = np.stack(train_df["item_features"].values).astype("float32")
    labels = train_df["label"].values.astype("float32").reshape(-1, 1)

    print("Building and compiling Two-Tower Model...")
    model = build_and_compile_model(embedding_dim=32, learning_rate=0.001)

    # Fit the model
    print("Training model...")
    model.fit(
        x={"user_features": user_inputs, "item_features": item_inputs},
        y=labels,
        epochs=15,
        batch_size=1024,
        validation_split=0.1,
    )

    # Save the trained weights
    weights_path = os.path.join(
        os.path.dirname(__file__), "two_tower_weights.weights.h5"
    )
    model.save_weights(weights_path)
    print(f"Model weights saved successfully to: {weights_path}")


if __name__ == "__main__":
    train_model()
