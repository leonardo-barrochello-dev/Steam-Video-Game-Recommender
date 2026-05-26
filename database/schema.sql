CREATE TABLE IF NOT EXISTS game_embeddings (
    app_id INTEGER PRIMARY KEY,
    embedding BLOB,
    name TEXT,
    genres TEXT
);
