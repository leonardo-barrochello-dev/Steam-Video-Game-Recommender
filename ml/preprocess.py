import pandas as pd
import json
import os
import numpy as np
from collections import Counter

ML_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def get_or_create_tag_vocabulary(meta_df, vocab_size=50):
    """
    Extracts the most frequent tags and saves them as the vocabulary.
    """
    vocab_path = os.path.join(ML_DIR, 'tag_vocabulary.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        print(f"Loaded existing tag vocabulary of size {len(vocab)}")
        return vocab

    print("Generating new tag vocabulary...")
    all_tags = []
    for tags in meta_df['tags']:
        if isinstance(tags, list):
            all_tags.extend(tags)

    tag_counts = Counter(all_tags)
    most_common = [tag for tag, _ in tag_counts.most_common(vocab_size)]

    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(most_common, f, ensure_ascii=False, indent=2)

    print(f"Generated and saved tag vocabulary of size {len(most_common)}")
    return most_common


def load_data(sample_frac=1.0):
    """
    Loads datasets from local data/ directory.
    """
    print("Loading games.csv from local data/...")
    games_df = pd.read_csv(os.path.join(DATA_DIR, "games.csv"))

    print(f"Loading recommendations.csv from local data/ (sample_frac={sample_frac})...")
    TOTAL_REC_ROWS = 41_000_000
    nrows = None
    if sample_frac < 1.0:
        nrows = max(1000, int(TOTAL_REC_ROWS * sample_frac))

    recs_df = pd.read_csv(
        os.path.join(DATA_DIR, "recommendations.csv"),
        nrows=nrows,
    )

    print("Loading games_metadata.json from local data/...")
    metadata = []
    meta_path = os.path.join(DATA_DIR, 'games_metadata.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                metadata.append(json.loads(line))
    meta_df = pd.DataFrame(metadata)

    print(f"Loaded: {len(games_df)} games, {len(recs_df)} recommendations, {len(meta_df)} metadata entries")
    return games_df, recs_df, meta_df


def multi_hot_encode_tags(meta_df, vocab):
    """
    Encodes the tags of each game into a multi-hot numpy array.
    """
    tag_to_idx = {tag: idx for idx, tag in enumerate(vocab)}
    encoded_tags = {}
    
    for _, row in meta_df.iterrows():
        app_id = row['app_id']
        tags = row['tags']
        vector = np.zeros(len(vocab), dtype=np.float32)
        if isinstance(tags, list):
            for tag in tags:
                if tag in tag_to_idx:
                    vector[tag_to_idx[tag]] = 1.0
        encoded_tags[app_id] = vector
        
    return encoded_tags

NON_GAME_TAGS = {
    'Video Production', 'Photo Editing', 'Audio Production', 'Utilities',
    'Software Training', 'Game Development', 'Animation & Modeling',
    'Web Publishing', 'Accounting', 'Design & Illustration'
}

def is_game(tags):
    """Filter out software, tools and non-game items based on tags."""
    if not isinstance(tags, list) or len(tags) == 0:
        return False
    if any(t in NON_GAME_TAGS for t in tags):
        return False
    return True

def engineer_features(games_df, recs_df, meta_df, vocab):
    """
    Creates Item and User features using the tag vocabulary.
    Only includes actual games (filters out software/tools/soundtracks).
    """
    print("Engineering features...")
    games_df = games_df.copy()
    games_df['price_norm'] = np.log1p(games_df['price_final']).astype(np.float32)

    # Merge games and metadata to get tags
    item_features = pd.merge(games_df[['app_id', 'title', 'price_norm']], meta_df[['app_id', 'tags']], on='app_id', how='left')

    # Filter out non-games (software, tools, etc.)
    print("Filtering non-game items...")
    initial_count = len(item_features)
    item_features = item_features[item_features['tags'].apply(is_game)].reset_index(drop=True)
    print(f"Filtered {initial_count - len(item_features)} non-game items. Remaining: {len(item_features)}")

    # Encode tags
    encoded_tags = multi_hot_encode_tags(item_features, vocab)

    # Attach multi-hot vectors as a column
    item_features['tags_vector'] = item_features['app_id'].map(encoded_tags)
    zero_vec = np.zeros(len(vocab), dtype=np.float32)
    item_features['tags_vector'] = item_features['tags_vector'].apply(lambda x: x if isinstance(x, np.ndarray) else zero_vec)

    # Filter recommendations to only include valid game app_ids
    valid_app_ids = set(item_features['app_id'])
    recs_df = recs_df[recs_df['app_id'].isin(valid_app_ids)].copy()

    # 2. Interactions (Labels)
    recs_df['label'] = ((recs_df['is_recommended'].astype(str).str.lower() == 'true') | (recs_df['hours'] > 2.0)).astype(int)

    return item_features, recs_df

def prepare_training_data(item_features, recs_df, vocab, neg_samples_per_pos=1, random_seed=42):
    print("Preparing training data (vectorized)...")
    np.random.seed(random_seed)

    app_to_tags = dict(zip(item_features['app_id'], item_features['tags_vector']))
    zero_vec = np.zeros(len(vocab), dtype=np.float32)

    print("Mapping app_ids to tag vectors...")
    tags_array = np.stack(recs_df['app_id'].map(lambda x: app_to_tags.get(x, zero_vec)).values)

    hours_array = recs_df['hours'].values.astype(np.float32)
    weighted_tags = tags_array * hours_array[:, np.newaxis]

    print("Grouping user features...")
    temp_df = pd.DataFrame(weighted_tags, columns=[f'tag_{i}' for i in range(len(vocab))])
    temp_df['user_id'] = recs_df['user_id'].values
    temp_df['hours'] = hours_array

    user_sums = temp_df.groupby('user_id').sum()

    # Per-user playtime stats
    user_playtimes = np.log1p(user_sums['hours'].values).astype(np.float32)
    user_playtime_dict = dict(zip(user_sums.index, user_playtimes))

    # Normalized tag preferences
    tag_cols = [f'tag_{i}' for i in range(len(vocab))]
    tag_sums = user_sums[tag_cols].values
    row_sums = tag_sums.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    normalized_prefs = (tag_sums / row_sums).astype(np.float32)

    user_profiles = dict(zip(user_sums.index, normalized_prefs))

    # Per-user: games owned, avg playtime, unique genres
    user_game_counts = recs_df.groupby('user_id')['app_id'].nunique()
    user_game_count_dict = dict(zip(user_game_counts.index, np.log1p(user_game_counts.values).astype(np.float32)))

    user_avg_playtime = (user_sums['hours'].values / user_game_counts.values.astype(np.float32))
    user_avg_playtime = np.where(user_game_counts.values == 0, 0.0, user_avg_playtime)
    user_avg_playtime_dict = dict(zip(user_sums.index, np.log1p(user_avg_playtime).astype(np.float32)))

    user_unique_genres = (tag_sums > 0).sum(axis=1).astype(np.float32)
    user_unique_genres_dict = dict(zip(user_sums.index, np.log1p(user_unique_genres).astype(np.float32)))

    print("Assembling final feature matrices...")
    user_pref_mapped = np.stack(recs_df['user_id'].map(lambda uid: user_profiles.get(uid, zero_vec)).values)
    user_time_mapped = recs_df['user_id'].map(lambda uid: user_playtime_dict.get(uid, 0.0)).values.astype(np.float32)

    item_tags_mapped = tags_array
    app_to_price = dict(zip(item_features['app_id'], item_features['price_norm']))
    item_price_mapped = recs_df['app_id'].map(lambda aid: app_to_price.get(aid, 0.0)).values.astype(np.float32)

    original_labels = recs_df['label'].values

    # Build extended user features: [playtime, owned, avg_pt, unique_tags, tag_prefs]
    def build_user_feat(uid):
        pref = user_profiles.get(uid, zero_vec)
        ut = user_time if 'user_time' in dir() else 0.0
        gc = user_game_count_dict.get(uid, 0.0)
        ap = user_avg_playtime_dict.get(uid, 0.0)
        ug = user_unique_genres_dict.get(uid, 0.0)
        pt = user_playtime_dict.get(uid, 0.0)
        return np.concatenate([[pt, gc, ap, ug], pref]).astype(np.float32)

    user_ids_arr = recs_df['user_id'].values
    user_feat_list = []
    for uid in user_ids_arr:
        pref = user_profiles.get(uid, zero_vec)
        pt = user_playtime_dict.get(uid, 0.0)
        gc = user_game_count_dict.get(uid, 0.0)
        ap = user_avg_playtime_dict.get(uid, 0.0)
        ug = user_unique_genres_dict.get(uid, 0.0)
        user_feat_list.append(np.concatenate([[pt, gc, ap, ug], pref]))
    user_features_matrix = np.stack(user_feat_list).astype(np.float32)

    item_features_matrix = np.concatenate([item_price_mapped[:, np.newaxis], item_tags_mapped], axis=1).astype(np.float32)

    # Negative sampling
    if neg_samples_per_pos > 0:
        print(f"Generating synthetic negatives (neg_samples_per_pos={neg_samples_per_pos})...")
        pos_mask = original_labels == 1
        n_pos = int(pos_mask.sum())
        n_negatives = n_pos * neg_samples_per_pos

        all_app_ids = np.array(sorted(app_to_tags.keys()))
        pos_user_ids = recs_df['user_id'].values[pos_mask]
        sampled_user_ids = np.random.choice(pos_user_ids, size=n_negatives, replace=True)
        sampled_app_ids = np.random.choice(all_app_ids, size=n_negatives, replace=True)

        neg_user_feats = []
        neg_item_feats = []

        unique_users = set(sampled_user_ids)
        user_feat_cache = {}
        for uid in unique_users:
            pref = user_profiles.get(uid, zero_vec)
            pt = user_playtime_dict.get(uid, 0.0)
            gc = user_game_count_dict.get(uid, 0.0)
            ap = user_avg_playtime_dict.get(uid, 0.0)
            ug = user_unique_genres_dict.get(uid, 0.0)
            user_feat_cache[uid] = np.concatenate([[pt, gc, ap, ug], pref]).astype(np.float32)

        unique_items = set(sampled_app_ids)
        item_feat_cache = {}
        for aid in unique_items:
            vec = app_to_tags.get(aid, zero_vec)
            price = app_to_price.get(aid, 0.0)
            item_feat_cache[aid] = np.concatenate([[price], vec]).astype(np.float32)

        for uid, aid in zip(sampled_user_ids, sampled_app_ids):
            neg_user_feats.append(user_feat_cache[uid])
            neg_item_feats.append(item_feat_cache[aid])

        synth_user_matrix = np.stack(neg_user_feats)
        synth_item_matrix = np.stack(neg_item_feats)
        synth_labels = np.zeros(n_negatives, dtype=np.float32)

        print(f"  Added {n_negatives} synthetic negatives.")

        user_features_matrix = np.concatenate([user_features_matrix, synth_user_matrix], axis=0)
        item_features_matrix = np.concatenate([item_features_matrix, synth_item_matrix], axis=0)
        all_labels = np.concatenate([original_labels, synth_labels], axis=0)
    else:
        all_labels = original_labels

    train_df = pd.DataFrame({
        'user_features': list(user_features_matrix),
        'item_features': list(item_features_matrix),
        'label': all_labels
    })

    user_stats = pd.DataFrame([
        {
            'user_id': uid,
            'total_playtime_norm': user_playtime_dict[uid],
            'game_count_norm': user_game_count_dict.get(uid, 0.0),
            'avg_playtime_norm': user_avg_playtime_dict.get(uid, 0.0),
            'unique_genres_norm': user_unique_genres_dict.get(uid, 0.0),
            'tag_preferences': user_profiles[uid]
        } for uid in user_sums.index
    ])

    return train_df, item_features, user_stats

if __name__ == "__main__":
    g, r, m = load_data(sample_frac=0.01)
    vocab = get_or_create_tag_vocabulary(m, vocab_size=50)
    items, recs = engineer_features(g, r, m, vocab)
    train, items, users = prepare_training_data(items, recs, vocab)
    print(f"Training data size: {len(train)}")
    print("Preprocessing completed.")
