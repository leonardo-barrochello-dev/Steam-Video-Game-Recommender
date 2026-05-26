import os
import requests
import pandas as pd
import sqlite3
from datetime import datetime
import kagglehub
from kagglehub import KaggleDatasetAdapter

STEAM_API_KEY = os.environ.get("STEAM_API_KEY")
KAGGLE_DATASET = "antonkozyriev/game-recommendations-on-steam"
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'database', 'recommender.db')


def get_owned_games(steam_id: str):
    url = f"http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/?key={STEAM_API_KEY}&steamid={steam_id}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'games' in data['response']:
            return data['response']['games']
    return []


def get_recently_played_games(steam_id: str):
    url = f"http://api.steampowered.com/IPlayerService/GetRecentlyPlayedGames/v0001/?key={STEAM_API_KEY}&steamid={steam_id}&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'response' in data and 'games' in data['response']:
            return data['response']['games']
    return []


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS collected_recommendations (
            app_id INTEGER,
            helpful INTEGER,
            funny INTEGER,
            date TEXT,
            is_recommended TEXT,
            hours REAL,
            user_id INTEGER,
            review_id INTEGER,
            PRIMARY KEY (app_id, user_id, review_id)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS collected_users (
            user_id INTEGER PRIMARY KEY,
            products INTEGER,
            reviews INTEGER
        )
    ''')
    conn.commit()
    conn.close()


def _load_collected_recs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM collected_recommendations", conn)
    conn.close()
    return df


def _load_collected_users():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM collected_users", conn)
    conn.close()
    return df


def _save_collected_recs(df):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM collected_recommendations')
    df.to_sql('collected_recommendations', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()


def _save_collected_users(df):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('DELETE FROM collected_users')
    df.to_sql('collected_users', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()


def update_dataset(steam_id: str):
    """
    Fetches a user's Steam library via the Steam API and persists their
    play data into the SQLite database (database/recommender.db).
    Base deduplication uses the Kaggle dataset as reference.
    """
    if not STEAM_API_KEY:
        print("Error: STEAM_API_KEY environment variable is not set.")
        return

    try:
        user_id = int(steam_id) % 1000000000
    except ValueError:
        print(f"Invalid steam_id: {steam_id}")
        return

    print(f"Fetching data for steam_id: {steam_id} (mapped to user_id: {user_id})")
    owned_games = get_owned_games(steam_id)
    recent_games = get_recently_played_games(steam_id)

    games_dict = {g['appid']: g for g in owned_games}
    for g in recent_games:
        games_dict[g['appid']] = g

    if not games_dict:
        print("No games found for this user.")
        return

    _init_db()

    # 1. Recommendations
    print("Loading base recommendations from Kaggle for deduplication...")
    df_recs_base = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET,
        "recommendations.csv",
    )

    df_recs_collected = _load_collected_recs()
    df_recs = pd.concat([df_recs_base, df_recs_collected], ignore_index=True)

    new_recs = []
    current_date = datetime.now().strftime('%Y-%m-%d')
    for app_id, game_data in games_dict.items():
        playtime_forever = game_data.get('playtime_forever', 0)
        hours = round(playtime_forever / 60.0, 1)
        is_recommended = hours > 2.0
        new_recs.append({
            'app_id': app_id,
            'helpful': 0,
            'funny': 0,
            'date': current_date,
            'is_recommended': str(is_recommended).lower(),
            'hours': hours,
            'user_id': user_id,
            'review_id': 0,
        })

    df_new_recs = pd.DataFrame(new_recs)

    df_recs = df_recs[df_recs['user_id'] != user_id]
    df_recs = pd.concat([df_recs, df_new_recs], ignore_index=True)

    _save_collected_recs(df_recs[df_recs['user_id'] == user_id])
    print(f"Saved {len(new_recs)} recommendation records for user {user_id} to database.")

    # 2. Users
    print("Loading base users from Kaggle for deduplication...")
    df_users_base = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        KAGGLE_DATASET,
        "users.csv",
    )
    df_users_collected = _load_collected_users()
    df_users = pd.concat([df_users_base, df_users_collected], ignore_index=True)

    user_record = {
        'user_id': user_id,
        'products': len(games_dict),
        'reviews': len(new_recs),
    }

    df_users = df_users[df_users['user_id'] != user_id]
    df_users = pd.concat([df_users, pd.DataFrame([user_record])], ignore_index=True)

    user_df = df_users[df_users['user_id'] == user_id]
    _save_collected_users(user_df if not user_df.empty else pd.DataFrame([user_record]))
    print(f"Saved user {user_id} record to database.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        update_dataset(sys.argv[1])
    else:
        print("Usage: python collect_steam_data.py <steam_id>")
