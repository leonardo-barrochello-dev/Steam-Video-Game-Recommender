import os
import requests
import pandas as pd
from datetime import datetime

# Fluxo 1 — Coleta de Dados (Dataset)
STEAM_API_KEY = os.environ.get("STEAM_API_KEY")

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

def update_dataset(steam_id: str):
    """
    Updates the dataset with data from the given steam_id.
    Maintains compatibility with recommendations.csv and users.csv.
    """
    if not STEAM_API_KEY:
        print("Error: STEAM_API_KEY environment variable is not set.")
        return

    # Convert steam_id (64-bit string) to a 32-bit user_id to match dataset format
    try:
        user_id = int(steam_id) % 1000000000  # simplistic hash to keep within int range
    except ValueError:
        print(f"Invalid steam_id: {steam_id}")
        return

    print(f"Fetching data for steam_id: {steam_id} (mapped to user_id: {user_id})")
    owned_games = get_owned_games(steam_id)
    recent_games = get_recently_played_games(steam_id)
    
    # Merge and deduplicate, preferring recent games data for playtime
    games_dict = {g['appid']: g for g in owned_games}
    for g in recent_games:
        games_dict[g['appid']] = g  # Override with recent data if available
    
    if not games_dict:
        print("No games found for this user.")
        return

    # 1. Update recommendations.csv
    # Schema: app_id,helpful,funny,date,is_recommended,hours,user_id,review_id
    rec_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'recommendations.csv')
    
    new_recs = []
    current_date = datetime.now().strftime('%Y-%m-%d')
    for app_id, game_data in games_dict.items():
        playtime_forever = game_data.get('playtime_forever', 0)
        hours = round(playtime_forever / 60.0, 1)
        
        # Determine is_recommended implicitly based on playtime > 2 hours
        is_recommended = hours > 2.0
        
        new_recs.append({
            'app_id': app_id,
            'helpful': 0,
            'funny': 0,
            'date': current_date,
            'is_recommended': str(is_recommended).lower(),
            'hours': hours,
            'user_id': user_id,
            'review_id': 0 # Dummy review id for implicit data
        })
        
    df_new_recs = pd.DataFrame(new_recs)
    
    try:
        df_recs = pd.read_csv(rec_path)
        # Remove existing records for this user to avoid duplication, then append
        df_recs = df_recs[df_recs['user_id'] != user_id]
        df_combined = pd.concat([df_recs, df_new_recs], ignore_index=True)
        df_combined.to_csv(rec_path, index=False)
        print(f"Updated recommendations.csv with {len(new_recs)} records.")
    except Exception as e:
        print(f"Error updating recommendations.csv: {e}")

    # 2. Update users.csv
    # Schema: user_id,products,reviews
    users_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'users.csv')
    try:
        df_users = pd.read_csv(users_path)
        
        user_record = {
            'user_id': user_id,
            'products': len(games_dict),
            'reviews': len(new_recs)  # Representing interactions
        }
        
        # Remove if exists and append
        df_users = df_users[df_users['user_id'] != user_id]
        df_users = pd.concat([df_users, pd.DataFrame([user_record])], ignore_index=True)
        df_users.to_csv(users_path, index=False)
        print(f"Updated users.csv for user_id {user_id}.")
    except Exception as e:
        print(f"Error updating users.csv: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        update_dataset(sys.argv[1])
    else:
        print("Usage: python collect_steam_data.py <steam_id>")
