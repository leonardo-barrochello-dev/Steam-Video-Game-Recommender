# Steam Video Game Recommender

A personalized game recommendation system built with a Two-Tower neural network (TensorFlow), served via a C# .NET Web API with real-time Steam API integration and Qdrant for fast ANN vector search.

## Architecture

```
data/*.csv ──> preprocess.py ──> train.py ──> two_tower_weights.weights.h5
               (load & feature      (train model)
                engineer)

two_tower_weights.weights.h5 ──> generate_embeddings.py ──> Qdrant (ANN search)
                                (extract item embeddings     vector + payload {app_id, name, genres}
                                 via Item Tower)

User Request:
  C# API (GET /recommend?steam_id=XXX)
    ├── SteamApiService: gets user's owned games from Steam API
    ├── RecommendService: POSTs owned_games to Python FastAPI
    │     └── /recommend endpoint: User Tower → Qdrant search → filter owned → Top N
    └── Returns Top-N recommendations as JSON
```

### Components

| Component | Stack | Location |
|-----------|-------|----------|
| **ML Training** | Python, TensorFlow | `ml/` |
| **ML Inference** | Python, FastAPI | `ml/server.py` |
| **Vector Search** | Qdrant (Docker) | `docker-compose.yml` / `ml/qdrant_manager.py` |
| **Web API** | C#, ASP.NET Core | `api/SteamRecommenderAPI/` |

## Dataset

The dataset is downloaded automatically via Kaggle CLI — see [Setup](#3-download-dataset).

Files: `games.csv` (~50k games), `recommendations.csv` (~41M reviews), `users.csv` (~14M users), `games_metadata.json` (~50k entries with tags).

## Setup

### Prerequisites
- Python 3.10+
- .NET 10 SDK
- Docker (for Qdrant)
- Steam API key ([get one here](https://steamcommunity.com/dev/apikey))

### 1. Python ML Environment

```bash
cd ml
python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Kaggle API Token

The Kaggle CLI needs a `kaggle.json` file with your credentials:

```powershell
mkdir ~\.kaggle -Force
```

1. Go to https://www.kaggle.com/settings/api
2. Under "Legacy API Credentials", click "Create Legacy API Key"
3. Move the downloaded `kaggle.json` to `~\.kaggle\kaggle.json`

### 3. Download Dataset

```bash
python ml/download_data.py
```

### 4. Start Qdrant

```bash
docker compose up -d
```

Qdrant runs on `localhost:6333` and stores data in `./qdrant_storage/`.

### 5. Configure Steam API Key

Set the key in `api/SteamRecommenderAPI/appsettings.json`:

```json
"Steam": {
  "ApiKey": "YOUR_STEAM_API_KEY"
}
```

Or use environment variable / `appsettings.Development.json`.

### 6. Train the Model

```bash
cd ml
python train.py
```

This loads data from `data/`, trains the Two-Tower model, and saves weights to `two_tower_weights.weights.h5`.

### 7. Generate Item Embeddings & Upload to Qdrant

```bash
cd ml
python generate_embeddings.py
```

Extracts item embeddings via the Item Tower and uploads them to Qdrant with metadata payload (`app_id`, `name`, `genres`).

### 8. Start the ML Server

```bash
cd ml
python server.py
```

Starts a FastAPI server on port 5000 with endpoints:
- `POST /embed-user` — legacy user embedding generation
- `POST /recommend` — User Tower → Qdrant search → filter owned → Top recommendations

### 9. Start the C# API

```bash
cd api/SteamRecommenderAPI
dotnet run
```

Starts the ASP.NET API on port 5088.

## Usage

```bash
curl "http://localhost:5088/recommend?steam_id=76561197960435530"
```

Returns a JSON array of recommended games:

```json
[
  { "app_id": 730, "name": "Counter-Strike 2", "score": 0.87 },
  { "app_id": 570, "name": "Dota 2", "score": 0.82 }
]
```

## Project Structure

```
├── docker-compose.yml            # Qdrant container
├── ml/                           # Python ML Service
│   ├── preprocess.py             # Data loading & feature engineering
│   ├── model.py                  # Two-Tower architecture
│   ├── train.py                  # Training pipeline
│   ├── generate_embeddings.py    # Item embeddings → Qdrant
│   ├── server.py                 # FastAPI inference server
│   ├── qdrant_manager.py         # Qdrant client wrapper
│   ├── download_data.py          # Dataset download from Kaggle
│   └── requirements.txt
├── api/SteamRecommenderAPI/      # C# Web API
│   ├── Controllers/
│   │   └── RecommendController.cs
│   ├── Services/
│   │   ├── RecommendService.cs   # Calls /recommend on ML API
│   │   ├── SteamApiService.cs
│   │   └── EmbeddingService.cs   # Legacy (unused)
│   └── Program.cs
└── data/                         # Dataset files (CSVs + JSON)
```

## Model Architecture

The Two-Tower model learns 32-dimensional embeddings for users and items:

- **User Tower**: `[playtime + 50 tag features]` → Dense(256) → Dense(128) → Dense(32) → L2 normalize
- **Item Tower**: `[price + 50 tag features]` → Dense(256) → Dense(128) → Dense(32) → L2 normalize
- **Output**: Dot product → Sigmoid → Binary cross-entropy loss

## Qdrant Collection

At startup, `generate_embeddings.py` creates a Qdrant collection named `games`:

| Property | Value |
|----------|-------|
| Vector size | 32 (matching embedding_dim) |
| Distance metric | Cosine |
| Payload | `app_id` (int), `name` (str), `genres` (str) |

## Notes

- `data/` directory contains the dataset files (CSVs + JSONL) required for training.
- Model weights (`*.h5`) are generated artifacts and not tracked in git.
- Qdrant storage (`qdrant_storage/`) is excluded from git.
