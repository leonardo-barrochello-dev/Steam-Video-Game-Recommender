from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
)
import numpy as np


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "games"
VECTOR_SIZE = 128  # matching embedding_dim from model


class QdrantManager:
    def __init__(self, host=QDRANT_HOST, port=QDRANT_PORT):
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self):
        collections = self.client.get_collections().collections
        names = [c.name for c in collections]

        if COLLECTION_NAME in names:
            self.client.delete_collection(COLLECTION_NAME)

        self.client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )
        print(f"Created collection '{COLLECTION_NAME}' (size={VECTOR_SIZE})")

    def upsert_many(self, embeddings: np.ndarray, payloads: list[dict], batch_size=500):
        total = len(embeddings)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            points = [
                PointStruct(id=i, vector=embeddings[i].tolist(), payload=payloads[i])
                for i in range(start, end)
            ]
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
            )
            print(f"  Uploaded {end}/{total}...")
        print(f"Done. {total} points upserted to Qdrant.")

    def search(
        self, query_vector: np.ndarray, top_k: int = 100
    ) -> list[dict]:
        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=top_k,
        )
        return [
            {
                "app_id": hit.payload["app_id"],
                "name": hit.payload.get("name", ""),
                "score": hit.score,
            }
            for hit in response.points
        ]

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return COLLECTION_NAME in [c.name for c in collections]
