from fastapi import FastAPI
from pydantic import BaseModel
from utils import compare_embeddings

from models.esm_2_650m import get_embedding as get_embedding_esm_2_650m

app = FastAPI()


class CompareRequest(BaseModel):
    sequence_1: str
    sequence_2: str
    model: str = "esm_2_650m"


model_mapping = {"esm_2_650m": get_embedding_esm_2_650m}


# ----------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "API is running. Use POST /compare to compare protein sequences."
    }


# ----------------------------------------------------------------------
@app.post("/compare")
def compare(request: CompareRequest):
    model = request.model

    if model not in model_mapping:
        return {"error": "Model not supported"}

    emb1 = model_mapping[model](request.sequence_1)
    emb2 = model_mapping[model](request.sequence_2)

    similarity, classification = compare_embeddings(emb1, emb2)

    return {
        "cosine_similarity": float(similarity),
        "classification": classification,
        "model": model,
    }
