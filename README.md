# protein-similarity-api

A lightweight, FastAPI service for comparing protein sequences using state-of-the-art transformer embeddings (ESM-2).

It provides a simple REST API to measure semantic similarity between protein sequences, returning a similarity score based on cosine distance in embedding space.

---

**The comparison results may be inaccurate. The goal of the project was to learn FastAPI & integrate with ESM-2. Feel free to copy the code and use it as you want.**

---

## Features

- Compare two protein sequences using [ESM-2](https://huggingface.co/facebook/esm2_t33_650M_UR50D)
- Returns cosine similarity + classification (homologous vs. non-homologous)
- Lightweight, deployable on free-tier platforms like Render or Railway
- Clean FastAPI structure, easy to extend (e.g. embeddings endpoint, caching)

---

## API Endpoints

### `POST /compare`

Compare two sequences and get similarity:

#### Request:

```json
{
  "sequence_1": "MSSKVIFF...",
  "sequence_2": "MTTRLIFF...",
  "model": "esm_2_650m"
}
```

#### Response:

```json
{
  "cosine_similarity": 0.57,
  "classification": "moderate similarity (possible remote homolog)",
  "model": "ESM-2 650M"
}
```

## How to run it?

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

uvicorn main:app --reload
```
