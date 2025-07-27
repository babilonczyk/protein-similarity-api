# protein-similarity-api

A lightweight, production-ready FastAPI service for comparing protein sequences using state-of-the-art transformer embeddings (ESM-2).  
It provides a simple REST API to measure semantic similarity between protein sequences, returning a similarity score based on cosine distance in embedding space.

---

## Why this matters?

Protein similarity is at the heart of functional annotation, homology detection, and structure prediction.  
This API lets anyone (researcher, student, dev) compare protein sequences using powerful models.

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
  "sequence_2": "MTTRLIFF..."
}
```

#### Response:

```json
{
  "cosine_similarity": 0.87,
  "classification": "likely homologous",
  "model": "ESM-2 650M"
}
```

## How to run it?

```
```

## How to deploy it? 

```
```
