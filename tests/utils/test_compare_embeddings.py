import numpy as np
from utils import compare_embeddings


def test_very_high_similarity():
    emb1 = np.array([0.1, 0.2, 0.3])
    emb2 = np.array([0.1, 0.2, 0.3])
    similarity, classification = compare_embeddings(emb1, emb2)

    assert similarity >= 0.85
    assert classification == "very high similarity (clear homology)"


def test_high_similarity():
    emb1 = np.array([1, 0, 0])
    emb2 = np.array([0.8, 0.6, 0])
    similarity, classification = compare_embeddings(emb1, emb2)

    assert 0.70 <= similarity < 0.85
    assert classification == "high similarity (likely homologous)"


def test_moderate_similarity():
    emb1 = np.array([1, 0, 0])
    emb2 = np.array([0.6, 0.6, 0.6])
    similarity, classification = compare_embeddings(emb1, emb2)

    assert 0.50 <= similarity < 0.70
    assert classification == "moderate similarity (possible remote homolog)"


def test_low_similarity():
    emb1 = np.array([1, 0, 0])
    emb2 = np.array([0.3, 0.95, 0])
    similarity, classification = compare_embeddings(emb1, emb2)

    assert 0.30 <= similarity < 0.50
    assert classification == "low similarity (likely not homologous)"


def test_very_low_similarity():
    emb1 = np.array([1, 0, 0])
    emb2 = np.array([0, 1, 0])
    similarity, classification = compare_embeddings(emb1, emb2)

    assert similarity < 0.30
    assert classification == "very low similarity (unrelated / random match)"
