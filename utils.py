from scipy.spatial.distance import cosine


def compare_embeddings(emb1, emb2):
    similarity = 1 - cosine(emb1, emb2)

    if similarity >= 0.85:
        classification = "very high similarity (clear homology)"
    elif similarity >= 0.70:
        classification = "high similarity (likely homologous)"
    elif similarity >= 0.50:
        classification = "moderate similarity (possible remote homolog)"
    elif similarity >= 0.30:
        classification = "low similarity (likely not homologous)"
    else:
        classification = "very low similarity (unrelated / random match)"

    return similarity, classification
