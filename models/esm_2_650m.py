from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")


# -------------------------------------------------------------------------------------------------------
# get_embedding:
# This function takes a protein sequence (like "MKTFFV...") and turns it into a single vector (embedding)
# using the ESM-2 language model for proteins. This vector is like a unique "fingerprint" of the sequence
# and can be compared with others using cosine similarity.

# Here's how it works step by step:

# 1. Tokenization (turn text into numbers):
#    The input sequence (a string of amino acids) is turned into tokens that the model understands.
#    The tokenizer:
#      - Adds special tokens like [CLS] at the beginning (used as a summary marker)
#      - Pads or truncates if needed
#      - Returns a PyTorch tensor with shape [1, sequence_length] so the model can process it.

# 2. Model Inference (generate the "hidden states" or embeddings):
#    We feed the tokenized input into the ESM-2 model. It outputs a 3D tensor:
#      [batch_size, sequence_length, embedding_dim] → e.g. [1, 35, 1280]
#    This means: for each of the 35 tokens (amino acids + [CLS]), we get a 1280-dimensional vector
#    that captures its meaning based on the entire sequence (like understanding a word in context).

# 3. Embedding Extraction:
#    We extract two types of vectors:
#      - CLS vector (position 0): a single vector meant to summarize the entire sequence
#      - Mean vector: we average all the other vectors (ignoring CLS) to get a smoothed-out view of the sequence

# 4. Feature Fusion (merge the summary + content vectors):
#    We concatenate the CLS vector and the mean vector, so our final embedding includes:
#      - Global summary (CLS)
#      - Averaged local context (mean)
#    This creates a more informative representation than using only one of them.

# 5. Normalization (make comparison fair):
#    We convert the final vector into a unit vector — meaning its length becomes 1.
#    This is essential for cosine similarity to work properly — we want to compare direction, not magnitude.

# Output:
#    A NumPy array representing the final embedding for the input protein sequence.
#    This vector can now be used for comparing sequences, clustering, or feeding into ML models.


def get_embedding(sequence: str):
    tokens = tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)

    cls_vec = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    mean_vec = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # Skip [CLS]

    # Concatenate CLS + mean
    embedding = torch.cat([cls_vec, mean_vec], dim=-1).squeeze()

    # Normalize the embedding (unit vector)
    embedding = embedding / embedding.norm()

    return embedding.numpy()
