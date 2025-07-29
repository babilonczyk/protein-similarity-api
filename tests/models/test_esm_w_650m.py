import numpy as np
from models.esm_2_650m import get_embedding


def test_get_embedding_shape_and_type():
    # Example short protein sequence
    sequence = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQ"
    embedding = get_embedding(sequence)

    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert embedding.shape[0] in [1280, 2560]
    assert embedding.dtype == np.float32 or embedding.dtype == np.float64
