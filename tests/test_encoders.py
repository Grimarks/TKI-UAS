import numpy as np

from hybrid_ir.config import EncoderConfig
from hybrid_ir.encoders import HashingEncoder


def _enc():
    return HashingEncoder(EncoderConfig(backend="hashing", hashing_dim=256))


def test_unit_norm_and_shape():
    v = _enc().encode_documents(["neural machine translation", "graph networks"])
    assert v.shape == (2, 256)
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_deterministic_across_instances():
    a = _enc().encode_queries(["keyphrase extraction"])
    b = _enc().encode_queries(["keyphrase extraction"])
    assert np.array_equal(a, b)


def test_similar_text_closer_than_unrelated():
    e = _enc()
    q, near, far = e.encode_queries(["neural machine translation"])[0], e.encode_documents(["machine translation with neural networks"])[0], e.encode_documents(["tumor segmentation in MRI"])[0]
    assert q @ near > q @ far


def test_empty_batch():
    assert _enc().encode_documents([]).shape == (0, 256)
