from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from trajectory_calibration.uq.eigenscore import (
    compute_eigenscore,
    compute_eigenscore_gram,
    compute_logdet,
    compute_umpire_metric,
    normalize_embedding,
)
from trajectory_calibration.uq.semantic_entropy import (
    EntailmentDeberta,
    FastStringEntailment,
    cluster_assignment_entropy,
    compute_cluster_assignment_entropy,
    compute_semantic_entropy,
    get_semantic_ids,
    predictive_entropy,
)


def test_entailment_deberta_mocked() -> None:
    """Verify EntailmentDeberta forward pass and logit class decoding (0=Contradiction, 1=Neutral, 2=Entailment)."""
    with (
        patch("transformers.AutoTokenizer.from_pretrained") as mock_tok_cls,
        patch("transformers.AutoModelForSequenceClassification.from_pretrained") as mock_model_cls,
    ):
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok_cls.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model_cls.return_value = mock_model

        deberta = EntailmentDeberta(device="cpu")

        # 1. Contradiction -> largest index 0
        mock_model.return_value = MagicMock(logits=torch.tensor([[10.0, -2.0, -2.0]]))
        assert deberta.check_implication("A dog is running", "No animals are present") == 0

        # 2. Neutral -> largest index 1
        mock_model.return_value = MagicMock(logits=torch.tensor([[-2.0, 10.0, -2.0]]))
        assert deberta.check_implication("A dog is running", "The dog is happy") == 1

        # 3. Entailment -> largest index 2
        mock_model.return_value = MagicMock(logits=torch.tensor([[-2.0, -2.0, 10.0]]))
        assert deberta.check_implication("A dog is running in the park", "There is a dog") == 2

        # 4. Test get_semantic_ids using DeBERTa model
        # With mutual entailment (logits predicting 2), answers should cluster together
        ids = get_semantic_ids(["answer A", "answer B"], model=deberta, strict_entailment=True)
        assert ids == [0, 0]


def test_get_semantic_ids_and_entropy() -> None:
    matcher = FastStringEntailment()
    strings = ["dog", "dog", "cat", "dog.", "bird"]
    ids = get_semantic_ids(strings, model=matcher)
    assert len(ids) == len(strings)
    # "dog" and "dog." should be clustered together
    assert ids[0] == ids[1] == ids[3]
    assert ids[2] != ids[0]  # cat
    assert ids[4] != ids[0]  # bird

    lps = [-0.1, -0.2, -0.5, -0.15, -0.8]
    se = compute_semantic_entropy(ids, lps)
    assert se >= 0.0

    pe = predictive_entropy(lps)
    assert pe > 0.0


def test_cluster_assignment_entropy() -> None:
    # All same cluster -> 0 entropy
    ids_uniform = [0, 0, 0, 0]
    h = compute_cluster_assignment_entropy(ids_uniform)
    assert pytest.approx(h, abs=1e-5) == 0.0

    # Equal split
    ids_split = [0, 1]
    h_split = cluster_assignment_entropy(ids_split)
    assert pytest.approx(h_split, abs=1e-5) == np.log(2.0)


def test_eigenscore_and_umpire() -> None:
    # Identical vectors (rank 1)
    vecs = np.ones((5, 10))
    normed = normalize_embedding(vecs)
    score_cov = compute_eigenscore(normed)
    score_gram = compute_eigenscore_gram(normed)
    assert isinstance(score_cov, float)
    assert isinstance(score_gram, float)

    k_mat = np.eye(4)
    logdet = compute_logdet(k_mat, alpha=1e-8)
    assert pytest.approx(logdet, abs=1e-3) == 0.0

    umpire = compute_umpire_metric(vecs, [-0.1, -0.2, -0.3, -0.4, -0.5])
    assert isinstance(umpire, float)
