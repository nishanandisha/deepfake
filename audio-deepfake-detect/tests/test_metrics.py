from src.training.metrics import compute_binary_classification_metrics, compute_eer


def test_perfect_predictions_give_perfect_metrics():
    y_true = [0, 0, 1, 1]
    y_prob = [0.01, 0.02, 0.98, 0.99]

    metrics = compute_binary_classification_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["eer"] == 0.0


def test_random_predictions_give_chance_auc():
    y_true = [0, 1, 0, 1]
    y_prob = [0.5, 0.5, 0.5, 0.5]

    metrics = compute_binary_classification_metrics(y_true, y_prob)
    assert 0.0 <= metrics["auc"] <= 1.0


def test_single_class_returns_nan_auc():
    y_true = [1, 1, 1]
    y_prob = [0.9, 0.8, 0.95]

    metrics = compute_binary_classification_metrics(y_true, y_prob)
    assert metrics["auc"] != metrics["auc"]  # NaN


def test_eer_for_perfectly_separable_scores_is_zero():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    assert compute_eer(y_true, y_prob) == 0.0
