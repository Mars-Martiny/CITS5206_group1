import torch
from torch.utils.data import DataLoader, TensorDataset

from modules.models import BiLSTMFeatureExtractor, SVMClassifierWrapper, BiLSTMClassifier


from pipeline.run_feature_inference import run_feature_pipeline


def make_dataloader():
    D = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [1, 2, 4, 0, 0],
            [5, 6, 7, 8, 0],
            [5, 6, 8, 9, 0],
            [2, 3, 4, 5, 6],
            [2, 3, 4, 7, 8],
            [1, 3, 5, 0, 0],
            [2, 4, 6, 0, 0],
        ],
        dtype=torch.long,
    )

    DL = torch.tensor([3, 3, 4, 4, 5, 5, 3, 3])
    Energy = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    Risk = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])

    dataset = TensorDataset(D, DL, Energy, Risk)
    return DataLoader(dataset, batch_size=2)


def test_svm_bilstm_feature_pipeline_logs_to_leaderboard(tmp_path):
    train_dl = make_dataloader()
    test_dl = make_dataloader()

    bilstm_model = BiLSTMClassifier(
        vocab_size=20,
        embedding_dim=8,
        hidden_dim=16,
        num_classes=2,
    )

    feature_extractor = BiLSTMFeatureExtractor(bilstm_model)
    classifier = SVMClassifierWrapper(
        kernel="linear",
        C=1.0,
        probability=True,
    )

    leaderboard_dir = tmp_path / "leaderboard"
    trained_models_dir = tmp_path / "trained_models"

    result = run_feature_pipeline(
        feature_extractor=feature_extractor,
        classifier=classifier,
        train_dl=train_dl,
        test_dl=test_dl,
        target="energy",
        model_type="SVM_BiLSTM_Features",
        run_name="test_svm_bilstm_leaderboard",
        parent_dir=str(trained_models_dir),
        leaderboard_dir=str(leaderboard_dir),
        log_leaderboard=True,
    )

    leaderboard_file = leaderboard_dir / "leaderboard.csv"

    assert leaderboard_file.exists()
    assert result["metrics"]["accuracy"] >= 0
    assert result["model_path"]
    assert result["summary_path"]
    assert "predictions" in result
    assert len(result["predictions"]) == 8