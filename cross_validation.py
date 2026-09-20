import os

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from NNProbe import train_mlp_regressor
from NNProbe.pipeline import FEATURE_COLUMNS, engineer_features, physical_log

N_SPLITS = 5
FILE_PATH = "data/bryant08-precalc-ML.npy"
RANDOM_STATE = 42


def load_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load the cleaned, engineered features and log-transformed target."""
    results = np.load(file_path)
    mask = ~((results[:, 0] == 0.0) * (results[:, 1] == 0.0))
    frame = pd.DataFrame(results[mask], columns=["eta", "J", "R", "delta", "Cp"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    features = engineer_features(frame)
    target = np.log(frame["J"].to_numpy()).reshape(-1, 1)
    return features.to_numpy(), target


def make_preprocessor() -> ColumnTransformer:
    feature_pipeline = Pipeline(
        [
            ("log", FunctionTransformer(physical_log, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("wide_range", feature_pipeline, list(range(len(FEATURE_COLUMNS))))
        ]
    )


def evaluate_fold(model, target_scaler, X_test, y_test) -> tuple[float, float]:
    model.eval()
    X_test_tensor = torch.as_tensor(X_test, dtype=torch.float32)
    model_device = next(model.parameters()).device
    with torch.no_grad():
        predictions = model(X_test_tensor.to(model_device)).cpu().numpy()

    predicted_log = target_scaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
    actual_log = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    predicted = np.exp(predicted_log)
    actual = np.exp(actual_log)
    relative_error = np.abs((predicted - actual) / (actual + 1e-15))
    return r2_score(actual_log, predicted_log), float(np.mean(relative_error))


def cross_validate(file_path: str = FILE_PATH) -> None:
    X, y = load_data(file_path)
    fold_metrics = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_indices, test_indices) in enumerate(splitter.split(X), start=1):
        preprocessor = make_preprocessor()
        X_train = np.asarray(preprocessor.fit_transform(X[train_indices]))
        X_test = np.asarray(preprocessor.transform(X[test_indices]))

        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y[train_indices]).ravel()
        y_test = target_scaler.transform(y[test_indices]).ravel()

        print(f"\nFold {fold}/{N_SPLITS}")
        model, _ = train_mlp_regressor(
            X_train,
            y_train,
            hidden_layer_sizes=(512, 256, 128, 64),
            alpha=2e-6,
            batch_size=256,
            learning_rate_init=0.00005,
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=RANDOM_STATE + fold,
        )

        r2, mean_relative_error = evaluate_fold(model, target_scaler, X_test, y_test)
        fold_metrics.append((r2, mean_relative_error))
        print(f"Fold R² (log-space): {r2:.6f}")
        print(f"Fold mean relative error: {mean_relative_error:.2%}")

    metrics = np.asarray(fold_metrics)
    print("\n" + "=" * 40)
    print(f"{N_SPLITS}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 40)
    print(f"R² (log-space): {metrics[:, 0].mean():.6f} +/- {metrics[:, 0].std():.6f}")
    print(
        f"Mean relative error: {metrics[:, 1].mean():.2%} +/- {metrics[:, 1].std():.2%}"
    )


if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Data file {FILE_PATH} not found.")
    cross_validate(FILE_PATH)
