import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from NNProbe import MLPRegressorTorch, log_run, train_mlp_regressor
from NNProbe.pipeline import (
    FEATURE_COLUMNS,
    engineer_features,
    evaluation_mask,
    make_preprocessor,
)

N_SPLITS = 5
FILE_PATH = "data/bryant08-precalc-ML.npy"
RANDOM_STATE = 42
EPOCHS = 2500
IN_RANGE_WEIGHT = 1.0  # loss weight of training rows inside the evaluation range
HIDDEN_LAYER_SIZES = (512, 256, 128, 64)
ALPHA = 0.0
BATCH_SIZE = 512
LEARNING_RATE_INIT = 2.5e-4
WARMUP_FRACTION = 0.02

RUN_SETTINGS = {
    "data file": FILE_PATH,
    "folds": N_SPLITS,
    "random state": f"{RANDOM_STATE} (split), {RANDOM_STATE} + fold (training)",
    "features": FEATURE_COLUMNS,
    "epochs": EPOCHS,
    "in-range weight": IN_RANGE_WEIGHT,
    "hidden layer sizes": HIDDEN_LAYER_SIZES,
    "alpha (weight decay)": ALPHA,
    "batch size": BATCH_SIZE,
    "learning rate init": LEARNING_RATE_INIT,
    "warmup fraction": WARMUP_FRACTION,
    "architecture": MLPRegressorTorch(len(FEATURE_COLUMNS), HIDDEN_LAYER_SIZES),
}


def load_data(file_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load all cleaned rows plus the mask for the evaluation parameter range."""
    results = np.load(file_path)
    mask = ~((results[:, 0] == 0.0) * (results[:, 1] == 0.0))
    frame = pd.DataFrame(results[mask], columns=["eta", "J", "R", "delta", "Cp"])
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    features = engineer_features(frame)
    target = np.log(frame["J"].to_numpy()).reshape(-1, 1)
    return features, target, evaluation_mask(frame).to_numpy()


def evaluate_fold(model, target_scaler, X_test, y_test) -> tuple[float, float, float]:
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
    return (
        r2_score(actual_log, predicted_log),
        float(np.mean(relative_error)),
        float(np.max(relative_error)),
    )


def cross_validate(file_path: str = FILE_PATH) -> None:
    X, y, evaluation_rows = load_data(file_path)
    fold_metrics = []
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_indices, test_indices) in enumerate(splitter.split(X), start=1):
        test_indices = test_indices[evaluation_rows[test_indices]]
        if len(test_indices) == 0:
            raise ValueError(f"Fold {fold} has no rows in the evaluation range.")

        preprocessor = make_preprocessor()
        X_train = np.asarray(preprocessor.fit_transform(X.iloc[train_indices]))
        X_test = np.asarray(preprocessor.transform(X.iloc[test_indices]))

        target_scaler = StandardScaler()
        y_train = target_scaler.fit_transform(y[train_indices]).ravel()
        y_test = target_scaler.transform(y[test_indices]).ravel()

        print(f"\nFold {fold}/{N_SPLITS}")
        model, _ = train_mlp_regressor(
            X_train,
            y_train,
            in_range=evaluation_rows[train_indices],
            in_range_weight=IN_RANGE_WEIGHT,
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            alpha=ALPHA,
            batch_size=BATCH_SIZE,
            learning_rate_init=LEARNING_RATE_INIT,
            epochs=EPOCHS,
            warmup_fraction=WARMUP_FRACTION,
            random_state=RANDOM_STATE + fold,
        )

        r2, mean_relative_error, max_relative_error = evaluate_fold(
            model, target_scaler, X_test, y_test
        )
        fold_metrics.append((r2, mean_relative_error, max_relative_error))
        print(f"Fold R² (log-space): {r2:.6f}")
        print(f"Fold mean relative error: {mean_relative_error:.2%}")
        print(f"Fold max relative error: {max_relative_error:.2%}")

    metrics = np.asarray(fold_metrics)
    print("\n" + "=" * 40)
    print(f"{N_SPLITS}-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 40)
    print(f"R² (log-space): {metrics[:, 0].mean():.6f} +/- {metrics[:, 0].std():.6f}")
    print(
        f"Mean relative error: {metrics[:, 1].mean():.2%} +/- {metrics[:, 1].std():.2%}"
    )
    print(
        f"Max relative error: {metrics[:, 2].mean():.2%} +/- {metrics[:, 2].std():.2%}"
    )


if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Data file {FILE_PATH} not found.")
    with log_run("cross_validation", RUN_SETTINGS):
        cross_validate(FILE_PATH)
