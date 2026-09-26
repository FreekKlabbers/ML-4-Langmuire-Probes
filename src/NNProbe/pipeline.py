import os
from pathlib import Path
from typing import TypedDict

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _resolve_output_path(path: str, default_name: str) -> Path:
    resolved = Path(path)
    if resolved.suffix == "" or resolved.exists() and resolved.is_dir():
        return resolved / default_name
    return resolved


class PreprocessedData(TypedDict):
    preprocessor: ColumnTransformer
    target_scaler: StandardScaler
    X_train: np.ndarray
    X_test: np.ndarray
    y_train_scaled: np.ndarray
    y_test_scaled: np.ndarray


FEATURE_COLUMNS = ["eta", "R", "delta", "Cp", "P0", "sqrt"]


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add the physics-derived features used by the trained model."""
    features = data.copy()
    features["s_ld"] = 1.2 * (features["eta"] ** 0.75)
    features["P0"] = np.exp(-features["delta"] * features["s_ld"])
    features["sqrt"] = np.sqrt(features["eta"] / features["R"])
    return features[FEATURE_COLUMNS]


def evaluation_mask(data: pd.DataFrame) -> pd.Series:
    """Return the rows in the parameter range used for evaluation."""
    return (
        data["eta"].between(1, 50)
        & data["delta"].ge(1e-3)
        & data["Cp"].ge(5)
        & data["R"].ge(10)
    )


def filter_settings(results, param: str, min=None, max=None):
    """Simple function to filter the results based on a specific parameter and its min/max values."""
    params = {"eta": 0, "J": 1, "R": 2, "delta": 3, "Cp": 4}
    if param not in params:
        raise ValueError(f"Invalid parameter. Choose from {list(params.keys())}")

    filt = results[:, params[param]]
    min_val = min if min is not None else -np.inf
    max_val = max if max is not None else np.inf
    return results[(filt >= min_val) & (filt <= max_val)]


def preprocess_data(
    file_path: str = "data/bryant08-precalc-ML.npy",
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreprocessedData:
    """
    Preprocess the data for training and testing.
    This function loads the data, filters it, and prepares it for training and testing.

    Parameters
    ----------
    file_path (str):
        Path to the .npy data file.

    Returns
    -------
    dict[str, object]:
        A dictionary containing the preprocessor, target scaler, training and testing data.
        "preprocessor": preprocessor,
        "target_scaler": target_scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_scaled": y_train_scaled,
        "y_test_scaled": y_test_scaled,

    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found.")

    # Load & Clean
    results = np.load(file_path)

    print(f"Loaded {len(results)} rows from {file_path}.")

    # Remove rows where both eta and J are zero
    mask = ~((results[:, 0] == 0.0) * (results[:, 1] == 0.0))
    df = pd.DataFrame(results[mask], columns=["eta", "J", "R", "delta", "Cp"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    test_df = test_df.loc[evaluation_mask(test_df)]
    print(
        f"Training on {len(train_df)} rows; evaluating on "
        f"{len(test_df)} rows in the filtered parameter range."
    )

    X_train = engineer_features(train_df)
    X_test = engineer_features(test_df)
    y_train = np.asarray(np.log(train_df["J"])).reshape(-1, 1)
    y_test = np.asarray(np.log(test_df["J"])).reshape(-1, 1)

    feat_pipeline = Pipeline(
        [
            (
                "clip",
                FunctionTransformer(
                    np.clip, kw_args={"a_min": 1e-15, "a_max": None}, validate=True
                ),
            ),
            ("log", FunctionTransformer(np.log10, validate=True)),
            ("scaler", StandardScaler()),
        ]
    )

    # Feature scaling
    preprocessor = ColumnTransformer(
        transformers=[("wide_range", feat_pipeline, FEATURE_COLUMNS)]
    )
    X_train = np.asarray(preprocessor.fit_transform(X_train))
    X_test = np.asarray(preprocessor.transform(X_test))

    # Target Scaling
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train).ravel()
    y_test_scaled = target_scaler.transform(y_test).ravel()

    return {
        "preprocessor": preprocessor,
        "target_scaler": target_scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train_scaled": y_train_scaled,
        "y_test_scaled": y_test_scaled,
    }


def save_preprocessor(
    preprocessor: ColumnTransformer,
    target_scaler: StandardScaler,
    preprocessor_path: str = "src/NNProbe/weights/preprocessor.pkl",
    target_scaler_path: str = "src/NNProbe/weights/target_scaler.pkl",
):
    """
    Save the preprocessor and target scaler to disk.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessor to be saved.
    target_scaler : StandardScaler
        The target scaler to be saved.
    preprocessor_path : str
        Path to save the preprocessor.
    target_scaler_path : str
        Path to save the target scaler.
    """
    res_preprocessor_path = _resolve_output_path(preprocessor_path, "preprocessor.pkl")
    res_target_scaler_path = _resolve_output_path(
        target_scaler_path, "target_scaler.pkl"
    )

    res_preprocessor_path.parent.mkdir(parents=True, exist_ok=True)
    res_target_scaler_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, res_preprocessor_path)
    joblib.dump(target_scaler, res_target_scaler_path)


def load_preprocessor(
    preprocessor_path: str = "src/NNProbe/weights/preprocessor.pkl",
    target_scaler_path: str = "src/NNProbe/weights/target_scaler.pkl",
) -> tuple[ColumnTransformer, StandardScaler]:
    """
    Load the preprocessor and target scaler from disk.

    Parameters
    ----------
    preprocessor_path : str
        Path to load the preprocessor from.
    target_scaler_path : str
        Path to load the target scaler from.

    Returns
    -------
    tuple[ColumnTransformer, StandardScaler]
        A tuple containing the loaded preprocessor and target scaler.
    """
    res_preprocessor_path = _resolve_output_path(preprocessor_path, "preprocessor.pkl")
    res_target_scaler_path = _resolve_output_path(
        target_scaler_path, "target_scaler.pkl"
    )

    return joblib.load(res_preprocessor_path), joblib.load(res_target_scaler_path)
