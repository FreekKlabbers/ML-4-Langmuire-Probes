"""Explain Langmuir probe predictions with SHAP feature attributions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from NNProbe.pipeline import FEATURE_COLUMNS, engineer_features
from NNProbe.predict import Predictor

RAW_COLUMNS = ["eta", "R", "delta", "Cp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate SHAP values for Langmuir probe input samples."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        help="CSV containing eta, R, delta, and Cp columns. Uses the full dataset when omitted.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("shap_values.csv"),
        help="Where to save predictions and SHAP values (default: shap_values.csv).",
    )
    parser.add_argument(
        "--background-size",
        type=int,
        default=25,
        help="Maximum number of background rows used by SHAP (default: 256).",
    )
    parser.add_argument(
        "--explain-batch-size",
        type=int,
        default=10,
        help="Number of input rows explained per SHAP batch (default: 1024).",
    )
    parser.add_argument(
        "--plot-sample-size",
        type=int,
        default=100,
        help="Maximum rows used for plots; the CSV still contains all rows (default: 10000).",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("shap_plots"),
        help="Directory for SHAP plots (default: shap_plots).",
    )
    return parser.parse_args()


def load_dataset(data_path: Path) -> pd.DataFrame:
    results = np.load(data_path)
    mask = ~((results[:, 0] == 0.0) * (results[:, 1] == 0.0))
    data = pd.DataFrame(results[mask], columns=["eta", "J", "R", "delta", "Cp"])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data[RAW_COLUMNS].copy()


def load_input(path: Path | None, data_path: Path) -> pd.DataFrame:
    data = load_dataset(data_path) if path is None else pd.read_csv(path)
    missing = sorted(set(RAW_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")
    return data[RAW_COLUMNS].copy()


def load_background(data_path: Path, preprocessor, background_size: int) -> np.ndarray:
    engineered = engineer_features(load_dataset(data_path))
    transformed = np.asarray(preprocessor.transform(engineered), dtype=np.float32)
    if background_size < 1:
        raise ValueError("--background-size must be at least 1")
    if len(transformed) > background_size:
        indices = np.linspace(0, len(transformed) - 1, background_size, dtype=int)
        transformed = transformed[indices]
    return transformed


def calculate_shap_values(
    explainer, transformed: np.ndarray, batch_size: int
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("--explain-batch-size must be at least 1")
    batches = []
    for start in range(0, len(transformed), batch_size):
        batch = torch.from_numpy(transformed[start : start + batch_size])
        values = explainer.shap_values(batch)
        if isinstance(values, list):
            values = values[0]
        batches.append(np.asarray(values).reshape(len(batch), len(FEATURE_COLUMNS)))
    return np.concatenate(batches, axis=0)


class OriginalUnitsModel(torch.nn.Module):
    """Expose the saved network with predictions converted back to J units."""

    def __init__(self, predictor: Predictor):
        super().__init__()
        self.model = predictor.model
        self.target_scaler = predictor.target_scaler

    def forward(self, inputs):
        scaled_prediction = self.model(inputs).reshape(-1, 1)
        mean = torch.as_tensor(
            self.target_scaler.mean_, dtype=inputs.dtype, device=inputs.device
        )
        scale = torch.as_tensor(
            self.target_scaler.scale_, dtype=inputs.dtype, device=inputs.device
        )
        return scaled_prediction * scale + mean


def save_plots(
    shap_values: np.ndarray,
    engineered: pd.DataFrame,
    expected_value: float,
    predictions: np.ndarray,
    plot_dir: Path,
    plot_sample_size: int,
) -> None:
    if plot_sample_size < 1:
        raise ValueError("--plot-sample-size must be at least 1")
    if len(shap_values) > plot_sample_size:
        plot_indices = np.linspace(0, len(shap_values) - 1, plot_sample_size, dtype=int)
    else:
        plot_indices = np.arange(len(shap_values))
    plot_values = shap_values[plot_indices]
    plot_engineered = engineered.iloc[plot_indices]
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_features = plot_engineered.copy()
    plot_features.columns = FEATURE_COLUMNS

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        plot_values,
        plot_features,
        feature_names=FEATURE_COLUMNS,
        max_display=len(FEATURE_COLUMNS),
        show=False,
    )
    plt.title("SHAP Feature Impact on Predicted J")
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_beeswarm.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    shap.summary_plot(
        plot_values,
        plot_features,
        feature_names=FEATURE_COLUMNS,
        plot_type="bar",
        max_display=len(FEATURE_COLUMNS),
        show=False,
    )
    plt.title("Mean Absolute SHAP Importance")
    plt.xlabel("mean(|SHAP value|) [J]")
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_importance_bar.png", dpi=180, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(
        values=plot_values[0],
        base_values=expected_value,
        data=plot_features.iloc[0].to_numpy(),
        feature_names=FEATURE_COLUMNS,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=len(FEATURE_COLUMNS), show=False)
    plt.title(
        f"SHAP Explanation for Sample 1 (prediction={predictions[plot_indices[0]]:.4f} J)"
    )
    plt.tight_layout()
    plt.savefig(plot_dir / "shap_waterfall_sample_1.png", dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_path = (
        args.output_csv
        if args.output_csv.is_absolute()
        else project_root / args.output_csv
    )
    plot_dir = (
        args.plot_dir if args.plot_dir.is_absolute() else project_root / args.plot_dir
    )
    data_path = project_root / "data" / "bryant08-precalc-ML.npy"
    weights = project_root / "src" / "NNProbe" / "weights"
    predictor = Predictor(weight_path=str(weights))

    inputs = load_input(args.input_csv, data_path)
    engineered = engineer_features(inputs)
    transformed = np.asarray(
        predictor.preprocessor.transform(engineered), dtype=np.float32
    )
    background = load_background(
        data_path,
        predictor.preprocessor,
        args.background_size,
    )

    model = OriginalUnitsModel(predictor)
    explainer = shap.DeepExplainer(model, torch.from_numpy(background))
    shap_values = calculate_shap_values(explainer, transformed, args.explain_batch_size)
    predictions = predictor.predict(inputs)
    expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

    output = inputs.copy()
    output["prediction_J"] = predictions
    for index, feature in enumerate(FEATURE_COLUMNS):
        output[f"shap_{feature}"] = shap_values[:, index]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    save_plots(
        shap_values,
        engineered,
        expected_value,
        predictions,
        plot_dir,
        args.plot_sample_size,
    )

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=FEATURE_COLUMNS
    ).sort_values(ascending=False)
    print(f"Saved SHAP values to {output_path}")
    print(f"Saved SHAP plots to {plot_dir}")
    print("Mean absolute SHAP importance in J:")
    for feature, value in importance.items():
        print(f"  {feature}: {value:.6g}")


if __name__ == "__main__":
    main()
