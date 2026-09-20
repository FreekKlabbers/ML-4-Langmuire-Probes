import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def test_dnn_performance(
    model: torch.nn.Module,
    preprocessor: object,
    target_scaler: StandardScaler,
    X_train: np.ndarray,
    X_test: np.ndarray | pd.DataFrame,
    y_train_scaled: np.ndarray,
    y_test_scaled: np.ndarray,
):
    """
    Test the performance of a DNN model on the provided test data. This function evaluates the model's performance
    by calculating the R² score and relative errors, and prints out the results.

    Parameters
    ----------
    model : object
        The trained DNN model to be evaluated.
    preprocessor : object
        The preprocessor used to transform the input features.
    target_scaler : object
        The scaler used to transform the target variable.
    X_train : pd.DataFrame
        The training input features.
    X_test : pd.DataFrame
        The testing input features.
    y_train_scaled : np.ndarray
        The scaled training target values.
    y_test_scaled : np.ndarray
        The scaled testing target values.
    """

    # The pipeline already returns preprocessed feature arrays.
    X_test_tensor = torch.as_tensor(np.asarray(X_test), dtype=torch.float32)
    model_device = next(model.parameters()).device
    X_test_tensor = X_test_tensor.to(model_device)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_tensor).cpu().numpy()

    y_pred_log = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_log = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    y_pred_orig = np.exp(y_pred_log)
    y_test_orig = np.exp(y_test_log)

    # Determine R² score and relative errors
    r2 = r2_score(y_test_log, y_pred_log)
    rel_errors = np.abs((y_pred_orig - y_test_orig) / (y_test_orig + 1e-15))

    # Print performance metrics
    print("\n" + "=" * 40)
    print("ENHANCED DNN PERFORMANCE")
    print("=" * 40)
    print(f"R² Score (log-space):      {r2:.6f}")
    print(f"Mean Relative Error:       {np.mean(rel_errors):.2%}")
    print(f"Max Relative Error:        {np.max(rel_errors):.2%}")
    print("=" * 40)

    worst_idx = np.argmax(rel_errors)
    print("\nWORST CASE ANALYSIS:")
    print(f"Actual J:     {y_test_orig[worst_idx]:.2e}")
    print(f"Predicted J:  {y_pred_orig[worst_idx]:.2e}")

    if isinstance(X_test, pd.DataFrame):
        worst_case = X_test.iloc[worst_idx]
        print(f"\nat Eta:       {worst_case['eta']:.2e}")
        print(f"At R:        {worst_case['R']:.2e}")
        print(f"At Cp:       {worst_case['Cp']:.2e}")
        print(f"At Delta:    {worst_case['delta']:.2e}")
