import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from .architecture import MLPRegressorTorch
from .pipeline import engineer_features, load_preprocessor


class Predictor:
    """
    A class to handle loading the preprocessor, target scaler, and trained model for making predictions.
    """

    def __init__(self, weight_path: str = "src/NNProbe/weights"):
        """
        Initializes the Predictor by loading the preprocessor, target scaler, and trained model.

        Parameters
        ----------
        weight_path : str
            The path to the directory containing the model weights and preprocessor files.
        """
        self.preprocessor: ColumnTransformer
        self.target_scaler: StandardScaler
        self.model: MLPRegressorTorch

        self.weight_path = weight_path

        self.preprocessor, self.target_scaler = load_preprocessor(
            f"{weight_path}/preprocessor.pkl", f"{weight_path}/target_scaler.pkl"
        )
        self.model = MLPRegressorTorch(
            input_dim=6, hidden_layer_sizes=(512, 256, 128, 64)
        )
        self.model.load_state_dict(
            torch.load(f"{weight_path}/mlp_regressor_model.pth", weights_only=True)
        )
        self.model.eval()  # Set the model to evaluation mode

    def predict(self, X):
        """
        Make predictions on new data.

        Parameters
        ----------
        X : array-like
            The input features for which to make predictions.

        Returns
        -------
        array-like
            The predicted values after inverse transforming the scaled predictions.
        """
        X_features = engineer_features(pd.DataFrame(X))
        X_proc = self.preprocessor.transform(X_features)
        device = next(self.model.parameters()).device
        X_tensor = torch.as_tensor(X_proc, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred_scaled = self.model(X_tensor)
        y_pred = self.target_scaler.inverse_transform(
            y_pred_scaled.cpu().numpy().reshape(-1, 1)
        ).ravel()
        return y_pred
