import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MLPRegressorTorch(nn.Module):
    """
    Feedforward network with ReLU activations, and a single linear output unit for regression.
    """

    def __init__(self, input_dim: int, hidden_layer_sizes=(128, 64, 32, 16)):
        """
        Initializes the MLPRegressorTorch model.
        """
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_regressor(
    X: np.ndarray,
    y: np.ndarray,
    hidden_layer_sizes=(128, 64, 32, 16),
    alpha: float = 0.0005,
    batch_size: int = 256,
    learning_rate_init: float = 0.001,
    max_iter: int = 2000,
    early_stopping: bool = True,
    n_iter_no_change: int = 20,
    validation_fraction: float = 0.1,
    random_state: int = 42,
    device: str = None,
):
    """
    Trains an MLPRegressorTorch model, mirroring sklearn's MLPRegressor defaults
    for the parameters given. Returns the trained model and loss history.

    Parameters
    ----------
    X : np.ndarray
        Input features for training.
    y : np.ndarray
        Target values for training.
    hidden_layer_sizes : tuple, optional
        The ith element represents the number of neurons in the ith hidden layer.
    alpha : float, optional
        L2 penalty (regularization term) parameter.
    batch_size : int, optional
        Size of minibatches for stochastic optimizers.
    learning_rate_init : float, optional
        The initial learning rate used. It controls the step-size in updating the weights.
    max_iter : int, optional
        Maximum number of iterations. The solver iterates until convergence or this number of iterations.
    early_stopping : bool, optional
        Whether to use early stopping to terminate training when validation score is not improving.
    n_iter_no_change : int, optional
        Number of iterations with no improvement to wait before stopping.
    validation_fraction : float, optional
        The proportion of training data to set aside as validation set for early stopping.
    random_state : int, optional
        Seed for random number generator for reproducibility.
    device : str, optional
        Device to run the training on. If None, will use 'cuda' if available, else 'cpu'.
    
    Returns
    -------
    model : MLPRegressorTorch
        The trained MLPRegressorTorch model.
    history : dict
        Dictionary containing training and validation loss history.
    """

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y, dtype=np.float32), dtype=torch.float32)

    full_dataset = TensorDataset(X_t, y_t)

    if early_stopping:
        n_val = int(len(full_dataset) * validation_fraction)
        n_train = len(full_dataset) - n_val
        generator = torch.Generator().manual_seed(random_state)
        train_dataset, val_dataset = random_split(
            full_dataset, [n_train, n_val], generator=generator
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = MLPRegressorTorch(
        input_dim=X.shape[1], hidden_layer_sizes=hidden_layer_sizes
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate_init, weight_decay=alpha
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_iter):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_samples += xb.size(0)

        train_loss = running_loss / n_samples
        history["train_loss"].append(train_loss)

        # early stopping check
        if early_stopping:
            model.eval()
            with torch.no_grad():
                X_val, y_val = val_dataset[:]
                X_val, y_val = X_val.to(device), y_val.to(device)
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val).item()
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss - 1e-4:  # small tolleranace
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= n_iter_no_change:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no improvement in {n_iter_no_change} epochs). "
                    f"Best val loss: {best_val_loss:.6f}"
                )
                break

        if (epoch + 1) % 50 == 0 or epoch == 0:
            msg = f"Epoch {epoch + 1}/{max_iter} - train_loss: {train_loss:.6f}"
            if early_stopping:
                msg += f" - val_loss: {val_loss:.6f}"
            print(msg)

    # restore best weights found during early stopping
    if early_stopping and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history


    