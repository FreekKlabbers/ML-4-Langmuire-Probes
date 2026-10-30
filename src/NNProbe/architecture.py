import inspect
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torch.utils.data import random_split


class MLPRegressorTorch(nn.Module):
    """
    Feedforward network with ReLU activations, and a single linear output unit for regression.
    """

    WEIGHTS_FILENAME = "model.pt"
    LEGACY_WEIGHTS_FILENAME = "mlp_regressor_model.pth"
    CONFIG_FILENAME = "config.json"

    def __init__(self, input_dim: int, hidden_layer_sizes=(128, 64, 32, 16)):
        """
        Initializes the MLPRegressorTorch model.
        """
        super().__init__()
        self._capture_config(locals())  # capture configuration for Hugging Face

        layers = []
        prev_dim = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.Tanh())
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def _capture_config(self, local_vars: dict):
        """
        Wraps the local variables of the __init__ method to capture the model configuration for saving/loading.
        """
        sig = inspect.signature(self.__class__.__init__)
        param_names = [p for p in sig.parameters if p != "self"]
        self.config = {k: local_vars[k] for k in param_names}

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def save_pretrained(self, save_dir: str | Path):
        """
        Save model weights and configuration to a directory.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        weights_path = save_dir / self.WEIGHTS_FILENAME
        legacy_weights_path = save_dir / self.LEGACY_WEIGHTS_FILENAME

        torch.save(self.state_dict(), weights_path)
        torch.save(self.state_dict(), legacy_weights_path)

        with open(save_dir / self.CONFIG_FILENAME, "w") as f:
            json.dump(self.config, f, indent=2)

        print(f"Saved weights + config to {save_dir}")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        revision: str | None = None,
        map_location: str | None = None,
        **override_kwargs,
    ):
        """
        Load model weights and configuration from a Hugging Face repository.

        Parameters
        ----------
        repo_id : str
            The repository ID on Hugging Face Hub.
        revision : str, optional
            The specific revision (branch, tag, or commit) to load. Defaults to the latest
        map_location : str, optional
            The device to map the model to. If None, defaults to 'cpu'.
        override_kwargs : dict
            Any additional keyword arguments to override the model's configuration.

        Returns
        -------
        MLPRegressorTorch
            An instance of the MLPRegressorTorch model with loaded weights and configuration.
        """
        local_dir = Path(repo_id)
        if local_dir.is_dir():
            weights_path = local_dir / cls.WEIGHTS_FILENAME
            config_path = local_dir / cls.CONFIG_FILENAME
        else:
            weights_path = hf_hub_download(
                repo_id=repo_id, filename=cls.WEIGHTS_FILENAME, revision=revision
            )
            config_path = hf_hub_download(
                repo_id=repo_id, filename=cls.CONFIG_FILENAME, revision=revision
            )

        with open(config_path) as f:
            config = json.load(f)
        config.update(override_kwargs)

        # hidden_layer_sizes round-trips through JSON as a list; restore tuple.
        for k, v in config.items():
            if isinstance(v, list):
                config[k] = tuple(v)

        model = cls(**config)
        state_dict = torch.load(weights_path, map_location=map_location or "cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model


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

    # keep the data on the device and slice batches directly; a DataLoader would
    # fetch and collate every sample in Python, which bottlenecks the GPU
    if early_stopping:
        n_val = int(len(X_t) * validation_fraction)
        n_train = len(X_t) - n_val
        generator = torch.Generator().manual_seed(random_state)
        train_idx, val_idx = random_split(
            range(len(X_t)), [n_train, n_val], generator=generator
        )
        train_idx = torch.as_tensor(list(train_idx))
        val_idx = torch.as_tensor(list(val_idx))
        X_train, y_train = X_t[train_idx].to(device), y_t[train_idx].to(device)
        X_val, y_val = X_t[val_idx].to(device), y_t[val_idx].to(device)
    else:
        X_train, y_train = X_t.to(device), y_t.to(device)

    n_train = len(X_train)
    shuffle_generator = torch.Generator(device=device).manual_seed(random_state)

    model = MLPRegressorTorch(
        input_dim=X.shape[1], hidden_layer_sizes=hidden_layer_sizes
    )
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate_init,
        weight_decay=alpha,
        fused=device.startswith("cuda"),
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(max_iter):
        model.train()
        # accumulate on the device; calling loss.item() every step would force a sync
        running_loss = torch.zeros((), device=device)
        perm = torch.randperm(n_train, device=device, generator=shuffle_generator)

        for i in range(0, n_train, batch_size):
            batch = perm[i : i + batch_size]
            xb, yb = X_train[batch], y_train[batch]

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach() * xb.size(0)

        train_loss = running_loss.item() / n_train
        history["train_loss"].append(train_loss)

        # early stopping check
        if early_stopping:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val)
                val_loss = criterion(val_preds, y_val).item()
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss * 0.99:  # 1% relative tolerance (inf-safe)
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
