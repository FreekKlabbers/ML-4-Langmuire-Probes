import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch import nn


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
    in_range: np.ndarray | None = None,
    in_range_weight: float = 1.0,
    hidden_layer_sizes=(128, 64, 32, 16),
    alpha: float = 0.0,
    batch_size: int = 512,
    learning_rate_init: float = 2.5e-4,
    epochs: int = 2000,
    warmup_fraction: float = 0.02,
    random_state: int = 42,
    device: str = None,
):
    """
    Trains an MLPRegressorTorch model on all of X for a fixed number of epochs with a
    weighted L1 loss and a warmup + cosine LR schedule that ends at 0, so the final
    weights are the converged ones. There is no validation split or early stopping.
    Returns the trained model and loss history.

    Parameters
    ----------
    X : np.ndarray
        Input features for training.
    y : np.ndarray
        Target values for training (the standardized log J).
    in_range : np.ndarray, optional
        Boolean mask of the training rows inside the evaluation range, from
        `evaluation_mask` on the raw feature frame. Ignored when None.
    in_range_weight : float, optional
        Loss weight of the rows in `in_range`; all other rows have weight 1.
    hidden_layer_sizes : tuple, optional
        The ith element represents the number of neurons in the ith hidden layer.
    alpha : float, optional
        L2 penalty (Adam weight decay).
    batch_size : int, optional
        Size of minibatches. The last partial batch of each epoch is dropped.
    learning_rate_init : float, optional
        Peak learning rate, reached at the end of the warmup.
    epochs : int, optional
        Number of passes over the training data.
    warmup_fraction : float, optional
        Fraction of the training steps spent on linear LR warmup, before the cosine decay.
    random_state : int, optional
        Seed for random number generator for reproducibility.
    device : str, optional
        Device to run the training on. If None, will use 'cuda' if available, else 'cpu'.
        On CUDA the whole training step runs as a captured CUDA graph.

    Returns
    -------
    model : MLPRegressorTorch
        The trained MLPRegressorTorch model.
    history : dict
        Dictionary containing the training loss history per epoch.
    """

    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda_graph = device.startswith("cuda")
    print(f"Using device: {device}")

    # keep the data on the device and slice batches directly; a DataLoader would
    # fetch and collate every sample in Python, which bottlenecks the GPU
    X_train = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    y_train = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
    if in_range is None:
        weights = np.ones(len(X_train))
    else:
        weights = np.where(np.asarray(in_range, dtype=bool), in_range_weight, 1.0)
    w_train = torch.tensor(weights, dtype=torch.float32, device=device)

    n_train = len(X_train)
    n_batches = n_train // batch_size  # drop the last partial batch: static shapes
    if n_batches == 0:
        raise ValueError(f"Need at least batch_size={batch_size} rows, got {n_train}.")
    total_steps = epochs * n_batches
    shuffle_generator = torch.Generator(device=device).manual_seed(random_state)

    model = MLPRegressorTorch(
        input_dim=X.shape[1], hidden_layer_sizes=hidden_layer_sizes
    )
    model.to(device)

    # a captured graph reads the LR from a device tensor; eager Adam takes a float
    lr_t = torch.tensor(0.0, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr_t if use_cuda_graph else 0.0,
        betas=(0.9, 0.99),
        weight_decay=alpha,
        foreach=True,
        capturable=use_cuda_graph,
    )

    def set_lr(lr: float):
        if use_cuda_graph:
            lr_t.fill_(lr)
        else:
            for group in optimizer.param_groups:
                group["lr"] = lr

    def lr_at(step: int) -> float:
        # linear warmup, then cosine decay to exactly 0 at the last step
        frac = (step + 1) / total_steps
        if frac < warmup_fraction:
            return learning_rate_init * frac / warmup_fraction
        progress = (frac - warmup_fraction) / (1 - warmup_fraction)
        return learning_rate_init * 0.5 * (1 + math.cos(math.pi * progress))

    # weighted L1: on log J, |error| ~ relative error, the evaluation metric
    def criterion(preds, target, weight):
        return (weight * (preds - target).abs()).sum() / weight.sum()

    if use_cuda_graph:
        # The whole training step (gather, forward, backward, Adam) is captured in a
        # CUDA graph, so the LR and batch indices live in static device tensors.
        static_idx = torch.arange(batch_size, device=device)

        def train_step():
            xb, yb, wb = X_train[static_idx], y_train[static_idx], w_train[static_idx]
            loss = criterion(model(xb), yb, wb)
            loss.backward()
            optimizer.step()
            return loss.detach()

        # warm up on a side stream before capture; lr_t is 0 so the weights do not move
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                train_step()
        torch.cuda.current_stream().wait_stream(side_stream)

        # grads are None at capture, so each replay overwrites them instead of accumulating
        graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            static_loss = train_step()

        def run_step(batch):
            static_idx.copy_(batch)
            graph.replay()
            return static_loss

    else:

        def run_step(batch):
            optimizer.zero_grad()
            loss = criterion(model(X_train[batch]), y_train[batch], w_train[batch])
            loss.backward()
            optimizer.step()
            return loss.detach()

    history = {"train_loss": []}

    step = 0
    for epoch in range(epochs):
        model.train()
        # accumulate on the device; calling loss.item() every step would force a sync
        running_loss = torch.zeros((), device=device)
        perm = torch.randperm(n_train, device=device, generator=shuffle_generator)

        for i in range(n_batches):
            set_lr(lr_at(step))
            running_loss += run_step(perm[i * batch_size : (i + 1) * batch_size])
            step += 1

        train_loss = running_loss.item() / n_batches
        history["train_loss"].append(train_loss)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - train_loss: {train_loss:.6f} "
                f"- lr: {lr_at(step - 1):.3e}"
            )

    return model, history
