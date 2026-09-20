# ML-4-Langmuire-Probes

Setup:

```Shell
pip install -e .
```

Which will install the package in editable mode.

Current peformance of the NN:

R² Score (log-space):      0.999984
Mean Relative Error:       1.63%
Max Relative Error:        26.73%

## 5-fold cross validation

R² (log-space): 0.999950 +/- 0.000042
Mean relative error: 2.59% +/- 1.28%
Max relative error: 37.05% +/- 15.38%

### Fold 1/5

Using device: cuda
Epoch 1/1000 - train_loss: 0.092831 - val_loss: 0.000432
Epoch 50/1000 - train_loss: 0.000008 - val_loss: 0.000006
Early stopping at epoch 97 (no improvement in 50 epochs). Best val loss: 0.000006
Fold R² (log-space): 0.999995
Fold mean relative error: 0.91%
Fold max relative error: 11.55%

### Fold 2/5

Using device: cuda
Epoch 1/1000 - train_loss: 0.101331 - val_loss: 0.000395
Epoch 50/1000 - train_loss: 0.000008 - val_loss: 0.000005
Early stopping at epoch 52 (no improvement in 50 epochs). Best val loss: 0.000096
Fold R² (log-space): 0.999902
Fold mean relative error: 4.00%
Fold max relative error: 48.11%

### Fold 3/5

Using device: cuda
Epoch 1/1000 - train_loss: 0.103529 - val_loss: 0.000496
Epoch 50/1000 - train_loss: 0.000008 - val_loss: 0.000005
Early stopping at epoch 52 (no improvement in 50 epochs). Best val loss: 0.000104
Fold R² (log-space): 0.999898
Fold mean relative error: 4.14%
Fold max relative error: 37.54%

### Fold 4/5

Using device: cuda
Epoch 1/1000 - train_loss: 0.096117 - val_loss: 0.000469
Epoch 50/1000 - train_loss: 0.000007 - val_loss: 0.000019
Early stopping at epoch 63 (no improvement in 50 epochs). Best val loss: 0.000015
Fold R² (log-space): 0.999984
Fold mean relative error: 1.62%
Fold max relative error: 56.52%

### Fold 5/5

Using device: cuda
Epoch 1/1000 - train_loss: 0.094381 - val_loss: 0.000490
Epoch 50/1000 - train_loss: 0.000008 - val_loss: 0.000006
Early stopping at epoch 58 (no improvement in 50 epochs). Best val loss: 0.000031
Fold R² (log-space): 0.999970
Fold mean relative error: 2.28%
Fold max relative error: 31.52%
