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

Fold 1/5
Using device: cuda
Epoch 1/1000 - train_loss: 0.103353 - val_loss: 0.000495
Epoch 50/1000 - train_loss: 0.000009 - val_loss: 0.000005
Early stopping at epoch 57 (no improvement in 50 epochs). Best val loss: 0.000026
Fold R² (log-space): 0.999974
Fold mean relative error: 2.09%

Fold 2/5
Using device: cuda
Epoch 1/1000 - train_loss: 0.098935 - val_loss: 0.000478
Epoch 50/1000 - train_loss: 0.000008 - val_loss: 0.000005
Early stopping at epoch 56 (no improvement in 50 epochs). Best val loss: 0.000028
Fold R² (log-space): 0.999973
Fold mean relative error: 2.13%

Fold 3/5
Using device: cuda
Epoch 1/1000 - train_loss: 0.097573 - val_loss: 0.000388
Epoch 50/1000 - train_loss: 0.000008 - val_loss: 0.000004
Early stopping at epoch 57 (no improvement in 50 epochs). Best val loss: 0.000022
Fold R² (log-space): 0.999977
Fold mean relative error: 1.96%

Fold 4/5
Using device: cuda
Epoch 1/1000 - train_loss: 0.090695 - val_loss: 0.000401
Epoch 50/1000 - train_loss: 0.000009 - val_loss: 0.000012
Early stopping at epoch 63 (no improvement in 50 epochs). Best val loss: 0.000014
Fold R² (log-space): 0.999987
Fold mean relative error: 1.51%

Fold 5/5
Using device: cuda
Epoch 1/1000 - train_loss: 0.103887 - val_loss: 0.000354
Epoch 50/1000 - train_loss: 0.000007 - val_loss: 0.000006
Early stopping at epoch 52 (no improvement in 50 epochs). Best val loss: 0.000098
Fold R² (log-space): 0.999902
Fold mean relative error: 4.05%

# 5-FOLD CROSS-VALIDATION RESULTS

R² (log-space): 0.999962 +/- 0.000031
Mean relative error: 2.35% +/- 0.88%
