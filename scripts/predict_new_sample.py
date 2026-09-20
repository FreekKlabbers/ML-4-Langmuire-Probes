import pandas as pd

from NNProbe import Predictor

Predictor = Predictor(weight_path="src/NNProbe/weights")

X = pd.DataFrame(
    {
        "eta": [1.060949, 1.560454, 1.898884, 2.314926],
        "R": [10.0, 10.0, 10.0, 10.0],
        "delta": [0.001, 0.001, 0.001, 0.001],
        "Cp": [6.324555, 6.324555, 6.324555, 6.324555],
    }
)

y_true = [[2.63968267], [2.83030303], [2.92561321], [3.02092339], [3.11623357]]

y_pred = Predictor.predict(X)

print("Predicted J values for the new samples:")
for i, pred in enumerate(y_pred):
    print(f"Sample {i + 1}: {pred:.4f}, True J: {y_true[i][0]:.4f}")
