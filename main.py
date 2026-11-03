import os

from NNProbe import (
    MLPRegressorTorch,
    log_run,
    preprocess_data,
    save_preprocessor,
    test_dnn_performance,
    train_mlp_regressor,
    upload,
)
from NNProbe.pipeline import FEATURE_COLUMNS

FILE_PATH = "data/bryant08-precalc-ML.npy"
WEIGHTS_DIR = "src/NNProbe/weights/"
RANDOM_STATE = 42
TEST_SIZE = 0.02
EPOCHS = 2500
IN_RANGE_WEIGHT = 1.0  # loss weight of training rows inside the evaluation range
HIDDEN_LAYER_SIZES = (512, 256, 128, 64)
ALPHA = 0.0
BATCH_SIZE = 512
LEARNING_RATE_INIT = 2.5e-4
WARMUP_FRACTION = 0.02

RUN_SETTINGS = {
    "data file": FILE_PATH,
    "weights dir": WEIGHTS_DIR,
    "random state": RANDOM_STATE,
    "test size": TEST_SIZE,
    "features": FEATURE_COLUMNS,
    "epochs": EPOCHS,
    "in-range weight": IN_RANGE_WEIGHT,
    "hidden layer sizes": HIDDEN_LAYER_SIZES,
    "alpha (weight decay)": ALPHA,
    "batch size": BATCH_SIZE,
    "learning rate init": LEARNING_RATE_INIT,
    "warmup fraction": WARMUP_FRACTION,
    "architecture": MLPRegressorTorch(len(FEATURE_COLUMNS), HIDDEN_LAYER_SIZES),
}


def main():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Data file {FILE_PATH} not found.")

    data_dict = preprocess_data(FILE_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    preprocessor_path = os.path.join(WEIGHTS_DIR, "preprocessor.pkl")
    target_scaler_path = os.path.join(WEIGHTS_DIR, "target_scaler.pkl")

    save_preprocessor(
        data_dict["preprocessor"],
        data_dict["target_scaler"],
        preprocessor_path,
        target_scaler_path,
    )

    X_train = data_dict["X_train"]
    X_test = data_dict["X_test"]
    y_train_scaled = data_dict["y_train_scaled"]
    y_test_scaled = data_dict["y_test_scaled"]

    model, history = train_mlp_regressor(
        X_train,
        y_train_scaled,
        in_range=data_dict["train_in_range"],
        in_range_weight=IN_RANGE_WEIGHT,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES,
        alpha=ALPHA,
        batch_size=BATCH_SIZE,
        learning_rate_init=LEARNING_RATE_INIT,
        epochs=EPOCHS,
        warmup_fraction=WARMUP_FRACTION,
        random_state=RANDOM_STATE,
    )

    print(f"Final training loss: {history['train_loss'][-1]}")

    model.save_pretrained(WEIGHTS_DIR)
    print(f"Model weights saved to {WEIGHTS_DIR}")

    test_dnn_performance(
        model,
        data_dict["preprocessor"],
        data_dict["target_scaler"],
        X_train,
        X_test,
        y_train_scaled,
        y_test_scaled,
    )

    # upload(
    #     repo_id="FreekKlabbers/plasma-probe",
    #     weights_path=os.path.join(WEIGHTS_DIR, "model.pt"),
    #     config_path=os.path.join(WEIGHTS_DIR, "config.json"),
    #     extra_paths=[preprocessor_path, target_scaler_path],
    #     private=False,
    #     commit_message="Upload weights from training run",
    # )


if __name__ == "__main__":
    with log_run("main", RUN_SETTINGS):
        main()
