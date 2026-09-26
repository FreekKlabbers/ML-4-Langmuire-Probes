import os

from NNProbe import (
    preprocess_data,
    save_preprocessor,
    test_dnn_performance,
    train_mlp_regressor,
    upload,
)

if __name__ == "__main__":
    FILE_PATH = "data/bryant08-precalc-ML.npy"
    RANDOM_STATE = 42

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Data file {FILE_PATH} not found.")

    data_dict = preprocess_data(FILE_PATH, test_size=0.02, random_state=RANDOM_STATE)
    preprocessor_path = "src/NNProbe/weights/preprocessor.pkl"
    target_scaler_path = "src/NNProbe/weights/target_scaler.pkl"

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
        hidden_layer_sizes=(512, 256, 128, 64),
        alpha=2e-6,
        batch_size=256,
        learning_rate_init=0.00005,
        max_iter=1000,
        early_stopping=True,
        n_iter_no_change=50,
        random_state=RANDOM_STATE,
    )

    print(
        f"Final training loss: {history['train_loss'][-1]}, Final validation loss: {history['val_loss'][-1]}"
    )

    model.save_pretrained("src/NNProbe/weights/")
    print("Model weights saved to src/NNProbe/weights/")

    test_dnn_performance(
        model,
        data_dict["preprocessor"],
        data_dict["target_scaler"],
        X_train,
        X_test,
        y_train_scaled,
        y_test_scaled,
    )

    upload(
        repo_id="FreekKlabbers/plasma-probe",
        weights_path="src/NNProbe/weights/model.pt",
        config_path="src/NNProbe/weights/config.json",
        extra_paths=[preprocessor_path, target_scaler_path],
        private=False,
        commit_message="Upload weights from training run",
    )
