#!/usr/bin/env python3
"""
Four-Model LOBOCV with 10 Seeds
Models: AE-GRU, AE-LSTM, GRU, LSTM
Evaluation: Leave-One-Out CV over 9 experiments
Seeds: 24 (anchor) + 9 random
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GRU, Input, LSTM, TimeDistributed
from tensorflow.keras.models import Model


X_FEATURES = ["HOUR", "Temperature", "VVM", "RPM", "OD600", "Glucresidual", "LA", "DO"]
Y_FEATURES = ["Next_OD600", "Next_Gluc", "Next_LA", "Next_DO"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=range(1, 12))
    if "Qgrowth" in df.columns:
        df = df.drop(columns=["Qgrowth"])

    df["Next_OD600"] = df.groupby("ExpID")["OD600"].shift(-1)
    df["Next_Gluc"] = df.groupby("ExpID")["Glucresidual"].shift(-1)
    df["Next_LA"] = df.groupby("ExpID")["LA"].shift(-1)
    df["Next_DO"] = df.groupby("ExpID")["DO"].shift(-1)
    df = df.dropna(subset=Y_FEATURES).reset_index(drop=True)
    return df


def select_val_exps(test_id: int, all_exps: list, split_seed_base: int) -> list:
    remaining = [e for e in all_exps if e != test_id]
    rng = np.random.default_rng(split_seed_base + int(test_id))
    return sorted(rng.choice(remaining, size=2, replace=False).tolist())


def make_split(
    df: pd.DataFrame,
    test_id: int,
    n_steps: int,
    split_seed_base: int,
    fixed_val_exps: list = None,
):
    unique_exps = sorted(df["ExpID"].unique())
    if fixed_val_exps is not None and len(fixed_val_exps) > 0:
        val_exps = sorted([int(v) for v in fixed_val_exps if int(v) != int(test_id)])
    else:
        val_exps = select_val_exps(test_id, unique_exps, split_seed_base)
    train_exps = [e for e in unique_exps if e != test_id and e not in val_exps]

    x_train_list, y_train_list = [], []
    x_val_list, y_val_list = [], []
    x_test_list, y_test_list = [], []

    for exp in unique_exps:
        d = df[df["ExpID"] == exp].sort_values("HOUR")
        if len(d) != n_steps:
            continue

        x_seq = d[X_FEATURES].values
        y_seq = d[Y_FEATURES].values

        if exp in train_exps:
            x_train_list.append(x_seq)
            y_train_list.append(y_seq)
        elif exp in val_exps:
            x_val_list.append(x_seq)
            y_val_list.append(y_seq)
        elif exp == test_id:
            x_test_list.append(x_seq)
            y_test_list.append(y_seq)

    x_train = np.vstack(x_train_list)
    y_train = np.vstack(y_train_list)
    x_val = np.vstack(x_val_list)
    y_val = np.vstack(y_val_list)
    x_test = np.vstack(x_test_list)
    y_test = np.vstack(y_test_list)

    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    x_train_std = scaler_x.fit_transform(x_train)
    x_val_std = scaler_x.transform(x_val)
    x_test_std = scaler_x.transform(x_test)
    y_train_std = scaler_y.fit_transform(y_train)
    y_val_std = scaler_y.transform(y_val)
    y_test_std = scaler_y.transform(y_test)

    n_features = x_train_std.shape[1]
    x_train_3d = x_train_std.reshape(-1, n_steps, n_features)
    y_train_3d = y_train_std.reshape(-1, n_steps, 4)
    x_val_3d = x_val_std.reshape(-1, n_steps, n_features)
    y_val_3d = y_val_std.reshape(-1, n_steps, 4)
    x_test_3d = x_test_std.reshape(-1, n_steps, n_features)
    y_test_3d = y_test_std.reshape(-1, n_steps, 4)

    return {
        "x_train_3d": x_train_3d,
        "y_train_3d": y_train_3d,
        "x_val_3d": x_val_3d,
        "y_val_3d": y_val_3d,
        "x_test_3d": x_test_3d,
        "y_test_3d": y_test_3d,
        "scaler_y": scaler_y,
        "train_exps": [int(e) for e in train_exps],
        "val_exps": [int(e) for e in val_exps],
    }


def build_ae(n_steps: int, n_features: int, latent_dim: int) -> Model:
    inputs = Input(shape=(n_steps, n_features))
    x = TimeDistributed(Dense(128, activation="relu"))(inputs)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    x = TimeDistributed(Dense(32, activation="relu"))(x)
    x = TimeDistributed(Dense(16, activation="relu"))(x)
    z = TimeDistributed(Dense(latent_dim, activation="relu"))(x)

    y = TimeDistributed(Dense(16, activation="relu"))(z)
    y = TimeDistributed(Dense(32, activation="relu"))(y)
    y = TimeDistributed(Dense(64, activation="relu"))(y)
    y = TimeDistributed(Dense(128, activation="relu"))(y)
    out = TimeDistributed(Dense(n_features, activation="linear"))(y)

    return Model(inputs=inputs, outputs=out)


def build_ae_gru(n_steps: int, n_features: int, latent_dim: int) -> Model:
    inputs = Input(shape=(n_steps, n_features))
    x = TimeDistributed(Dense(128, activation="relu"))(inputs)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    x = TimeDistributed(Dense(32, activation="relu"))(x)
    x = TimeDistributed(Dense(16, activation="relu"))(x)
    x = TimeDistributed(Dense(latent_dim, activation="relu"))(x)

    x = GRU(
        64,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(x)
    x = TimeDistributed(Dense(16, activation="relu"))(x)
    out = TimeDistributed(Dense(4, activation="linear"))(x)

    return Model(inputs=inputs, outputs=out)


def build_ae_lstm(n_steps: int, n_features: int, latent_dim: int) -> Model:
    inputs = Input(shape=(n_steps, n_features))
    x = TimeDistributed(Dense(128, activation="relu"))(inputs)
    x = TimeDistributed(Dense(64, activation="relu"))(x)
    x = TimeDistributed(Dense(32, activation="relu"))(x)
    x = TimeDistributed(Dense(16, activation="relu"))(x)
    x = TimeDistributed(Dense(latent_dim, activation="relu"))(x)

    x = LSTM(
        64,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(1e-4),
        recurrent_regularizer=regularizers.l2(1e-4),
    )(x)
    x = TimeDistributed(Dense(16, activation="relu"))(x)
    out = TimeDistributed(Dense(4, activation="linear"))(x)

    return Model(inputs=inputs, outputs=out)


def build_gru(n_steps: int, n_features: int) -> Model:
    inputs = Input(shape=(n_steps, n_features))
    x = GRU(
        64,
        return_sequences=True,
        kernel_regularizer=regularizers.l1(1e-4),
        recurrent_regularizer=regularizers.l1(1e-4),
    )(inputs)
    x = TimeDistributed(Dense(16, activation="relu"))(x)
    out = TimeDistributed(Dense(4, activation="linear"))(x)

    return Model(inputs=inputs, outputs=out)


def build_lstm(n_steps: int, n_features: int) -> Model:
    inputs = Input(shape=(n_steps, n_features))
    x = LSTM(
        64,
        return_sequences=True,
        kernel_regularizer=regularizers.l1(1e-4),
        recurrent_regularizer=regularizers.l1(1e-4),
    )(inputs)
    x = TimeDistributed(Dense(16, activation="relu"))(x)
    out = TimeDistributed(Dense(4, activation="linear"))(x)

    return Model(inputs=inputs, outputs=out)


def rollout_od600(model: Model, x_test_3d: np.ndarray, y_test_3d: np.ndarray, scaler_y: MinMaxScaler, n_steps: int):
    virtual_x = np.zeros((1, n_steps, len(X_FEATURES)))
    virtual_x[:, 0, :] = x_test_3d[:, 0, :]

    generated = []
    for t in range(n_steps):
        pred_all = model.predict(virtual_x, verbose=0)
        next_pred = pred_all[:, t, :]
        generated.append(next_pred[0])
        if t < n_steps - 1:
            virtual_x[:, t + 1, 0:4] = x_test_3d[:, t + 1, 0:4]
            virtual_x[:, t + 1, 4:8] = next_pred

    generated_arr = np.array(generated)
    pred_real = scaler_y.inverse_transform(generated_arr)[:, 0]

    dummy = np.zeros((n_steps, 4))
    dummy[:, 0] = y_test_3d[0, :, 0]
    true_real = scaler_y.inverse_transform(dummy)[:, 0]

    return true_real, pred_real


def generate_seeds(anchor_seed: int, n_extra: int, seed_base: int) -> list:
    """Generate reproducible seed list: [anchor_seed, 9 random seeds]"""
    rng = np.random.default_rng(seed_base)
    extra_seeds = rng.choice(10000, size=n_extra, replace=False).tolist()
    return [anchor_seed] + extra_seeds


def train_evaluate_model(
    model_type: str,
    split: dict,
    n_steps: int,
    latent_dim: int,
    ae_model: Model = None,
    ae_epochs: int = 400,
    gru_epochs: int = 400,
    batch_size: int = 3,
    patience: int = 20,
):
    """Train and evaluate a single model"""
    x_train_3d = split["x_train_3d"]
    y_train_3d = split["y_train_3d"]
    x_val_3d = split["x_val_3d"]
    y_val_3d = split["y_val_3d"]
    x_test_3d = split["x_test_3d"]
    y_test_3d = split["y_test_3d"]
    scaler_y = split["scaler_y"]

    es = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0)
    n_features = x_train_3d.shape[-1]

    # Transfer-learning models require AE pre-training
    if model_type in ["AE-GRU", "AE-LSTM"]:
        if ae_model is None:
            ae = build_ae(n_steps, n_features, latent_dim)
            ae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss="mse", metrics=["mae"])
            ae.fit(
                x_train_3d,
                x_train_3d,
                validation_data=(x_val_3d, x_val_3d),
                batch_size=batch_size,
                epochs=ae_epochs,
                callbacks=[es],
                shuffle=True,
                verbose=0,
            )
        else:
            ae = ae_model

        if model_type == "AE-GRU":
            model = build_ae_gru(n_steps, n_features, latent_dim)
        else:  # AE-LSTM
            model = build_ae_lstm(n_steps, n_features, latent_dim)

        for i in range(1, 6):
            model.layers[i].set_weights(ae.layers[i].get_weights())
            model.layers[i].trainable = False

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    else:
        # Pure models
        if model_type == "GRU":
            model = build_gru(n_steps, n_features)
        else:  # LSTM
            model = build_lstm(n_steps, n_features)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])

    model.fit(
        x_train_3d,
        y_train_3d,
        validation_data=(x_val_3d, y_val_3d),
        batch_size=batch_size,
        epochs=gru_epochs,
        callbacks=[es],
        shuffle=True,
        verbose=0,
    )

    y_true, y_pred = rollout_od600(model, x_test_3d, y_test_3d, scaler_y, n_steps)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-Model LOBOCV with 10 Seeds")
    parser.add_argument("--data", default="LAB_train2025_171.csv")
    parser.add_argument("--n-steps", type=int, default=18)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--ae-epochs", type=int, default=400)
    parser.add_argument("--gru-epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--split-seed-base", type=int, default=20260305)
    parser.add_argument("--seed-base", type=int, default=20260410)
    parser.add_argument("--output", default="results_four_models.csv")
    args = parser.parse_args()

    df = load_data(args.data)
    unique_exps = sorted(df["ExpID"].unique())

    # Generate 10 seeds: [24, 9 random]
    seeds = generate_seeds(anchor_seed=24, n_extra=9, seed_base=args.seed_base)
    print(f"Generated seeds: {seeds}")

    # Define models
    models = ["AE-GRU", "AE-LSTM", "GRU", "LSTM"]

    # Results storage
    results = []

    # LOBOCV: iterate over test experiments and seeds
    total_runs = len(unique_exps) * len(seeds) * len(models)
    run_count = 0

    for test_exp in unique_exps:
        for seed in seeds:
            tf.keras.backend.clear_session()
            set_seed(seed)

            split = make_split(df, test_exp, args.n_steps, args.split_seed_base)

            for model_type in models:
                run_count += 1
                print(
                    f"[{run_count}/{total_runs}] test_exp={test_exp}, seed={seed}, model={model_type}...",
                    end=" ",
                    flush=True,
                )

                metrics = train_evaluate_model(
                    model_type=model_type,
                    split=split,
                    n_steps=args.n_steps,
                    latent_dim=args.latent_dim,
                    ae_epochs=args.ae_epochs,
                    gru_epochs=args.gru_epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                )

                results.append(
                    {
                        "test_exp": test_exp,
                        "seed": seed,
                        "model": model_type,
                        "r2": metrics["r2"],
                        "mse": metrics["mse"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                    }
                )
                print(f"R2={metrics['r2']:.4f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")

    # Generate summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Mean R^2 by Model (across all experiments and seeds)")
    print("=" * 80)
    summary = results_df.groupby("model")["r2"].agg(["mean", "std", "min", "max"])
    print(summary)

    print("\n" + "=" * 80)
    print("DETAILED: Mean R^2 by Model and Test Experiment")
    print("=" * 80)
    pivot_by_exp = results_df.pivot_table(values="r2", index="test_exp", columns="model", aggfunc="mean")
    print(pivot_by_exp)

    return results_df


if __name__ == "__main__":
    main()
