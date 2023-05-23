# 1. Load reqruired libraries
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

import wandb
import params

from feature_engine.encoding import OrdinalEncoder
from sklearn.model_selection import StratifiedShuffleSplit

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings

warnings.filterwarnings("ignore")


# 2. Load dataset
def load_data(data_at):
    df = pd.read_csv(data_at)
    return df


# 3. Encoding the target variable
def log_data(X_train, X_valid, X_test, y_train, y_valid, y_test):
    y_train = y_train.reshape(-1, 1)
    y_valid = y_valid.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    train_data = pd.DataFrame(np.concatenate((X_train, y_train), axis=1))
    valid_data = pd.DataFrame(np.concatenate((X_valid, y_valid), axis=1))
    test_data = pd.DataFrame(np.concatenate((X_test, y_test), axis=1))

    train_data_at = wandb.Artifact(params.TRAIN_DATA_AT, type="train_data")
    train_data_at.add(wandb.Table(dataframe=train_data_at), "train_data")

    valid_data_at = wandb.Artifact(params.VALID_DATA_AT, type="valid_data")
    valid_data_at.add(wandb.Table(dataframe=valid_data_at), "valid_data")

    test_data_at = wandb.Artifact(params.TEST_DATA_AT, type="test_data")
    test_data_at.add(wandb.Table(dataframe=test_data_at), "test_data")

    wandb.log_artifact(train_data_at)
    wandb.log_artifact(valid_data_at)
    wandb.log_artifact(test_data_at)


def preprocess_data(df):
    target_encoder = OrdinalEncoder(
        encoding_method="arbitrary", variables="Accident_severity"
    )
    df = target_encoder.fit_transform(df)

    X = df.drop("Accident_severity", axis=1).values
    y = df["Accident_severity"].values

    # Initialize the StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    # Split for train and temp (which will be further divided into validation and test)
    for train_index, temp_index in sss.split(X, y):
        X_train, X_temp = X[train_index], X[temp_index]
        y_train, y_temp = y[train_index], y[temp_index]

    # Now split the temp data into validation and test sets
    sss_valid_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    for valid_index, test_index in sss_valid_test.split(X_temp, y_temp):
        X_valid, X_test = X_temp[valid_index], X_temp[test_index]
        y_valid, y_test = y_temp[valid_index], y_temp[test_index]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


# 5. Model Training
def log_predictions(y_true, y_pred, name):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    # Create a wandb.Table
    table = wandb.Table(dataframe=df)
    # Log the table
    wandb.log({name: table})


def log_metrics(y_true, y_pred, name):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    classification_report_ = classification_report(y_true, y_pred, output_dict=True)

    # Create a wandb.Table
    table = wandb.Table(dataframe=pd.DataFrame(classification_report_).transpose())
    # Log the table
    wandb.log({name: table})
    wandb.log({name + "_rmse": rmse})


# Define a config dictionary object
config = {
    "eta": 0.3,
    "min_child_weight": 1,
    "max_depth": 3,
    "gamma": 0.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "scale_pos_weight": 1,
    "objective": "multi:softmax",
    "eval_metric": "rmse",
    "seed": 2022,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument(
        "--eta", type=float, default=config["eta"], help="learning rate"
    )
    parser.add_argument(
        "--min_child_weight",
        type=int,
        default=config["min_child_weight"],
        help="min_child_weight",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=config["max_depth"],
        help="max depth of the tree",
    )
    parser.add_argument("--gamma", type=float, default=config["gamma"], help="gamma")
    parser.add_argument(
        "--subsample", type=float, default=config["subsample"], help="subsample"
    )
    parser.add_argument(
        "--colsample_bytree",
        type=float,
        default=config["colsample_bytree"],
        help="colsample_bytree",
    )
    parser.add_argument("--lambda", type=float, default=1.0, help="lambda")
    parser.add_argument("--alpha", type=float, default=config["alpha"], help="alpha")
    parser.add_argument(
        "--scale_pos_weight",
        type=int,
        default=config["scale_pos_weight"],
        help="scale_pos_weight",
    )
    parser.add_argument(
        "--objective", type=str, default=config["objective"], help="objective"
    )
    parser.add_argument(
        "--eval_metric", type=str, default=config["eval_metric"], help="eval_metric"
    )
    parser.add_argument("--seed", type=int, default=config["seed"], help="seed")
    args = parser.parse_args()
    config.update(args.__dict__)
    return


def train(config):
    # WANDB RUN
    run = wandb.init(
        project=params.WANDB_PROJECT,
        entity=params.ENTITY,
        job_type="training-xgboost",
        config=config,
    )
    config = wandb.config

    # Load the data
    df = load_data("data/RTA Dataset Transformed.csv")

    # Preprocess the data
    X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess_data(df)

    # Train the model
    xgboost = xgb.XGBClassifier(
        eta=wandb.config["eta"],
        min_child_weight=wandb.config["min_child_weight"],
        max_depth=wandb.config["max_depth"],
        gamma=wandb.config["gamma"],
        subsample=wandb.config["subsample"],
        colsample_bytree=wandb.config["colsample_bytree"],
        reg_lambda=wandb.config["lambda"],
        alpha=wandb.config["alpha"],
        scale_pos_weight=wandb.config["scale_pos_weight"],
        objective=wandb.config["objective"],
        eval_metric=wandb.config["eval_metric"],
        seed=wandb.config["seed"]
    )

    xgboost = xgboost.fit(X_train, y_train)

    # Validation predictions
    y_pred = xgboost.predict(X_valid)
    # log the predictions
    log_predictions(y_valid, y_pred, name="valid")

    # Test predictions
    y_pred = xgboost.predict(X_test)
    # log the predictions
    log_predictions(y_test, y_pred, name="test")

    # Log the metrics
    log_metrics(y_valid, y_pred, name="valid")
    log_metrics(y_test, y_pred, name="test")

    wandb.finish()


if __name__ == "__main__":
    parse_args()
    train(config)
