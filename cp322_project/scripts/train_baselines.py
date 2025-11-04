"""
Train lightweight baseline regressors for the demo.

Usage:
    python scripts/train_baselines.py

Outputs:
    models/house_price_model.pkl
    models/energy_efficiency_model.pkl
    models/training_report.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW_DIR = Path("data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_regression(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_house_price_model() -> Dict[str, float]:
    df = pd.read_csv(RAW_DIR / "train.csv")
    target = "SalePrice"
    feature_cols: List[str] = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "FullBath",
        "YearBuilt",
        "LotArea",
        "Neighborhood",
        "KitchenQual",
    ]
    df = df[feature_cols + [target]].dropna(subset=[target])

    X_train, X_val, y_train, y_val = train_test_split(
        df[feature_cols],
        df[target],
        test_size=0.2,
        random_state=42,
    )

    numeric_features = [
        "OverallQual",
        "GrLivArea",
        "GarageCars",
        "GarageArea",
        "TotalBsmtSF",
        "FullBath",
        "YearBuilt",
        "LotArea",
    ]
    categorical_features = ["Neighborhood", "KitchenQual"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    metrics = evaluate_regression(y_val, val_pred)
    dump(model, MODEL_DIR / "house_price_model.pkl")
    return metrics


def train_energy_model() -> Dict[str, float]:
    df = pd.read_excel(RAW_DIR / "ENB2012_data.xlsx")
    df.columns = [c.strip() for c in df.columns]
    target_cols = ["Y1", "Y2"]
    feature_cols = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]

    X_train, X_val, y_train, y_val = train_test_split(
        df[feature_cols],
        df[target_cols],
        test_size=0.2,
        random_state=42,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_pipeline, feature_cols)]
    )

    base_regressor = GradientBoostingRegressor(random_state=42)
    regressor = MultiOutputRegressor(base_regressor)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    metrics = evaluate_regression(y_val, val_pred)
    dump(model, MODEL_DIR / "energy_efficiency_model.pkl")
    return metrics


def main():
    metrics = {
        "house_prices": train_house_price_model(),
        "energy_efficiency": train_energy_model(),
    }
    with open(MODEL_DIR / "training_report.json", "w") as fp:
        json.dump(metrics, fp, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
