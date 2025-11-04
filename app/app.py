"""Streamlit demo for predicting house prices and building energy loads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    return load(MODEL_DIR / filename)


def predict_house_price():
    st.subheader("House Price Estimator")
    st.caption(
        "Inputs align with the Kaggle Ames Housing dataset. Adjust values to explore scenarios."
    )

    model = load_model("house_price_model.pkl")
    col1, col2 = st.columns(2)
    with col1:
        overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 7)
        gr_liv_area = st.number_input("Above Ground Living Area (sqft)", 400, 6000, 2000)
        garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
        garage_area = st.number_input("Garage Area (sqft)", 0, 1500, 480)
        lot_area = st.number_input("Lot Area (sqft)", 1300, 215000, 9500, step=100)
    with col2:
        total_bsmt = st.number_input("Total Basement Area (sqft)", 0, 4000, 900)
        full_bath = st.slider("Full Bathrooms", 0, 4, 2)
        year_built = st.number_input("Year Built", 1900, 2010, 2003)
        neighborhood = st.selectbox(
            "Neighborhood",
            options=[
                "CollgCr",
                "Veenker",
                "Crawfor",
                "NoRidge",
                "Mitchel",
                "Somerst",
                "NWAmes",
                "OldTown",
                "BrkSide",
                "Sawyer",
                "NridgHt",
                "NAmes",
                "SawyerW",
                "IDOTRR",
                "MeadowV",
                "Edwards",
                "Timber",
                "Gilbert",
                "StoneBr",
                "ClearCr",
                "NPkVill",
                "Blmngtn",
                "BrDale",
                "SWISU",
                "Blueste",
                "Greens",
                "GrnHill",
                "Landmrk",
            ],
            index=5,
        )
        kitchen_qual = st.selectbox(
            "Kitchen Quality",
            options=["Ex", "Gd", "TA", "Fa"],
            index=1,
        )

    features = pd.DataFrame(
        [
            {
                "OverallQual": overall_qual,
                "GrLivArea": gr_liv_area,
                "GarageCars": garage_cars,
                "GarageArea": garage_area,
                "TotalBsmtSF": total_bsmt,
                "FullBath": full_bath,
                "YearBuilt": year_built,
                "LotArea": lot_area,
                "Neighborhood": neighborhood,
                "KitchenQual": kitchen_qual,
            }
        ]
    )

    if st.button("Predict Price", key="house_predict"):
        prediction = model.predict(features)[0]
        st.success(f"Estimated Sale Price: ${prediction:,.0f}")
        st.caption(
            "Tip: capture a screenshot of this section with meaningful inputs for the final presentation."
        )


def predict_energy_load():
    st.subheader("Energy Load Estimator")
    st.caption(
        "Based on the UCI Energy Efficiency dataset (Y1 = heating load, Y2 = cooling load)."
    )
    model = load_model("energy_efficiency_model.pkl")

    col1, col2 = st.columns(2)
    with col1:
        x1 = st.slider("Relative Compactness (X1)", 0.62, 1.0, 0.85, step=0.01)
        x2 = st.number_input("Surface Area (X2)", 500.0, 1100.0, 650.0, step=5.0)
        x3 = st.number_input("Wall Area (X3)", 250.0, 430.0, 300.0, step=5.0)
        x4 = st.number_input("Roof Area (X4)", 110.0, 300.0, 150.0, step=5.0)
    with col2:
        x5 = st.slider("Overall Height (X5)", 3.5, 7.0, 5.0, step=0.5)
        x6 = st.select_slider("Orientation (X6)", options=list(range(2, 6)), value=3)
        x7 = st.slider("Glazing Area (X7)", 0.0, 0.4, 0.1, step=0.01)
        x8 = st.select_slider(
            "Glazing Area Distribution (X8)", options=list(range(1, 6)), value=3
        )

    features = pd.DataFrame(
        [
            {
                "X1": x1,
                "X2": x2,
                "X3": x3,
                "X4": x4,
                "X5": x5,
                "X6": x6,
                "X7": x7,
                "X8": x8,
            }
        ]
    )

    if st.button("Predict Loads", key="energy_predict"):
        heating, cooling = model.predict(features)[0]
        st.success(
            f"Estimated Heating Load (Y1): {heating:.2f} | Cooling Load (Y2): {cooling:.2f}"
        )
        st.caption(
            "Use the sample values above when capturing demo screenshots for the report."
        )


def main():
    st.set_page_config(page_title="CP322 Tabular Regression Demo", layout="wide")
    st.title("House Prices & Energy Efficiency Demo")
    st.write(
        "Interactive baseline models built for CP322 Topic 3. "
        "Switch tabs below to predict either home sale prices or building energy demand."
    )

    tab1, tab2 = st.tabs(["🏡 House Prices", "🏢 Energy Usage"])
    with tab1:
        predict_house_price()
    with tab2:
        predict_energy_load()

    st.markdown("---")
    st.caption(
        "Models: RandomForest (houses) & GradientBoosting (energy). "
        "See `models/training_report.json` for metrics."
    )


if __name__ == "__main__":
    main()
