"""
Main application
"""

import pandas as pd
import streamlit as st

from src.data import load_metadata
from src.models import load_model


def main():
    """
    Main function of the app.
    """

    st.title("Iris Flower Prediction App")

    # Load the trained model and other saved data
    model = load_model()
    feature_names, target_names = load_metadata()

    # Create input fields for features
    st.subheader("Введите характеристики цветка:")
    feature_inputs = []
    for feature in feature_names:
        value = st.slider(f"{feature} (см)", 0.0, 10.0, 5.0)
        feature_inputs.append(value)

    # Make prediction
    if st.button("Предсказать вид ириса"):
        features = [feature_inputs]
        prediction = model.predict(features)
        species = target_names[prediction[0]]
        st.success(f"Предсказанный вид ириса: {species}")

    # Display feature importance
    st.subheader("Важность признаков:")
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    st.bar_chart(feature_importance.set_index("feature"))


if __name__ == "__main__":
    main()
