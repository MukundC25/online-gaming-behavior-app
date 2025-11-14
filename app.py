import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Online Gaming Behavior", page_icon="🎮", layout="wide")

# Load saved objects
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
enc_inp = joblib.load("enc_InGamePurchases.pkl")
enc_diff = joblib.load("enc_GameDifficulty.pkl")
enc_genre = joblib.load("enc_GameGenre.pkl")
enc_target = joblib.load("enc_EngagementLevel.pkl")
feature_order = joblib.load("feature_order.pkl")

# Dataset for UI options
df = pd.read_csv("online_gaming_behavior_dataset.csv")

# ---------------- UI ------------------
st.title("🎮 Online Gaming Behavior – ML Prediction")

menu = st.sidebar.radio("Navigation", ["Predict"])

if menu == "Predict":
    st.header("Predict Engagement Level")

    # User input
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 15, 50, 25)
        playtime = st.slider("Daily Playtime (Hours)", 0, 24, 8)
        sessions = st.slider("Sessions Per Week", 0, 24, 10)
        avg_session = st.slider("Avg Session Duration (Minutes)", 0, 300, 95)
    with col2:
        purchase = st.selectbox("In-Game Purchases", sorted(df["InGamePurchases"].unique().astype(str)))
        difficulty = st.selectbox("Difficulty", sorted(df["GameDifficulty"].unique().astype(str)))
        genre = st.selectbox("Game Genre", sorted(df["GameGenre"].unique().astype(str)))

    # ---------------- Preprocessing ------------------

    # DataFrame in training order
    input_df = pd.DataFrame([{
        "Age": age,
        "PlayTimeHours": playtime,
        "SessionsPerWeek": sessions,
        "AvgSessionDurationMinutes": avg_session,
        "InGamePurchases": purchase,
        "GameDifficulty": difficulty,
        "GameGenre": genre
    }])

    # Apply label encoding (exact same classes as training)
    input_df["InGamePurchases"] = enc_inp.transform(input_df["InGamePurchases"].astype(str))
    input_df["GameDifficulty"] = enc_diff.transform(input_df["GameDifficulty"].astype(str))
    input_df["GameGenre"] = enc_genre.transform(input_df["GameGenre"].astype(str))

    # Reorder columns exactly as training
    input_df = input_df[feature_order]

    # Scale
    input_scaled = scaler.transform(input_df)

    # Prediction
    pred_num = model.predict(input_scaled)[0]
    pred_label = enc_target.inverse_transform([pred_num])[0]

    # Output
    st.success(f"Predicted Engagement Level: {pred_label}")

    # Feature importance (optional)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=model.feature_importances_, y=feature_order, ax=ax)
    st.pyplot(fig)
