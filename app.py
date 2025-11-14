import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Online Gaming Behavior", page_icon="🎮", layout="wide")

# Load model and scaler
model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_order = ["Age","PlayTimeHours","SessionsPerWeek",
                 "AvgSessionDurationMinutes","InGamePurchases",
                 "GameDifficulty","GameGenre"]

df = pd.read_csv("online_gaming_behavior_dataset.csv")

# Manual mappings (same as training)
map_purchase = {"No": 0, "Yes": 1}
map_difficulty = {"Easy": 0, "Medium": 1, "Hard": 2}
map_genre = {"Action": 0, "RPG": 1, "Simulation": 2, "Sports": 3, "Strategy": 4}
map_target = {0: "Low", 1: "Medium", 2: "High"}

st.title("🎮 Online Gaming Behavior — ML Prediction")

menu = st.sidebar.radio("Navigation", ["Predict"])

if menu == "Predict":
    st.header("Predict Player Engagement Level")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 15, 50, 25)
        playtime = st.slider("Daily Playtime (Hours)", 0, 24, 8)
        sessions = st.slider("Sessions Per Week", 0, 24, 10)
        avg_session = st.slider("Avg Session Duration (Minutes)", 0, 300, 95)

    with col2:
        purchase = st.selectbox("In-Game Purchases", ["No", "Yes"])
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
        genre = st.selectbox("Game Genre", ["Action","RPG","Simulation","Sports","Strategy"])

    input_raw = {
        "Age": age,
        "PlayTimeHours": playtime,
        "SessionsPerWeek": sessions,
        "AvgSessionDurationMinutes": avg_session,
        "InGamePurchases": map_purchase[purchase],
        "GameDifficulty": map_difficulty[difficulty],
        "GameGenre": map_genre[genre]
    }

    input_df = pd.DataFrame([input_raw])[feature_order]

    # scale numerical features
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    pred_label = map_target[pred]

    st.success(f"Predicted Engagement Level: {pred_label}")

    # Feature importance
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=model.feature_importances_, y=feature_order, ax=ax)
    st.pyplot(fig)
