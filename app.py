import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Online Gaming Behavior Analysis", layout="wide")

# Load Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("online_gaming_behavior_dataset.csv")
    return df

df = load_data()

# Load ML model
model = joblib.load("lightgbm_model.pkl")

st.title("🎮 Online Gaming Behavior Analysis & Prediction")
st.write("A Machine Learning Dashboard for Understanding Player Engagement Patterns")

# Sidebar
st.sidebar.header("Navigation")
menu = st.sidebar.radio("Go to:", ["📊 Dataset Overview", "📈 Visual Analysis", "🤖 ML Prediction"])

# --- Dataset Overview ---
if menu == "📊 Dataset Overview":
    st.header("📊 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Dataset Summary")
    st.write(df.describe())

    st.info("Dataset contains **40,034 players**, with attributes related to gameplay, demographics, and engagement.")

# --- Visual Analysis ---
elif menu == "📈 Visual Analysis":
    st.header("📈 Exploratory Data Analysis")

    # Age Distribution
    st.subheader("🎂 Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Age"], bins=[15,20,25,30,35,40,45,50], kde=False, ax=ax)
    st.pyplot(fig)

    # Gender Distribution
    st.subheader("🚻 Gender Distribution")
    fig, ax = plt.subplots()
    df["Gender"].value_counts().plot.pie(autopct="%.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

    # Engagement vs Achievements
    st.subheader("🔥 Engagement Level vs Achievements")
    pivot = df.groupby("EngagementLevel")["AchievementsUnlocked"].mean()
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", ax=ax)
    st.pyplot(fig)

    # Heatmap sample
    st.subheader("📌 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(df.corr(numeric_only=True), cmap="viridis", ax=ax)
    st.pyplot(fig)

# --- ML Prediction ---
elif menu == "🤖 ML Prediction":
    st.header("🤖 Predict Player Engagement Level")

    st.write("Enter player details and the ML model will predict their engagement level.")

    # User input
    age = st.slider("Age", 15, 50, 25)
    playtime = st.slider("Daily Playtime (Hours)", 0, 24, 8)
    sessions = st.slider("Sessions Per Week", 0, 20, 10)
    difficulty = st.selectbox("Game Difficulty", ["Easy", "Medium", "Hard"])
    genre = st.selectbox("Game Genre", ["Action", "RPG", "Simulation", "Sports", "Strategy"])
    purchase = st.selectbox("In-Game Purchases", ["No", "Yes"])

    # Encoding
    mapping = {
        "Easy": 0, "Medium": 1, "Hard": 2,
        "Action": 0, "RPG": 1, "Simulation": 2, "Sports": 3, "Strategy": 4,
        "No": 0, "Yes": 1
    }

    input_data = np.array([[age, playtime, sessions, mapping[purchase], mapping[difficulty], mapping[genre]]])

    pred = model.predict(input_data)[0]
    labels = {0: "Low", 1: "Medium", 2: "High"}

    st.success(f"📢 **Predicted Engagement Level:** {labels[pred]} 🔥")

    # Feature importance
    st.subheader("🎯 Feature Importance")
    fig, ax = plt.subplots()
    importance = model.feature_importances_
    feature_names = ["Age", "PlayTimeHours", "SessionsPerWeek", "InGamePurchases", "Difficulty", "Genre"]
    sns.barplot(x=importance, y=feature_names, ax=ax)
    st.pyplot(fig)
