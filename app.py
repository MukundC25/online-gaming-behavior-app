# app.py (corrected + gaming-themed UI)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Page config + theme
st.set_page_config(page_title="Online Gaming Behavior", page_icon="🎮", layout="wide")

# Minimal CSS for dark gaming vibe
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0f1020, #081028); color: #E6F0FF; }
    .title { color: #7DF9FF; font-weight: 700; }
    .stButton>button { background: #2b2b8a; color: white; border-radius: 10px; }
    .stSlider > div > div > div > div { color: #E6F0FF; }
    </style>
    """,
    unsafe_allow_html=True
)

# Load saved artifacts
@st.cache_data
def load_artifacts():
    model = joblib.load("lightgbm_model.pkl")
    scaler = joblib.load("scaler.pkl")
    enc_gender = joblib.load("enc_Gender.pkl")
    enc_location = joblib.load("enc_Location.pkl")
    enc_genre = joblib.load("enc_GameGenre.pkl")
    enc_difficulty = joblib.load("enc_GameDifficulty.pkl")
    enc_purchase = joblib.load("enc_InGamePurchases.pkl")
    enc_target = joblib.load("enc_EngagementLevel.pkl")   # for mapping numeric -> label
    feature_order = joblib.load("feature_order.pkl")
    return {
        "model": model, "scaler": scaler, "encoders": {
            "Gender": enc_gender,
            "Location": enc_location,
            "GameGenre": enc_genre,
            "GameDifficulty": enc_difficulty,
            "InGamePurchases": enc_purchase,
            "EngagementLevel": enc_target
        }, "feature_order": feature_order
    }

art = load_artifacts()
model = art["model"]
scaler = art["scaler"]
encoders = art["encoders"]
feature_order = art["feature_order"]

# Header
st.title("🎮 Online Gaming Behavior — Live Demo", anchor=None)
st.markdown("**Interactive demo:** enter player attributes to predict *engagement level*. Model & preprocessing loaded from training artifacts.")

# Sidebar navigation
menu = st.sidebar.radio("Navigation", ["Overview", "Visuals", "Predictor", "About"])

# Load dataset preview (optional)
@st.cache_data
def load_df():
    return pd.read_csv("online_gaming_behavior_dataset.csv")
df = load_df()

# ========== Overview ==========
if menu == "Overview":
    st.header("Dataset Overview")
    col1, col2 = st.columns([2,1])
    with col1:
        st.dataframe(df.head(200))
    with col2:
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.markdown("**Top features**")
        st.write(feature_order)

# ========== Visuals ==========
elif menu == "Visuals":
    st.header("Exploratory Visuals")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Age distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], bins=[15,20,25,30,35,40,45,50], ax=ax)
        ax.set_facecolor("#0f1622")
        st.pyplot(fig)
    with c2:
        st.subheader("Gender distribution")
        fig, ax = plt.subplots()
        df["Gender"].value_counts().plot.pie(autopct="%.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("Engagement vs Achievements")
    fig, ax = plt.subplots()
    df.groupby("EngagementLevel")["AchievementsUnlocked"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# ========== Predictor ==========
elif menu == "Predictor":
    st.header("Predict Player Engagement Level")
    st.markdown("Fill in player info — model will use the *same encoders & scaler* as training.")

    # input widgets (match ranges from your data)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Age", 15, 50, 25)
        playtime = st.slider("Daily Playtime (Hours)", 0, 24, 8)
        sessions = st.slider("Sessions Per Week", 0, 24, 10)
    with col2:
        avg_session = st.slider("Avg Session Duration (Minutes)", 0, 300, 95)
        gender = st.selectbox("Gender", sorted(df["Gender"].unique().tolist()))
        location = st.selectbox("Location", sorted(df["Location"].unique().tolist()))
    with col3:
        difficulty = st.selectbox("Game Difficulty", sorted(df["GameDifficulty"].unique().tolist()))
        genre = st.selectbox("Game Genre", sorted(df["GameGenre"].unique().tolist()))
        purchase = st.selectbox("In-Game Purchases", sorted(df["InGamePurchases"].unique().tolist()))

    # Build DataFrame in exact same order as training
    input_df = pd.DataFrame([{
        "Age": age,
        "PlayTimeHours": playtime,
        "SessionsPerWeek": sessions,
        "AvgSessionDurationMinutes": avg_session,
        "InGamePurchases": purchase,
        "GameDifficulty": difficulty,
        "GameGenre": genre
    }])

    # Encode categorical cols using saved encoders
    for col in ["InGamePurchases", "GameDifficulty", "GameGenre", "Gender", "Location"]:
        if col in input_df.columns:
            if col in encoders:
                # if encoder exists for this col
                le = encoders[col]
                # Note: LabelEncoder expects exact string values it was fit on.
                input_df[col] = le.transform(input_df[col].astype(str))
            else:
                # fallback simple mapping (shouldn't be needed)
                input_df[col] = input_df[col].astype(str)

    # Reorder to feature_order
    input_df = input_df[feature_order]

    # Scale numeric features with saved scaler
    # Note: scaler was fit on training features in same order
    input_scaled = scaler.transform(input_df)

    # Prediction
    pred_num = model.predict(input_scaled)[0]
    # map numeric prediction back to original label using saved enc_engagement
    pred_label = encoders["EngagementLevel"].inverse_transform([pred_num])[0]

    # Show results
    st.markdown("### Prediction")
    st.metric(label="Predicted Engagement", value=f"🔮 {pred_label}")

    # Live output: show model input values (for debugging/verification)
    with st.expander("Show encoded & scaled input (debug)"):
        st.write("Encoded input (raw):")
        st.write(input_df)
        st.write("Scaled input (what model sees):")
        st.write(input_scaled)

    # Feature importance (if model has it)
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature importance")
        fi = model.feature_importances_
        fig, ax = plt.subplots(figsize=(6,4))
        # human-friendly feature names (order must match training)
        names = feature_order
        sns.barplot(x=fi, y=names, ax=ax)
        ax.set_title("Feature importance (LightGBM)")
        st.pyplot(fig)

# ========== About ==========
else:
    st.header("About & Notes")
    st.markdown("""
    - Model, scaler and encoders are loaded from training artifacts.
    - Make sure you uploaded the files to the same repo:
      `lightgbm_model.pkl`, `scaler.pkl`, `enc_Gender.pkl`, `enc_Location.pkl`,
      `enc_GameGenre.pkl`, `enc_GameDifficulty.pkl`, `enc_InGamePurchases.pkl`, `enc_EngagementLevel.pkl`, `feature_order.pkl`
    - If predictions look odd: double-check the encoders were fitted on the exact dataset used for training.
    """)
