import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="Online Gaming Behavior",
    page_icon="🎮",
    layout="wide"
)

# ------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f0f1e, #13132b);
            color: #E6F0FF;
        }
        h1, h2, h3 {
            color: #7DF9FF !important;
        }
        .css-1d391kg, .css-1kyxreq {
            color: white !important;
        }
        .sidebar .sidebar-content {
            background: #101020 !important;
        }
        .stSelectbox label, .stSlider label {
            color: #C7D3F3 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------- LOAD REQUIRED FILES -------------------
df = pd.read_csv("online_gaming_behavior_dataset.csv")

model = joblib.load("lightgbm_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_order = [
    "Age", "PlayTimeHours", "SessionsPerWeek",
    "AvgSessionDurationMinutes", "InGamePurchases",
    "GameDifficulty", "GameGenre"
]

# ------------------- MANUAL MAPPINGS -------------------
map_purchase = {"No": 0, "Yes": 1}
map_difficulty = {"Easy": 0, "Medium": 1, "Hard": 2}
map_genre = {"Action": 0, "RPG": 1, "Simulation": 2, "Sports": 3, "Strategy": 4}
map_target = {0: "Low", 1: "Medium", 2: "High"}

# ------------------- SIDEBAR NAVIGATION -------------------
menu = st.sidebar.radio("Navigation", [
    "📊 Dataset Overview",
    "📈 Visual Analysis",
    "🤖 ML Prediction",
    "ℹ️ About Project"
])

# ------------------- DATASET OVERVIEW -------------------
if menu == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write("First 200 rows of the dataset:")
    st.dataframe(df.head(200))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Shape")
        st.write(df.shape)

    with col2:
        st.subheader("Columns")
        st.write(df.columns.tolist())

    st.subheader("Summary Statistics")
    st.write(df.describe())

# ------------------- VISUAL ANALYSIS -------------------
elif menu == "📈 Visual Analysis":
    st.title("📈 Visual Analysis (EDA)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎂 Age Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df["Age"], bins=[15,20,25,30,35,40,45,50], ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("🚻 Gender Distribution")
        fig, ax = plt.subplots()
        df["Gender"].value_counts().plot.pie(autopct="%.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    st.subheader("🔥 Engagement vs Achievements")
    fig, ax = plt.subplots()
    df.groupby("EngagementLevel")["AchievementsUnlocked"].mean().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("🎮 Game Difficulty Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["GameDifficulty"], ax=ax)
    st.pyplot(fig)

# ------------------- ML PREDICTION -------------------
elif menu == "🤖 ML Prediction":
    st.title("🤖 ML Prediction — Player Engagement")

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

    # Build input row
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
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)[0]
    pred_label = map_target[pred]

    st.success(f"Predicted Engagement Level: {pred_label}")

    # Feature Importance
    st.subheader("🎯 Feature Importance")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=model.feature_importances_, y=feature_order, ax=ax)
    st.pyplot(fig)

# ------------------- ABOUT PAGE -------------------
elif menu == "ℹ️ About Project":
    st.title("ℹ️ About This Project")
    st.write("""
    This project analyzes **Online Gaming Behavior** using data from 40,000+ players.
    
    ### Included:
    - Exploratory Data Analysis (EDA)
    - Statistical Insights
    - ML Model (LightGBM)
    - Engagement Level Prediction
    - Full Streamlit UI

    ### Tools Used:
    - Python, Pandas, NumPy
    - LightGBM
    - Scikit-learn
    - Streamlit
    - Google Colab
    """)

