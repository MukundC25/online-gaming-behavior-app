# 🎮 Online Gaming Behavior Analysis & Prediction  
### Machine Learning • Data Analysis • Streamlit Web App

![banner](https://img.shields.io/badge/Project-Online%20Gaming%20Behavior-blueviolet?style=for-the-badge)
![python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge)
![streamlit](https://img.shields.io/badge/Streamlit-Cloud-red?style=for-the-badge)
![ml](https://img.shields.io/badge/Machine%20Learning-LightGBM-green?style=for-the-badge)

---

## 📌 Overview  
This project analyzes **Online Gaming Behavior** using data from **40,034 players** and builds a **Machine Learning model** to predict a player's **Engagement Level (Low / Medium / High)** based on gameplay attributes.

The repository contains:  
- Complete **Colab notebook (.ipynb)**  
- **Streamlit Web App** (`app.py`)  
- **Dataset**  
- **Project PDF report**  
- **ML model files** (`.pkl`)  
- Visual analysis & EDA plots  
- Fully deployed Streamlit demo  

---

## 🚀 Live Demo  
🔗 **Streamlit App:**  
> https://your-streamlit-app-url.streamlit.app  
*(Replace with your actual deployed URL)*

---

## 📊 Dataset Description  

| Feature | Description |
|--------|-------------|
| **Age** | Player age (15–50) |
| **Gender** | Male / Female |
| **PlayTimeHours** | Daily playtime |
| **SessionsPerWeek** | Weekly gaming frequency |
| **AvgSessionDurationMinutes** | Avg minutes per session |
| **GameDifficulty** | Easy / Medium / Hard |
| **GameGenre** | Action, RPG, Simulation, Sports, Strategy |
| **InGamePurchases** | Yes / No |
| **EngagementLevel** | Low / Medium / High |

---

## 🧠 Machine Learning  

The project evaluates multiple ML models:

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | ~81% | Baseline |
| Random Forest | ~87% | Good generalization |
| XGBoost | ~88% | Strong performer |
| **LightGBM (Selected)** | **88.1%** | Best accuracy & speed |

### ✔ Final Model  
- **Algorithm:** LightGBM  
- **Preprocessing:** Manual mapping + StandardScaler  
- **Target:** Engagement Level (3-class classification)  

---

## 🖥️ Streamlit App Features

✔ Multi-page interface  
✔ Dataset preview  
✔ Visualizations:  
  - Age distribution  
  - Gender ratio  
  - Difficulty distribution  
  - Engagement vs Achievements  
✔ Live ML prediction page  
✔ Feature importance graph  
✔ Neon/dark themed UI  

---

## 🔧 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/username/online-gaming-behavior-app.git
cd online-gaming-behavior-app



