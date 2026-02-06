# ✈️ Aero Guard: Predictive Maintenance Dashboard

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-green)
![Accuracy](https://img.shields.io/badge/Accuracy-85%25-brightgreen)

## 📌 Project Overview
**Aero Guard** is a Data Science application designed to predict the **Remaining Useful Life (RUL)** of turbofan jet engines. 

Utilizing the **NASA CMAPSS** dataset, this project implements an **XGBoost Regressor** trained on over **20,000+ sensor readings**. The model leverages advanced feature engineering (Rolling Means, Standard Deviations, and Lag Features) to achieve **~85% Accuracy**, enabling proactive maintenance scheduling and failure prevention.

## 🚀 Live Demo
*(Optional: If you deploy this to Streamlit Cloud, put the link here. Otherwise, delete this line)*

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** XGBoost, Scikit-Learn
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly, Streamlit
* **Dataset:** NASA CMAPSS (Turbofan Engine Degradation Simulation)

## 📊 Key Features
* **Real-time RUL Prediction:** Interactive dashboard for simulating sensor inputs.
* **Advanced Feature Engineering:** Incorporates rolling window statistics and lag features to capture temporal dependencies.
* **Visual Analytics:** Radar charts for sensor health and Gauge charts for RUL status.
* **High Accuracy:** Optimized using GridSearchCV to achieve ~85% R² score.

## 📂 Project Structure
```text
Aero-Guard/
├── data/                  # NASA Dataset (Train/Test/RUL)
├── app.py                 # Streamlit Dashboard Interface
├── train_model.py         # XGBoost Training Pipeline
├── requirements.txt       # Dependencies
└── README.md              # Project Documentation
```

## ⚙️ How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/durgesh-kanzariya/Aero-Guard.git](https://github.com/durgesh-kanzariya/Aero-Guard.git)
    cd Aero-Guard
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Train the Model**
    ```bash
    python train_model.py
    ```

4.  **Launch the Dashboard**
    ```bash
    streamlit run app.py
    ```

## 📈 Model Performance
* **Algorithm:** XGBoost Regressor
* **R² Score:** 84.73%
* **RMSE:** 16.41 Cycles

## 👨‍💻 Author
**Durgesh Jitendrabhai Kanzariya**
* [LinkedIn](https://linkedin.com/in/durgesh-kanzariya)
* [GitHub](https://github.com/durgesh-kanzariya)