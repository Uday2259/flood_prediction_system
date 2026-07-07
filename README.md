# 🌊 Flood Prediction System

## 📌 Project Overview

The **Flood Prediction System** is a Machine Learning-based web application that predicts the possibility of floods using rainfall and weather-related data. It helps users estimate flood risk by entering environmental parameters through a simple and user-friendly web interface. The application is built using **Python**, **Flask**, and the **XGBoost** Machine Learning algorithm.

---

# 🚀 Features

- Predicts flood risk using Machine Learning
- User-friendly Flask web application
- Accepts rainfall and weather-related inputs
- Displays instant prediction results
- Fast and lightweight prediction model
- Modern responsive user interface
- Easy to deploy and run locally

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Flask | Web Framework |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Scikit-learn | Machine Learning |
| XGBoost | Prediction Model |
| Joblib | Model Serialization |
| HTML5 | Web Structure |
| CSS3 | Styling |
| JavaScript | Client-side Functionality |
| Git | Version Control |
| GitHub | Project Hosting |
| VS Code | Development Environment |

---

# 📊 Input Features

The prediction model uses the following parameters:

- Temperature
- Humidity
- Cloud Cover
- Annual Rainfall
- Jan-Feb Rainfall
- Mar-May Rainfall
- Jun-Sep Rainfall
- Oct-Dec Rainfall
- Average June Rainfall
- Subdivision Rainfall

---

# 📂 Project Structure

```text
Flood_Prediction_System/
│
├── app.py
├── train_model.py
├── floods.save
├── transform.save
├── flood dataset.xlsx
├── requirements.txt
├── README.md
│
├── static/
│   ├── css/
│   │   └── main.css
│   ├── js/
│   └── images/
│
├── templates/
│   ├── home.html
│   ├── index.html
│   ├── chance.html
│   └── no_chance.html
│
├── screenshots/
├── model/
├── dataset/
└── notebooks/
```

---

# ⚙️ Installation

## Step 1: Clone the Repository

```bash
git clone https://github.com/uday2259/Flood_Prediction_System.git
```

## Step 2: Navigate to the Project Folder

```bash
cd Flood_Prediction_System
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Run the Flask Application

```bash
python app.py
```

## Step 5: Open Your Browser

```
http://127.0.0.1:5000
```

---

# ▶️ How to Use

1. Run the Flask application.
2. Open **http://127.0.0.1:5000** in your browser.
3. Click **Predict Flood**.
4. Enter the required weather and rainfall details.
5. Click **Predict Flood**.
6. View the prediction result.
7. The application displays either:
   - 🚨 Flood Risk Detected
   - ✅ No Flood Risk Detected

---

# 🏗️ System Architecture

```
                User
                  │
                  ▼
        Flask Web Application
                  │
                  ▼
         Data Preprocessing
         (StandardScaler)
                  │
                  ▼
       Trained XGBoost Model
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Flood Risk Detected   No Flood Risk
   (chance.html)      (no_chance.html)
```

---

# 🔄 System Workflow

```
User Input
     │
     ▼
Input Validation
     │
     ▼
Feature Scaling
     │
     ▼
Machine Learning Model
     │
     ▼
Prediction
     │
 ┌───┴──────────┐
 ▼              ▼
Flood Risk   Safe Result
```

---

# 📈 Model Performance

| Model | Accuracy |
|--------|----------|
| Decision Tree | 100% |
| Random Forest | 100% |
| K-Nearest Neighbors | 91.30% |
| XGBoost | 100% |

**Final Selected Model:** XGBoost

---

# 📈 Future Scope

- Integrate real-time weather APIs
- Improve prediction accuracy using larger datasets
- Send SMS and Email flood alerts
- Display flood-prone regions using GIS maps
- Develop Android and iOS mobile applications
- Deploy the application on cloud platforms

---

# 👨‍💻 Developer

**Name:** Uday Kumar

**Course:** B.Tech – Computer Science and Engineering

**Project:** Flood Prediction System

**Technologies:** Python, Flask, Machine Learning, XGBoost

---

# 📄 License

This project was developed for educational purposes as part of a Machine Learning academic project.
