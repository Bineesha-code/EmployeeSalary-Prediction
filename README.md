# 💼 Employee Salary Prediction

## 📌 Overview
This project predicts **annual employee salary** based on multiple factors such as **education**, **experience**, **country**, **job role**, and **organization size**.  
The model is trained using **Gradient Boosting Regressor**, fine-tuned with `GridSearchCV` for optimal performance, and deployed as an interactive **Streamlit** web application.  
It also integrates a **live currency conversion API** to display salaries in INR and other currencies.

---

## 🎯 Features
- Predicts salary based on **multiple job and demographic factors**.
- **Ethical consideration** – Avoids using sensitive attributes like age to prevent discrimination.
- **Live currency conversion** using [ExchangeRate-API](https://www.exchangerate-api.com/).
- Fine-tuned **Gradient Boosting Regressor** for improved accuracy.
- User-friendly **Streamlit** web interface.
- **Downloadable CSV** of prediction results.
- Developed in **Google Colab**, deployed locally or via **pyngrok** for remote testing.

---

## 🛠 System Requirements

**Operating System**  
- Windows 10/11, macOS, or Linux

**Frontend**  
- Streamlit

**Backend**  
- Python (version 3.9 or higher)  
- Google Colab (for model training and fine-tuning)

**Python Libraries**  
- `pandas` – Data manipulation & preprocessing  
- `numpy` – Numerical computations  
- `scikit-learn` – Model training & evaluation  
- `joblib` – Model persistence (saving & loading)  
- `requests` – Fetching live currency exchange rates  
- `streamlit` – Web app framework  
- `matplotlib` – Data visualization  
- `xgboost` – Additional boosting model for experimentation  
- `pyngrok` – Public tunneling for Colab testing  
- `os`, `zipfile`, `threading`, `time`, `google.colab`

---

## 📊 Model Development

### **1. Data Preprocessing**
- Handled missing values.
- Encoded categorical features using **LabelEncoder** and **One-Hot Encoding**.
- Created interaction features like:
  - `Edu_Exp` = Education × Experience
  - `Country_Exp` = Country × Experience

### **2. Model Selection**
- Compared multiple models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
  - **Gradient Boosting Regressor** (Selected as best performer)

### **3. Fine-Tuning**
Used **GridSearchCV** to find the best hyperparameters:
python
n_estimators: [100, 200]
learning_rate: [0.05, 0.1, 0.2]
max_depth: [3, 5, 7]

4. Best Model
Tuned Gradient Boosting Regressor
R² Score: 0.59
MAE: 21,342.58
RMSE: 28,437.08

** 🚀 Try It it out on live now **
  : [Employee Salary Prediction on Streamlit](https://employeesalary-prediction-bklsn9blyf2pxkuk3e4sqn.streamlit.app/)
