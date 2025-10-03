# 🚗 Car Price Prediction Web App

A user-friendly web application built with **Streamlit** that predicts the price of used cars using a **machine learning model**.  
This project demonstrates the complete pipeline of **data cleaning, model training, and deploying the model** as an interactive web app.

---

## ✨ Features
- **Interactive UI**: A clean and simple interface for users to input car specifications.  
- **Real-Time Predictions**: Instantly get price predictions based on the input features.  
- **User Authentication**: A secure login system for accessing the application.  
- **Data Visualization**: *(Optional, if enabled)* Interactive charts and plots to explore the car dataset.  
- **Model Integration**: Utilizes a pre-trained **Linear Regression model (.pkl file)** to make predictions.  

---

## 🛠️ Technologies Used
- **Backend & ML**: Python, Scikit-learn, Pandas, NumPy, Pickle  
- **Frontend**: Streamlit  
- **Data Visualization**: Matplotlib, Plotly  
- **IDE**: VS Code  

---

## 🚀 Setup and Installation

Follow these steps to get the application running on your local machine.  

### 1. Prerequisites  
Make sure you have **Python 3.9+** installed.  

### 2. Clone the Repository
```arduino
git clone https://github.com/YOUR_USERNAME/Car-Price-Predictor.git
cd Car-Price-Predictor
```
(Replace YOUR_USERNAME with your GitHub username)

3. Create a Virtual Environment
```arduino
# Create virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```
4. Install Dependencies
```arduino
pip install -r requirements.txt
```

5. Run the Application
```arduino
streamlit run app.py
```

## 📂 Project Structure

Car-Price-Predictor/
├── .venv/                  # Virtual environment folder (ignored by git)
├── assets/                 # Folder for images, css, etc.
├── __pycache__/            # Python cache (ignored by git)
├── app.py                  # Main Streamlit application script
├── auth.py                 # Authentication logic script
├── Car.csv                 # Primary dataset used for training
├── Car Price Predictor.ipynb # Jupyter Notebook for EDA and model training
├── LinearRegressionModel.pkl # Pre-trained machine learning model
├── prediction_history.csv  # Log of predictions (ignored by git)
├── Quikr_car.csv           # Secondary dataset
├── requirements.txt        # List of Python dependencies
├── users.json              # User credentials (ignored by git)
└── README.md               # This file

## 💡 Future Enhancements

* Enhance UI/UX with improved design and responsiveness.
* Integrate multiple ML models (Random Forest, XGBoost, etc.) for better accuracy.
* Add a visualization dashboard for deeper insights.
* Implement role-based authentication (admin/user).
* Deploy on Streamlit Cloud, Heroku, or AWS for public access.