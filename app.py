import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
@st.cache_data
def load_data():
    col_names = ['bi_rads', 'age', 'shape', 'margin', 'density', 'severity']
    data = pd.read_csv('mammographic_masses.data.txt', header=None, names=col_names, na_values='?')
    return data

# Data Discovery
def data_discovery(data):
    st.write("### Data Overview")
    st.write(data.head())
    st.write("### Summary Statistics")
    st.write(data.describe())
    st.write("### Missing Values")
    missing_percentage = (data.isnull().sum() / len(data)) * 100
    st.write(missing_percentage)

    st.write("### Data Distribution")
    fig, ax = plt.subplots(figsize=(12, 8))
    data.hist(ax=ax)
    st.pyplot(fig)

    st.write("### Missing Values Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.isnull(), cbar=False, ax=ax)
    st.pyplot(fig)

    st.write("### Pairplot of Features")
    fig = sns.pairplot(data.dropna(), hue='severity')
    st.pyplot(fig)

# Preprocessing
def preprocess_data(data):
    data = data.dropna()
    X = data.drop(columns=['severity'])
    y = data['severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

# Train and Predict with chosen model
def train_and_predict_model(model_name, X_train, y_train, X_test, y_test, scaler):
    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression()
    }
    
    model = models[model_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    st.write(f"### Prediction Metrics with {model_name}")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    
    # Input parameters for prediction
    st.write("### Input Parameters for Prediction")
    bi_rads = st.number_input("BI-RADS", min_value=0, max_value=5, value=4)
    age = st.number_input("Age", min_value=0, max_value=100, value=50)
    shape = st.selectbox("Shape", [1, 2, 3, 4])
    margin = st.selectbox("Margin", [1, 2, 3, 4, 5])
    density = st.selectbox("Density", [1, 2, 3, 4])
    
    input_data = np.array([[bi_rads, age, shape, margin, density]])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    
    st.write(f"Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")

# Main function
def main():
    st.title("Mammogram Mass Severity Prediction")
    data = load_data()
    
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select a section", ["Data Discovery", "Model Prediction"])
    
    if options == "Data Discovery":
        data_discovery(data)
    elif options == "Model Prediction":
        model_choice = st.sidebar.selectbox("Choose a model", ["Decision Tree", "Random Forest", "KNN", "Naive Bayes", "SVM", "Logistic Regression"])
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
        train_and_predict_model(model_choice, X_train, y_train, X_test, y_test, scaler)

if __name__ == '__main__':
    main()
