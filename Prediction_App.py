import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Species Prediction")

data = pd.read_csv("Iris.csv")
data = data.drop("Id", axis=1)

X = data.drop("Species", axis=1)
y = data["Species"]

model_option = st.sidebar.selectbox(
    "Select Model",
    ("KNN", "SVM", "Decision Tree", "Random Forest")
)
if model_option == "KNN":
    model = KNeighborsClassifier()

elif model_option == "SVM":
    model = SVC()

elif model_option == "Decision Tree":
    model = DecisionTreeClassifier()

else:
    model = RandomForestClassifier()

model.fit(X, y)

sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5)

if st.button("Predict"):

    prediction = model.predict(
        [[sepal_length, sepal_width, petal_length, petal_width]]
    )

    st.success(f"Predicted Species: {prediction[0]}")