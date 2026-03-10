import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

st.title("Model Comparison")
data = pd.read_csv("Iris.csv")
data = data.drop("Id", axis=1)

X = data.drop("Species", axis=1)
y = data["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
models = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

results = {}
for name, model in models.items():

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    results[name] = accuracy

results_df = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])

st.subheader("Model Accuracy Comparison")

st.table(results_df)

st.subheader("Accuracy Graph")
st.bar_chart(results_df.set_index("Model"), color="#FF69B4")
st.subheader("Confusion Matrix (Random Forest)")

model = RandomForestClassifier()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

cm = confusion_matrix(y_test, prediction)

fig, ax = plt.subplots()

sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")

plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(y_test, prediction, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.write(report_df)