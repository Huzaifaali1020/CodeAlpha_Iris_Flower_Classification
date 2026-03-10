import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

st.title("Feature Importance")

data = pd.read_csv("Iris.csv")

data = data.drop("Id", axis=1)

X = data.drop("Species", axis=1)
y = data["Species"]

model = RandomForestClassifier()

model.fit(X, y)

importance = model.feature_importances_

features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

st.subheader("Feature Importance Table")
st.write(importance_df)

st.subheader("Feature Importance Graph")
fig, ax = plt.subplots()
ax.bar(features, importance, color="grey")
plt.xticks(rotation=45)
st.pyplot(fig)