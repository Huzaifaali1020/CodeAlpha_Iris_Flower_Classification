import streamlit as st
import pandas as pd
import seaborn as sns

st.title("Exploratory Data Analysis")

data = pd.read_csv("Iris.csv")

data = data.drop("Id", axis=1)

st.subheader("Dataset Preview")
st.write(data.head(160))

st.subheader("Dataset Shape")
st.write(data.shape)

st.subheader("Statistical Summary")
st.write(data.describe())

st.subheader("Pairplot")

fig = sns.pairplot(data, hue="Species")
st.pyplot(fig)
