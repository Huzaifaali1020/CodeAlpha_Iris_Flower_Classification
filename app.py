import streamlit as st
st.sidebar.markdown("##Developer")
st.sidebar.write("Huzaifa Ali")
st.set_page_config(
    page_title="Iris ML Dashboard",
    layout="wide"
)
st.image("iris.jpg", width=300)
st.title("Iris Flower Machine Learning App")
st.markdown("""
This project demonstrates Machine Learning models that classify Iris flowers.
### Models Used
- KNN
- SVM
- Decision Tree
- Random Forest
******************************************************************************
Use the sidebar to explore:

• Exploratory Data Analysis  
• Feature Importance  
• Model Comparison  
• Prediction Tool
""")