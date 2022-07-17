import streamlit as st
from multiapp import MultiApp
from apps import app_with_wage_info, app_without_wage_info # import your app modules here

app = MultiApp()

st.markdown("""
# H1B Visa status Prediction

This H1B Visa status Prediction app is used to predict into 2 categories:
    
1. Based on Employee's skillset Information
2. Based on Employee's Wage related Information

Here, using the [below navigator,](https://github.com/upraneelnihar/streamlit-multiapps) and its framework developed by [Technocolabs Team.](https://technocolabs.com/). Also check out his [H1B visa data from github](https://github.com/Technocolabs100/Machine-Learning-Project-Code-TCS47C/blob/main/Paper-Work%20Visa%20Approval.pdf).

""")

# Add all your application here

app.add_app("Employee skillset Information", app_without_wage_info.app)
app.add_app("Wage rate related Information", app_with_wage_info.app)

# The main app
app.run()
