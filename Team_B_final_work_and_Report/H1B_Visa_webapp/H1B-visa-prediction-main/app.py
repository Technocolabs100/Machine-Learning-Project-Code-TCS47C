import streamlit as st
from multiapp import MultiApp
from apps import deploymentH1B_naiveBayesWithout_wage_FinalWebApp, deploymentH1B_naiveBayes_wage_FinalWebApp, Deploy_skillset_wage # import your app modules here

app = MultiApp()

st.markdown("""
# H1B Visa status Prediction

This H1B Visa status Prediction app is used to predict into three categories:
1. Based on the Employee Information
2. Related to the Wage pay Information
3. Both Employee skillset and wage pay Information

 Here, Choose above using the following select options. Also check out his [H1B visa data from github](https://github.com/Technocolabs100/Machine-Learning-Project-Code-TCS47C/blob/main/Paper-Work%20Visa%20Approval.pdf).
Current model is developed in cooperation with [Technocolabs Team.](https://technocolabs.com/)
""")

# Add all your application here
app.add_app("Employee skillset Information", deploymentH1B_naiveBayesWithout_wage_FinalWebApp.app)
app.add_app("Wage rate related Information", deploymentH1B_naiveBayes_wage_FinalWebApp.app)
app.add_app("Employee skillset and wage Information", Deploy_skillset_wage.app)
# The main app

app.run()
