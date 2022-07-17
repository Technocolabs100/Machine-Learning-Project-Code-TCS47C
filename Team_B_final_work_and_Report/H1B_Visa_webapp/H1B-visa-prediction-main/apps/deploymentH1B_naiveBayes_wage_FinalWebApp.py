#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import streamlit as st


# In[2]:


loaded_model = pickle.load(open('trained_model_H1B_visa_wage_rate.pkl','rb'))
wage_unit = {
    100 : "A Not Selected",
    4: "Year",
    2: "Month",
    0: "Bi-weekly",
    3: "Week",
    1: "Hour",
    }


#creating a function for prediction

def Visa_prediction(WAGE_RATE_OF_PAY_FROM_1, WAGE_RATE_OF_PAY_TO_1, WAGE_UNIT_OF_PAY_1_N, PREVAILING_WAGE_1):

    Visa = [[WAGE_RATE_OF_PAY_FROM_1, WAGE_RATE_OF_PAY_TO_1, WAGE_UNIT_OF_PAY_1_N, PREVAILING_WAGE_1]]
    result = loaded_model.predict(Visa)

    print(result)
    if (result[0] == 0):
        return 'Congratulations, Your H1B Visa is approved. Soon, You will receive the official confirmation through an e-mail.'
    else:
        return 'Regret to inform you, Your H1B Visa is rejected. Better luck next time.'


#EMPLOYER_BRANCH = int(input('enter the enterprise type:'))
#JOB_TITLE_NEW = int(input('enter the job title:'))
#SOC_CODE_NEW = int(input('enter the first two digits of soc code:'))
#NAICS_CODE_NEW = int(input('enter the naics code:'))'''
#df = {'EMPLOYER_BRANCH':['TECH SOLUTIONS', 'OTHERS', 'USA BASED COMPANIES ', 'CONSULTING COMPANIES', 'TOP TECHS', 'FINANCE AND MEDICAL SOLUTIONS',
#'ELECTRONIC & LOGISTICS SERVICES', 'RESEARCH LABS & NETWORK& MOBILE SERVICES', 'UNIVERSITY', 'CIVIL & AUTOMOTIVE & ELECTRICAL', 'PRODUCT &MANUFACTURERS &ENTERPRISE COMPANIES', 'BUSINESS SOLUTIONS', 'ASIAN BASED COMPANIES', 'MARKETS & BANKING COMPANIES', 'ECOMMERCE & LAW SERVICES'],
#        'Age':[11, 8, 14, 4, 12, 6, 5, 10, 13, 3, 9, 2, 1, 7, 0]}

def app():

    #giving a title
    st.subheader('Please provide the information of pay rate.')

    # getting the input data from the user

    #SOC_TITLE_NEW, EMPLOYER_BRANCH, JOB_TITLE_NEW, SOC_CODE_NEW, NAICS_CODE_NEW

    WAGE_RATE_OF_PAY_FROM_1 = st.text_input('Enter the wage rate of pay from:')
    WAGE_RATE_OF_PAY_TO_1 = st.text_input('Enter the wage rate of pay to:')


     #st.help(df)
    #EMPLOYER_BRANCH = st.text_input('enter the enterprise type:')
    WAGE_UNIT_OF_PAY_1_N = st.selectbox('Select the Wage unit of pay:',options= (100,4, 2, 0, 3, 1),format_func=lambda x: wage_unit.get(x),)
    PREVAILING_WAGE_1 = st.text_input('Enter the first prevailing wage:')
    #NAICS_CODE_NEW = st.text_input('enter the naics code:')


    #code for prediction

    reviewer = ''

    #creating a button for prediction

    if st.button('Employee Request for H1B visa'):
        reviewer = Visa_prediction(WAGE_RATE_OF_PAY_FROM_1, WAGE_RATE_OF_PAY_TO_1, WAGE_UNIT_OF_PAY_1_N, PREVAILING_WAGE_1)

    st.success(reviewer)


if __name__ == '__app__':
    app()

# In[ ]:
