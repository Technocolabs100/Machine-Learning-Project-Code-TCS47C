import numpy as np
import pickle
import streamlit as st


# In[2]:

loaded_model = pickle.load(open('trained_model_H1B_visa_wage_skillset.pkl','rb'))



employer_branch = {
    100 : "A Not Selected",
    11: "TECH SOLUTIONS",
    8: "OTHERS",
    14: "USA BASED COMPANIES",
    4: "CONSULTING COMPANIES",
    12: "TOP TECHS",
    6: "FINANCE AND MEDICAL SOLUTIONS",
    5: "ELECTRONIC & LOGISTICS SERVICES",
    10: "RESEARCH LABS & NETWORK& MOBILE SERVICES",
    13: "UNIVERSITY",
    3: "CIVIL & AUTOMOTIVE & ELECTRICAL",
    9: "PRODUCT &MANUFACTURERS &ENTERPRISE COMPANIES",
    2: "BUSINESS SOLUTIONS",
    1: "ASIAN BASED COMPANIES",
    7: "MARKETS & BANKING COMPANIES",
    0: "ECOMMERCE & LAW SERVICES",
    }

soc_title = {
    100 : "A Not Selected",
    10: "IT ENGINEERS",
    4: "DATABASE & SCIENTISTS",
    11: "MANAGER",
    6: "ELECTRONICS & LOGISTICS",
    12: "MECHANICAL & CIVIL",
    7: "Education",
    8: "FINANCE",
    13: "MEDICAL",
    2: "AUDIT & ADVERTISEMENT",
    15: "SALES & EXECUTIVES",
    16: "TECHNICIANS",
    9: "H.R & FASHION",
    1: "AGRICULTURE & CHEFS",
    0: "ADMINSTRATIVE & LAW",
    14: "P.R & URBAN ",
    5: "EDUCATION & TRAINING ",
    3: "CHEMICAL ENGINEERS ",
    }

job_title = {
    100 : "A Not Selected",
    6: "IT & SOFTWARE ENGINEERS",
    12: "others",
    11: "BUSINESS TEAM",
    1: "Manager & DIRECTORS",
    2: "DATABASE & SCIENTISTS",
    9: "MECHANICAL & CIVIL ENGINEER",
    3: "EDUCATIONAL ORGANISATION",
    0: "ARCHITECT",
    4: "ELECTRONICS & ELECTRICAL ENGINEERS TEAM",
    10: "MEDICAL TEAM",
    8: "MARKETING TEAM",
    5: "FINANCE TEAM",
    7: "LAW TEAM",
    }

wage_unit = {
    100 : "A Not Selected",
    4: "Year",
    2: "Month",
    0: "Bi-weekly",
    3: "Week",
    1: "Hour",
    }

#creating a function for prediction

def Visa_prediction(SOC_TITLE_NEW, EMPLOYER_BRANCH, JOB_TITLE_NEW, SOC_CODE_NEW, WAGE_UNIT_OF_PAY_1, WAGE_RATE_OF_PAY_FROM_1, WAGE_RATE_OF_PAY_TO_1, PREVAILING_WAGE_1):

    Visa = [[SOC_TITLE_NEW, EMPLOYER_BRANCH, JOB_TITLE_NEW, SOC_CODE_NEW, WAGE_UNIT_OF_PAY_1, WAGE_RATE_OF_PAY_FROM_1, WAGE_RATE_OF_PAY_TO_1,PREVAILING_WAGE_1]]
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
    st.subheader('Please provide the information of Employee skills and pay rate data.')

    # getting the input data from the user

    #SOC_TITLE_NEW, EMPLOYER_BRANCH, JOB_TITLE_NEW, SOC_CODE_NEW, NAICS_CODE_NEW

    SOC_TITLE_NEW = st.selectbox('Select the SOC Title:',options= (100,10, 4, 11, 6, 12, 7, 8, 13, 2, 15, 16, 9, 1, 0, 14, 5, 3),format_func=lambda x: soc_title.get(x),)
    EMPLOYER_BRANCH = st.selectbox('Choose the Employee field type:',options= (100,11, 8, 14, 4, 12, 6, 5, 10, 13, 3, 9, 2, 1, 7, 0),format_func=lambda x: employer_branch.get(x),)
    JOB_TITLE_NEW = st.selectbox('Choose the Employee Job title:',options= (100,6, 12, 11, 1, 2, 9, 3, 0, 4, 10, 8, 5, 7),format_func=lambda x: job_title.get(x),)
    SOC_CODE_NEW = st.selectbox('Select the provided first two digits of SOC code:',options= ('A Not Selected',15, 17, 13, 11, 19, 29, 25, 27, 41, 23, 21, 43, 3, 'Others'),)
    WAGE_UNIT_OF_PAY_1 = st.selectbox('Select the Wage unit of pay:',options= (100,4, 2, 0, 3, 1),format_func=lambda x: wage_unit.get(x),)
    WAGE_RATE_OF_PAY_FROM_1 = st.text_input('Enter the wage rate of pay from:')
    WAGE_RATE_OF_PAY_TO_1 = st.text_input('Enter the wage rate of pay to:')
    PREVAILING_WAGE_1 = st.text_input('Enter the Prevailing wage:')



    #code for prediction

    reviewer = ''

    #creating a button for prediction

    if st.button('Employee Request for H1B visa'):
        reviewer = Visa_prediction(SOC_TITLE_NEW, EMPLOYER_BRANCH, JOB_TITLE_NEW, SOC_CODE_NEW, WAGE_UNIT_OF_PAY_1, WAGE_RATE_OF_PAY_FROM_1, WAGE_RATE_OF_PAY_TO_1, PREVAILING_WAGE_1)

    st.success(reviewer)


if __name__ == '__app__':
    app()
