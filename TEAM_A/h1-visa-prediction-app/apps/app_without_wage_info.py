#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import streamlit as st



loaded_model = pickle.load(open('LR2_without_wage_info.pkl','rb'))

employer_branch = {
    8: "TECH SOLUTIONS",
    11: "OTHERS",
    3: "CONSULTING COMPANIES",
    4: "ELECTRONIC & LOGISTICS SERVICES",
    5: "FINANCE AND MEDICAL SOLUTIONS",
    10: "UNIVERSITY",
    9: "TOP TECH",
    2: "BUSINESS SOLUTIONS",
    1: "BANKING COMPANIES",
    7: "RESEARCH LABS & NETWORK",
    0: "AUTOMOTIVE & ELECTRICAL",
    6: "PRODUCT &ENTERPRISE COMPANIES",
    }

soc_title = {
    10: "Medical Occupations",
    4: "Computer Occupations",
    11: "Others",
    6: "Financial Occupation",
    7: "Management Occupation",
    8: "Marketing Occupation",
    2: "Architecture & Engineering",
    9: "Mathematical Occupations",
    1: "Advance Sciences",
    0: "Administrative Occupation",
    5: "Education Occupations",
    3: "Business Occupation",
    }

job_title = {
    6: "IT & SOFTWARE ENGINEERS",
    12: "SENIOR TEAM",
    13: "others",
    11: "Manager & DIRECTORS",
    1: "BUSINESS TEAM",
    2: "DATABASE & SCIENTISTS",
    9: "MECHANICAL & CIVIL ENGINEER",
    3: "ARCHITECT",
    0: "EDUCATIONAL ORGANISATION",
    4: "ELECTRONICS & ELECTRONICS ENGINEERS TEAM",
    10: "MEDICAL TEAM",
    8: "MARKETING TEAM",
    5: "FINANCE TEAM",
    7: "LAW TEAM",
    }

agent = {
    1: "YES",
    0: "NO",
    }

sec = {
    1: "YES",
    0: "NO",
    }

fulltime = {
    1: "YES",
    0: "NO",
    }

willful = {
    1: "YES",
    0: "NO",
    }

new_conc = {
    0: '0',
    1: "1",
    2: ">1",
    }

change_prev = {
    0: '0',
    1: "1",
    2: ">1",
    }

amended = {
    0: '0',
    1: "1",
    2: ">1",
    }

total = {
    0: "1",
    1: ">1",
    }

#creating a function for prediction

def Visa_prediction(AGENT_REPRESENTING_EMPLOYER_N, SECONDARY_ENTITY_1_N, FULL_TIME_POSITION_N, WILLFUL_VIOLATOR_N, 
                    NEW_CONCURRENT_EMP_N, CHANGE_PREVIOUS_EMP_N, AMENDED_PETITION_N, TOTAL_WORKER_POSITIONS_N, OCCUPATION_N, JOB_TITLE_N, 
                    EMPLOYER_BRANCH_N):
    
    Visa = [[AGENT_REPRESENTING_EMPLOYER_N, SECONDARY_ENTITY_1_N, FULL_TIME_POSITION_N, WILLFUL_VIOLATOR_N, 
             NEW_CONCURRENT_EMP_N, CHANGE_PREVIOUS_EMP_N, AMENDED_PETITION_N, TOTAL_WORKER_POSITIONS_N, OCCUPATION_N, JOB_TITLE_N, 
             EMPLOYER_BRANCH_N]]
    
    result = loaded_model.predict(Visa)

    print(result)
    if (result[0] == 1):
        return 'The person can be granted H1B visa'
    else:
        return 'Sorry, but the person will not be granted H1B visa'


def app():
    
    #giving a title 
    st.title('H1-B Visa Prediction App')
    
    # getting the input data from the user
    
    #SOC_TITLE_NEW, EMPLOYER_BRANCH, JOB_TITLE_NEW, SOC_CODE_NEW, NAICS_CODE_NEW
    
    AGENT_REPRESENTING_EMPLOYER_N = st.selectbox("AGENT_REPRESENTING_EMPLOYER", options = (0, 1), format_func = lambda x: agent.get(x),)
    SECONDARY_ENTITY_1_N = st.selectbox("SECONDARY_ENTITY_1", options = (0, 1), format_func = lambda x: sec.get(x),)
    FULL_TIME_POSITION_N = st.selectbox("FULL_TIME_POSITION", options = (0, 1), format_func = lambda x: fulltime.get(x),)
    WILLFUL_VIOLATOR_N = st.selectbox("WILLFUL_VIOLATOR", options = (0, 1), format_func = lambda x: willful.get(x),)
    NEW_CONCURRENT_EMP_N = st.selectbox("NEW_CONCURRENT", options = (0, 1, 2), format_func = lambda x: new_conc.get(x),)
    CHANGE_PREVIOUS_EMP_N = st.selectbox("CHANGE_PREVIOUS", options = (0, 1, 2), format_func = lambda x: change_prev.get(x),)
    AMENDED_PETITION_N = st.selectbox("AMENDED_PETITION", options = (0, 1, 2), format_func = lambda x: amended.get(x),)
    TOTAL_WORKER_POSITIONS_N = st.selectbox("TOTAL_WORKER_POSITIONS", options = (0, 1), format_func = lambda x: total.get(x),)
    
    #OCCUPATION = st.selectbox("Enter Occupation", ['Administrative Occupation', 'Advance Sciences', 'Architecture & Engineering', 
    #'Business Occupation', 'Computer Occupations', 'Advance Sciences', 
    #'Financial Occupations', 'Management Occupations', 'Marketing Occupations', 
    #'Mathemetical Occupations', 'Medical Occupations', 'Others'])
    
    OCCUPATION_N = st.selectbox('Select Occupation:', options = (10, 4, 11, 6, 7, 8, 13, 2, 9, 1, 0, 5, 3), format_func = lambda x: soc_title.get(x),)
    
    
    #JOB_TITLE_NEW = st.selectbox("Enter Job Title", ['IT & SOFTWARE ENGINEERS', 'SENIOR TEAM', 'others', 'Manager & DIRECTORS'                          
    #'BUSINESS TEAM', 'DATABASE & SCIENTISTS', 'MECHANICAL & CIVIL ENGINEER'                  
    #'EDUCATIONAL ORGANISATION', 'ARCHITECT', 'ELECTRONICS & ELECTRONICS ENGINEERS TEAM'     
    #'MEDICAL TEAM', 'MARKETING TEAM', 'FINANCE TEAM', 'LAW TEAM'])
    
    JOB_TITLE_N = st.selectbox('Choose the Employee Job title:', options = (6, 12, 13, 11, 1, 2, 9, 3, 0, 4, 10, 8, 5, 7), format_func = lambda x: job_title.get(x),)
    
                                 
    #EMPLOYER_BRANCH = st.selectbox("Enter Employer Name", ['TECH SOLUTIONS', 'others', 'CONSULTING COMPANIES', 'TOP TECH', 'FINANCE AND MEDICAL SOLUTIONS', 
    #'ELECTRONIC & LOGISTICS SERVICES', 'RESEARCH LABS & NETWORK', 'AUTOMOTIVE & ELECTRICAL'              
    #'BANKING COMPANIES', 'PRODUCT &ENTERPRISE COMPANIES', 'UNIVERSITY', 'BUSINESS SOLUTIONS'])
    
    EMPLOYER_BRANCH_N = st.selectbox('Choose the Employee field type:', options = (8, 11, 3, 4, 5, 10, 9, 2, 1, 7, 0, 6), format_func = lambda x: employer_branch.get(x),)
    
                     
    #code for prediction
    
    reviewer = ''
    
    #creating a button for prediction
    
    if st.button('Employee Request for H1B visa'):
        reviewer = Visa_prediction(AGENT_REPRESENTING_EMPLOYER_N, SECONDARY_ENTITY_1_N, FULL_TIME_POSITION_N, WILLFUL_VIOLATOR_N, 
                                   NEW_CONCURRENT_EMP_N, CHANGE_PREVIOUS_EMP_N, AMENDED_PETITION_N, TOTAL_WORKER_POSITIONS_N, OCCUPATION_N, JOB_TITLE_N, 
                                   EMPLOYER_BRANCH_N)
    st.success(reviewer)
    
    
if __name__ == '__app__':
    app()
        


# In[ ]:




