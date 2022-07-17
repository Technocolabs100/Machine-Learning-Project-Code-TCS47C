

import numpy as np
import pickle
import streamlit as st



loaded_model = pickle.load(open('LR3_with_wage_info.pkl','rb'))


#creating a function for prediction

def Visa_prediction(WAGE_RATE_OF_PAY_FROM_1, PREVAILING_WAGE_1, WAGE_RATE_OF_PAY_TO_1, WAGE_UNIT_OF_PAY_1_N):
    
    Visa = [[WAGE_RATE_OF_PAY_FROM_1, PREVAILING_WAGE_1, WAGE_RATE_OF_PAY_TO_1, WAGE_UNIT_OF_PAY_1_N]]
    
    result = loaded_model.predict(Visa)

    print(result)
    if (result[0] == 1):
        return 'The person can be granted H1B visa'
    else:
        return 'Sorry, but the person will not be granted H1B visa'


def app():
    
    #giving a title 
    st.title('H1-B Visa Prediction App on the basis of Employers Wage Info')
    
    # getting the input data from the user
    
    
    WAGE_UNIT_OF_PAY_1_N = st.selectbox("Paycheck Frequency", [0, 1, 2, 3, 4])
    WAGE_RATE_OF_PAY_FROM_1 = st.number_input('Insert Wage Rate of Pay From')
    PREVAILING_WAGE_1 = st.number_input('Insert Prevailing Wage')
    WAGE_RATE_OF_PAY_TO_1 = st.number_input('Insert Wage Rate of Pay To')
    #code for prediction
    
    reviewer = ''
    
    #creating a button for prediction
    
    if st.button('Employee Request for H1B visa'):
        reviewer = Visa_prediction(WAGE_RATE_OF_PAY_FROM_1, PREVAILING_WAGE_1, WAGE_RATE_OF_PAY_TO_1, WAGE_UNIT_OF_PAY_1_N)
    st.success(reviewer)
    
    
if __name__ == '__app__':
    app()
        


# In[ ]:




