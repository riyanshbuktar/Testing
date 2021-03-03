import streamlit as st
import pickle
import numpy as np

# Loading the saved Model
model=pickle.load(open('model.pkl','rb'))


def blood_default(features):

    features = np.array(features).astype(np.float64).reshape(1,-1)
    
    predict = model.predict(features)
    probability = model.predict_proba(features)

    return predict, probability


def main():
    st.title("Blood Donation Default Prediction")
    html_temp = """
    <div style="background-color:#dd88b3 ;padding:10px">
    <h2 style="color:white;text-align:center;">Blood Donation Default Prediction App </h2>
    </div>
    """    
    st.markdown(html_temp, unsafe_allow_html=True)

    Recency = st.text_input("Recency (months)")
    Frequency = st.text_input("Frequency(times)")
    Monetory = st.text_input("Monetory(cc blood)")
    


    if st.button("Predict"):
        
        features = [Recency,Frequency,Monetory]
        predict, proba = blood_default(features)
        if predict[0] == 1:
            

            st.success('The person will default with sureity of {} %'.format(round(np.max(proba)*100),2))

        else:
            st.success('The person will not default with sureity of {} %'.format(round(np.max(proba)*100),2))




if __name__ == '__main__':
    main()