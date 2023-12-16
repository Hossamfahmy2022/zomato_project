## Import Libraries
import streamlit as st
import joblib
import numpy as np
from utlis import process_new
import pandas as pd
df=pd.read_csv("zomato.csv")

## Load the model
model = joblib.load('xgb.pkl')




def zomato_classification():

    ## Title
    st.title('Restaurant success Classification Prediction ....')
    st.markdown('<hr>', unsafe_allow_html=True)



    ## Input fields

    Delivery_rest = st.selectbox('Delivery or rest', options=['Delivery and rest', 'Delivery Only'])
    online_order = st.selectbox('online_order', options=['Yes', 'No'])
    book_table = st.selectbox('book_table', options=['Yes', 'No'])
    location = st.selectbox('location', options=df["location"].unique().tolist())
    rest_types = st.slider('restaurante are suitable for # shape (restaurant only _ restaurant&cafe)',0,5,1)
    cuisines = st.slider('restaurante are offer # cuisines of food ',0,10,1)
    menu_items = st.slider('restaurante are offer # menu_items of food ',0,800,1)
    resturant_catogray = st.selectbox('resturant_catogray', options=df["listed_in(type)"].unique().tolist())
    city = st.selectbox('city', options=df["listed_in(city)"].unique().tolist())
    approx_cost = st.slider('approx_cost for two people in restaurante ',40,1000,20)      
   

    st.markdown('<hr>', unsafe_allow_html=True)


    if st.button('Predict success ...'):

        ## Concatenate the users data
        new_data = np.array([Delivery_rest, online_order, book_table, location, rest_types,
                            cuisines, menu_items, resturant_catogray, city,approx_cost])
        
        ## Call the function from utils.py to apply the pipeline
        X_processed = process_new(X_new=new_data)

        ## Predict using Model
        
        y_pred = model.predict(X_processed)


        y_pred = bool(y_pred)

        ## Display Results
        st.success(f'restaurante success Prediction is ... {y_pred}')



if __name__ == '__main__':
    ## Call the function
    zomato_classification()

