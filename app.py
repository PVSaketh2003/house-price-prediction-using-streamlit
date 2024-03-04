# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 11:43:14 2024

@author: hp
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pickle

data=pd.read_csv('house_price.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Location']=le.fit_transform(data['Location'])
print(data['Location'])
X=data.drop('Price',axis='columns')
y=data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc=RandomForestClassifier()
a=rfc.fit(X_train,y_train)
st.title("house price predictor")
print("testing score :",rfc.score(X_test,y_test))
print("training score :",rfc.score(X_train,y_train))

nav=st.sidebar.radio("Navigation",['Analyser','Predictor','Contribute'])

if nav=="Analyser":
    st.image("img.jpeg")
    if st.checkbox("show table"):
        st.table(data)
    graph=st.selectbox("Analysis type :",['non-interactive','interactive'])
    
    if graph=='non-interactive':
        plt.figure(figsize=(12,6))
        sns.set()
        st.header("population wrt location")
        a=sns.countplot(x='Location',data=data)
        for ax in a.containers:
            a.bar_label(ax) #from this we can tell that majority of people are choosing Bommanahalli
        st.pyplot()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write(" we can tell that majority of people are choosing Bommanahalli")
        #to check location wrt price
        st.header("location vs price")
        sns.set()
        b=sns.barplot(x='Location',y='Price',data=data)
        for bx in b.containers:
            b.bar_label(bx)
        st.pyplot()
        st.write("from this we can tell that , price of Whitefield is costlier than Boomanhalli")
        #to check bhk wrt price
        st.header("BHK vs price")
        sns.set()
        b=sns.barplot(x='BHK',y='Price',data=data)
        for bx in b.containers:
            b.bar_label(bx)
        st.pyplot()
        st.write("from this we can tell that 3bhk is costlier compared to 2 bhk")
        #to check bhk wrt price and location
        st.header("BHK vs Price wrt Location")
        sns.set()
        plt.figure(figsize=(10,7))
        b=sns.barplot(x='BHK',y='Price',data=data,hue='Location')
        for bx in b.containers:
            b.bar_label(bx)
        st.pyplot()
        st.write("from this we can tell that in whitefield both 2bhk and 3bhk are costlier than Boomanhalli")
    if graph=='interactive':
        st.header("Population vs Buying Range")
        sns.distplot(data['Price'],color='blue')
        st.pyplot()
        st.write("from this we can tell that majority of people are purchasing in the range of ~20000")
        #floor vs price
        st.header("floor vs price")
        sns.set()
        plt.figure(figsize=(17,5))
        b=sns.barplot(x='Floor',y='Price',data=data)
        for bx in b.containers:
            b.bar_label(bx)
        st.pyplot()
        st.write("""from this we can tell that floor 8,floor 12,floor 13 is costlier compared to other floor from this we can tell that
                 as the floor increases the price also increases""")
        #floor vs price wrt location
        st.header("floor vs price wrt location")
        plt.figure(figsize=(17,5))
        b=sns.barplot(x='Floor',y='Price',data=data,hue='Location')
        for bx in b.containers:
            b.bar_label(bx)
        st.pyplot()
        st.write("from this we can tell that as the floor increases the price also increases more in case of Whitefield")
        #old vs price
        st.header("old vs price")
        sns.set()
        plt.figure(figsize=(10,7))
        b=sns.barplot(x='Old(years)',y='Price',data=data)
        for bx in b.containers:
            b.bar_label(bx)
        st.pyplot()
        st.write("from this we can tell that the house which is old it loses its value and new house will have more value")

if nav=='Predictor':
    st.header("House price prediction:")
    Location=st.text_input("enter the location :")
    st.write(Location)
    st.write("0 --> Bommenahalli and 1-->Whitefield")
    BHK=st.text_input("BHK :")
    st.write(BHK)
    st.write("2 -->2bhk , 2-->3bhk")
    Furnished=st.text_input("Furnished or not")
    st.write(Furnished)
    st.write("0-->not furnished ,1--> furnished")
    SquareFeet=st.text_input("Square feet")
    st.write(SquareFeet)
    Old=st.text_input("Old(years)")
    st.write(Old)
    st.write("1 for 1 year old , 5 for 5 years old ,10 for 10 years old")
    Floor=st.text_input(" no of floors")
    st.write(Floor)
    st.write("predicted house price will be :")
    if st.checkbox("actucal values"):
        st.table(y_test)
    btn=st.button("predict")
    if btn:
        rfc=RandomForestClassifier()
        a=rfc.fit(X_train,y_train)
        pickle.dump(a,open("rfc.pkl","wb"))
        activities=['RandomForestClassifier']
        option=st.sidebar.selectbox('Which model would you like to use?',activities)
        st.subheader(option)
        inputs=[["Location","BHK","Furnishing","Sq.ft","Old(years)","Floor"]]
        rfc.predict(inputs)
        st.write(rfc.predict(inputs))
                

        
        


    
        
    
        

    