# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:52:22 2019

@author: surya
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle
import os.path

def load_data(f):
    # loading the data
    df = pd.read_csv(f)
    
    return df

def check_df(df):
    #checking for nan values and missing values in data
    
    print(df.isnull().any(axis = 0))
    assert df[df.isnull().any(axis = 1)].index.tolist()==[]
    
def preprocessing(df):
    #creating a copy of the original data and changing the categorical columns using dummy variables for further processing
    train_data = pd.DataFrame.copy(df)
    data_col =['season', 'yr', 'mnth', 'hr', 'weekday', 'weathersit']
    def features(df, data_col):
        dummy = pd.get_dummies(df[data_col], prefix = data_col)
        df = pd.concat([df, dummy], axis = 1)
        return df
   
    for data_col in data_col:
        train_data = features(train_data, data_col)
    
    train_data.drop(['season','yr','mnth','hr','weekday','weathersit','instant','dteday'], axis = 1, inplace = True)
    
    return train_data
    
def total_count(train_data):
    # training the model for total count 
    
    drop = ['casual','registered']
    label = ['cnt']
    '''
    taking all months of 0th year and except 11th and 12th month of first year for training the model.
    
    for the test set 11th month of 1st year is predicted.
    '''
    train_cnt  = train_data.drop(columns= drop).query('~(yr_1 == 1 & mnth_11 == 1 & mnth_12 == 1)')
    test_cnt  = train_data.drop(columns = drop).query('yr_1 == 1& mnth_11 ==1')
    
    global rfm_cnt
    
    rfm_cnt = RandomForestRegressor(n_estimators= 1001, n_jobs= -1, bootstrap= True, max_depth= None,
                                max_features='auto', max_leaf_nodes = None, random_state = 101)
    
    rfm_cnt.fit(X = train_cnt.drop(columns = label), y = train_cnt[label].values.ravel())
    
    predict_cnt = rfm_cnt.predict( X = test_cnt.drop(columns = label))
    
    fig, ax = plt.subplots(1,1, figsize=(20,7), sharex=True)
    ax.plot(test_cnt[label].values,  "-r")
    ax.plot(predict_cnt, "-b")
    ax.legend(["True values", "Predicted"])
    ax.set_title(f"Test set {label[0]}")
    plt.show()
    fig.tight_layout()
    
    return rfm_cnt, print("Mean Absolute Error for Total count is: ", mean_absolute_error(test_cnt[label], predict_cnt))
    
def reg_users(train_data):
        #training the model for registered users
    drop = ['casual','cnt']
    label = ['registered']
    
    train_reg  = train_data.drop(columns= drop).query('~(yr_1 == 1 & mnth_11 == 1 & mnth_12 == 1)')
    test_reg  = train_data.drop(columns = drop).query('yr_1 == 1& mnth_11 ==1')
    
    global rfm_reg
    
    rfm_reg = RandomForestRegressor(n_estimators= 1001, n_jobs= -1, bootstrap= True, max_depth= None,
                                max_features='auto', max_leaf_nodes = None, random_state = 101)
    
    rfm_reg.fit(X = train_reg.drop(columns = label), y = train_reg[label].values.ravel())
    
    predict_reg = rfm_reg.predict( X = test_reg.drop(columns = label))
    
    fig, ax = plt.subplots(1,1, figsize=(20,7), sharex=True)
    ax.plot(test_reg[label].values,  "-r")
    ax.plot(predict_reg, "-b")
    ax.legend(["True values", "Predicted"])
    ax.set_title(f"Test set {label[0]}")
    plt.show()
    fig.tight_layout()
    
    return rfm_reg, print("Mean Absolute Error for Registered users is: ", mean_absolute_error(test_reg[label], predict_reg))
    
def casual_users(train_data):
    #training the model for casusal users
        
    drop = ['cnt','registered']
    label = ['casual']
    
    train_casual  = train_data.drop(columns= drop).query('~(yr_1 == 1 & mnth_11 == 1 & mnth_12 == 1)')
    test_casual  = train_data.drop(columns = drop).query('yr_1 == 1& mnth_11 ==1')
    
    global rfm_casual
    
    rfm_casual = RandomForestRegressor(n_estimators= 1001, n_jobs= -1, bootstrap= True, max_depth= None,
                                max_features='auto', max_leaf_nodes = None, random_state = 101)
    
    rfm_casual.fit(X = train_casual.drop(columns = label), y = train_casual[label].values.ravel())
    
    predict_casual = rfm_casual.predict( X = test_casual.drop(columns = label))
    
    fig, ax = plt.subplots(1,1, figsize=(20,7), sharex=True)
    ax.plot(test_casual[label].values,  "-r")
    ax.plot(predict_casual, "-b")
    ax.legend(["True values", "Predicted"])
    ax.set_title(f"Test set {label[0]}")
    plt.show()
    fig.tight_layout()
    return rfm_casual, print("Mean Absolute Error for Casual users is: ", mean_absolute_error(test_casual[label], predict_casual))

def Saving_model(rfm_reg, rfm_casual, rfm_cnt):
    # saving the model using pickle
    pickle.dump(rfm_reg , open('rfm_reg','wb'))
    pickle.dump(rfm_casual, open('rfm_casual','wb'))
    pickle.dump(rfm_cnt, open('rfm_cnt','wb'))
        
        
def loading_model(rfm_reg, rfm_casual, rfm_cnt):
    #loading the required model
    
    while True:
        model_name = input('Please enter the model you want to load "registered","casual" or "count": ')
    
        if model_name == 'registered' and os.path.exists('rfm_reg'):
            model = pickle.load(open('rfm_reg','rb'))
            break

        elif model_name == 'casual' and os.path.exists('rfm_casual'):
            model  = pickle.load(open('rfm_casual', 'rb'))
            break
        
        elif model_name == 'count' and os.path.exists('rfm_cnt'):
            model  = pickle.load(open('rfm_cnt','rb'))
            break
        else:
            continue
        
    return model
    
        


    
    
if __name__ == "__main__":
    df =load_data('hour.csv')
    check_df(df)
    train_data =preprocessing(df)
    total_count(train_data)
    reg_users(train_data)
    casual_users(train_data)
    Saving_model(rfm_reg, rfm_casual, rfm_cnt)
    model = loading_model(rfm_reg, rfm_casual, rfm_cnt)  
    #new data
    print("data:", model)

    
    
    
    