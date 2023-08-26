# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import trend_finder_lr as tflr
import trend_finder_ovr as tfovr
import yfinance as yf
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder

## Support Vector Machine for multi-class using One-Vs-One
class svm_ovr:
    def __init__(self, name):
        self.name = name
        self.model = svm.SVC()
        self.accuracy = 0
        
    def train_model(self, structured_df):
        selected_columns = [col for col in structured_df.columns if col != 'Buy/Sell']
        
        X = structured_df[selected_columns]
        
        y = structured_df['Buy/Sell']
        
        # Balance the data using RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        print(y_resampled.value_counts())
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25)
        model = svm.SVC(decision_function_shape='ovo')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        print('The testing accuracy is: %0.2f' % accuracy)
        #print('Coefficients:\n', model.coef_)
        self.model = model
        self.accuracy = accuracy
    
    def save_current_model(self):
        filename = '{}_lr_{}.sav'.format(self.name, self.accuracy)
        pickle.dump(self.model, open(filename, 'wb'))
        print(f'Model saved to {filename}')
        
    def load_model(self, file):
        self.model = pickle.load(open(file, 'rb'))
        
    
    def get_current_prediction(self, stock):
        ## Get prediction at current date
        stock = yf.Ticker(f'{stock}')
        stock = pd.DataFrame(stock.history(period='120d'))
        stock = stock[['Open', 'Close', 'Volume']]
        
        ## Pull the prices and normalize them
        stock_prices = stock.Open.tolist() + stock.Close.tolist()
        price_min, price_max = min(stock_prices), max(stock_prices)
        
        for i, val in enumerate(stock_prices):
            stock_prices[i] = (val-price_min) / (price_max - price_min)
            
            
        ## Pull the volume and normalize
        stock_volume = stock.Volume.tolist()
        volume_min, volume_max = min(stock_volume), max(stock_volume)
        for i, val in enumerate(stock_volume):
            stock_volume[i] = (val-volume_min) / (volume_max - volume_min)
            
        full_list = stock_prices + stock_volume
        full_list_df = pd.DataFrame(full_list).transpose()
        
            
        print(self.model.predict(full_list_df))


## logistic regression
class logistic_regression:
    def __init__(self, name):
        self.name = name
        self.model = LogisticRegression()
        self.accuracy = 0
        
    def train_model(self, structured_df):
        selected_columns = [col for col in structured_df.columns if col != 'Buy/Sell']
        
        X = structured_df[selected_columns]
        
        y = structured_df['Buy/Sell']
        
        # Balance the data using RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25)
        
        model = LogisticRegression(solver='liblinear')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(y_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        print('The testing accuracy is: %0.2f' % accuracy)
        print('Coefficients:\n', model.coef_)
        self.model = model
        self.accuracy = accuracy
    
    def save_current_model(self):
        filename = '{}_lr_{}.sav'.format(self.name, self.accuracy)
        pickle.dump(self.model, open(filename, 'wb'))
        print(f'Model saved to {filename}')
        
    def load_model(self, file):
        self.model = pickle.load(open(file, 'rb'))
        
    
    def get_current_prediction(self, stock):
        ## Get prediction at current date
        stock = yf.Ticker(f'{stock}')
        stock = pd.DataFrame(stock.history(period='120d'))
        stock = stock[['Open', 'Close', 'Volume']]
        
        ## Pull the prices and normalize them
        stock_prices = stock.Open.tolist() + stock.Close.tolist()
        price_min, price_max = min(stock_prices), max(stock_prices)
        
        for i, val in enumerate(stock_prices):
            stock_prices[i] = (val-price_min) / (price_max - price_min)
            
            
        ## Pull the volume and normalize
        stock_volume = stock.Volume.tolist()
        volume_min, volume_max = min(stock_volume), max(stock_volume)
        for i, val in enumerate(stock_volume):
            stock_volume[i] = (val-volume_min) / (volume_max - volume_min)
            
        full_list = stock_prices + stock_volume
        full_list_df = pd.DataFrame(full_list).transpose()
        
            
        print(self.model.predict(full_list_df))

def main():
    
    '''
                        ##Train model
    tesla = tfovr.Historydf('TSLA', 'MAX', 10, 10, 120)
    tesla.find_trends()
    tesla.structure_data()
    tesla_ovr = svm_ovr('Tesla')
    acc = 0
    mod = svm.SVC()
    
    for i in range(0, 200):
        tesla_ovr.train_model(tesla.structured_df)
        print(acc)
        if tesla_ovr.accuracy > acc:
            acc = tesla_ovr.accuracy
            mod = tesla_ovr.model
    tesla_ovr.model = mod
    tesla_ovr.accuracy = acc
    tesla_ovr.save_current_model()
    
    '''
    tesla_ovr = svm_ovr('Tesla')
    tesla_ovr.load_model('Tesla_lr_0.6590909090909091.sav')
    tesla_ovr.get_current_prediction('TSLA')
    
    
        
if __name__ == "__main__":
    main()