# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import trend_finder_lr as tflr
import trend_finder_ovr as tfovr
import yfinance as yf
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from imblearn.over_sampling import RandomOverSampler

class one_vs_rest:
    def __init__(self, name):
        self.name = name
        self.logmodel = OneVsRestClassifier(LogisticRegression())
        self.accuracy = 0
        
    def train_model(self, structured_df):
        selected_columns = [col for col in structured_df.columns if col != 'Buy/Sell']
        
        X = structured_df[selected_columns]
        
        y = structured_df['Buy/Sell']
        
        # Balance the data using RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25)
        logmodel = OneVsRestClassifier(LogisticRegression(solver='sag', max_iter=5000))
        logmodel.fit(X_train, y_train)
        y_pred = logmodel.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        print('The testing accuracy is: %0.2f' % accuracy)
        #print('Coefficients:\n', logmodel.coef_)
        self.logmodel = logmodel
        self.accuracy = accuracy
    
    def save_current_model(self):
        filename = '{}_lr_{}.sav'.format(self.name, self.accuracy)
        pickle.dump(self.logmodel, open(filename, 'wb'))
        print(f'Model saved to {filename}')
        
    def load_model(self, file):
        self.logmodel = pickle.load(open(file, 'rb'))
        
    
    def get_current_prediction(self, stock):
        ## Get prediction at current date
        stock = yf.Ticker(f'{stock}')
        stock = pd.DataFrame(stock.history(period='60d'))
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
        
            
        print(self.logmodel.predict(full_list_df))
    
class logistic_regression:
    def __init__(self, name):
        self.name = name
        self.logmodel = LogisticRegression()
        self.accuracy = 0
        
    def train_model(self, structured_df):
        selected_columns = [col for col in structured_df.columns if col != 'Buy/Sell']
        
        X = structured_df[selected_columns]
        
        y = structured_df['Buy/Sell']
        
        # Balance the data using RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X, y)
        
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25)
        
        logmodel = LogisticRegression(solver='liblinear')
        logmodel.fit(X_train, y_train)
        y_pred = logmodel.predict(X_test)
        
        accuracy = metrics.accuracy_score(y_test, y_pred)
        
        print('The testing accuracy is: %0.2f' % accuracy)
        print('Coefficients:\n', logmodel.coef_)
        self.logmodel = logmodel
        self.accuracy = accuracy
    
    def save_current_model(self):
        filename = '{}_lr_{}.sav'.format(self.name, self.accuracy)
        pickle.dump(self.logmodel, open(filename, 'wb'))
        print(f'Model saved to {filename}')
        
    def load_model(self, file):
        self.logmodel = pickle.load(open(file, 'rb'))
        
    
    def get_current_prediction(self, stock):
        ## Get prediction at current date
        stock = yf.Ticker(f'{stock}')
        stock = pd.DataFrame(stock.history(period='90d'))
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
        
            
        print(self.logmodel.predict(full_list_df))

def main():
    tesla = tflr.Historydf('TSLA', '5y', 10, 5, 60)
    tesla.find_trends()
    tesla.structure_data()
    tesla_lr = one_vs_rest('Tesla')
    for i in range(0, 200):
        tesla_lr.train_model(tesla.structured_df)
        if tesla_lr.accuracy >= .57:
            tesla_lr.save_current_model()
    #tesla_lr.save_current_model()
    #tesla_lr.get_current_prediction('TSLA')

    
    
        
if __name__ == "__main__":
    main()