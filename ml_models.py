# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import trend_finder as tf
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
from imblearn.over_sampling import RandomOverSampler

class svc:
    def __init__(self, name):
        self.name = name
    
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
        
    
    def get_current_prediction(self):
        pass
        '''
        ## Get prediction at current date
        stock = yf.Ticker('TSLA')
        stock = pd.DataFrame(stock.history(period='20d'))
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
        
            
        print(logmodel.predict(full_list_df))
        '''

def main():
    tesla = tf.Historydf('TSLA', '5y', 30, 5, 60)
    tesla.find_trends()
    tesla.structure_data()
    tesla_lr = logistic_regression('Tesla')
    tesla_lr.train_model(tesla.structured_df)
    tesla_lr.save_current_model()

    
    
        
if __name__ == "__main__":
    main()