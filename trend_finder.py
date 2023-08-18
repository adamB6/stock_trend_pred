# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 14:48:28 2023

@author: asbot
"""

import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from imblearn.under_sampling import RandomUnderSampler

"""
A df of desired stock's history and it's attributes for modification
    Attributes:
        name                   The name of the stock.
        trend_length            The length of desired trend.
        percent_change          The percentage of desired change throughout the trend. Negative values Correspond to decreases in value.
        num_of_pretrend_days      Number of trading days that occured before the trend. These days are used for the logistic regression model parameters.
        df                   The full dataframe of the stock's history. Only contains open and close price, and volume for each day.
        pretrends_df         The df after searching for trends and saving the pre-trend days.
        structured_df        The dataframe for holding normalized and restructured data.
        balanced_df          The dataframe for holding the data after its been balanced. Can be upsampled or downsampled.
"""
    
class Historydf:

    def __init__(self, name, trend_length, percent_change, num_of_pretrend_days):
        self.name = name
        self.df = self.get_history_df(self.name)
        self.trend_length = trend_length
        self.percent_change = percent_change
        self.num_of_pretrend_days = num_of_pretrend_days
        self.pretrends_df = pd.DataFrame()
        self.structured_df = pd.DataFrame()
        self.balanced_df = pd.DataFrame()
        
        
    # Get df of entire given stock history. df contains dfs of [open, close, volume]
    def get_history_df(self, test):
        stock = yf.Ticker(test)
        stock = pd.DataFrame(stock.history(period='3y'))
        stock.reset_index(drop=True)
        stock = stock[['Open', 'Close', 'Volume']]
        new_stock = stock.reset_index(drop=True)
        return new_stock
    
    # Process df based on desired trend lengths and desired percentage of change
    def find_trends(self):
        new_df = pd.DataFrame(columns=['Open', 'Close', 'Volume', 'Buy/Sell'])
        start_index = self.num_of_pretrend_days - 1
        end_index = len(self.df) - (self.trend_length)
        print(f"start index = {start_index}, end index = {end_index}")
        
        for i in range(start_index, end_index):
            trend_percentage_change = ((self.df.iloc[i + self.trend_length, 1] - self.df.iloc[i, 1]) / self.df.iloc[i, 1]) * 100
            if trend_percentage_change >= self.percent_change:
                for j in range(i - (self.num_of_pretrend_days), i):
                    # If percentage change is positive, label every set of pretrend days with "1", denoting a buy signal
                    # This will occur at every {num_of_pretrend_days}
                    if j == i - 1:
                        new_row = pd.DataFrame(self.df.iloc[j]).transpose()
                        new_row['Buy/Sell'] = 1
                        new_df = pd.concat([new_df, new_row], axis=0, ignore_index=True)
                    else:
                       new_row = pd.DataFrame(self.df.iloc[j]).transpose()
                       new_df = pd.concat([new_df, new_row], axis=0, ignore_index=True)
            elif trend_percentage_change <= -self.percent_change:
                for j in range(i - (self.num_of_pretrend_days), i):
                    # If percentage change is negative, label every set of pretrend days with "0", denoting a sell signal
                    # This will occur at every {num_of_pretrend_days}
                    if j == i - 1:
                        new_row = pd.DataFrame(self.df.iloc[j]).transpose()
                        new_row['Buy/Sell'] = 0
                        new_df = pd.concat([new_df, new_row], axis=0, ignore_index=True)
                    else:
                        new_row = pd.DataFrame(self.df.iloc[j]).transpose()
                        new_df = pd.concat([new_df, new_row], axis=0, ignore_index=True)

        self.pretrends_df = new_df
        
    # Normalize the df for every desired number of pretrend days (num_of_pretrend_days)
    # This helps to deal with the increase of stock price over time 
    # and difference in stock prices and volumes between companies
    def structure_data(self):
        open_prices = self.pretrends_df[['Open']].transpose()
        close_prices = self.pretrends_df[['Close']].transpose()
        volume = self.pretrends_df[['Volume']].transpose()
        buy_sell = self.pretrends_df[['Buy/Sell']]
        buy_sell = buy_sell.dropna(axis=0)
        open_prices = open_prices.reset_index(drop=True)
        close_prices = close_prices.reset_index(drop=True)
        volume = volume.reset_index(drop=True)
        buy_sell = buy_sell.reset_index(drop=True)
        total_columns = open_prices.shape[1]

        # Initialize empty lists to store the chunks of (open, close, volume)
        combined_chunks = []
        normalized_prices_chunk = []
        normalized_volume_chunk = []
        scaler = MinMaxScaler()
        buy_sell_counter = 0

        # Iterate through the columns in chunks of {num_of_pretrend_days}
        for i in range(0, len(self.pretrends_df), self.num_of_pretrend_days):
            chunk_start = i
            chunk_end = min(i + self.num_of_pretrend_days, total_columns)

            # Get the chunk of columns
            chunk_open = open_prices.iloc[:, chunk_start:chunk_end]
            chunk_close = close_prices.iloc[:, chunk_start:chunk_end]
            chunk_volume = volume.iloc[:, chunk_start:chunk_end]

            # Concatenate the chunks along rows (axis=1) to create a single row dataframe
            # Then normalize the open and close prices together, by chunk
            combined_chunk = pd.concat([chunk_open, chunk_close], axis=1, ignore_index=True)
            normalized_prices_chunk = pd.DataFrame(scaler.fit_transform(combined_chunk.T).T, columns=combined_chunk.columns)
            normalized_volume_chunk = pd.DataFrame(scaler.fit_transform(chunk_volume.T).T, columns=chunk_volume.columns)

            # Get the 'Buy/Sell' value for the current chunk
            current_buy_sell = buy_sell.loc[buy_sell_counter, 'Buy/Sell']
            buy_sell_counter += 1

            # Append the combined chunk to the list
            combined_chunk_with_buy_sell = pd.concat([normalized_prices_chunk, normalized_volume_chunk], axis=1, ignore_index=True)
            combined_chunk_with_buy_sell['Buy/Sell'] = current_buy_sell
            combined_chunks.append(combined_chunk_with_buy_sell)           
        
            # Concatenate the list of combined chunks along rows to create the final dataframe
            final_combined_df = pd.concat(combined_chunks, ignore_index=True)
        
        # Store the structured and normalized dataframe
        self.structured_df = final_combined_df