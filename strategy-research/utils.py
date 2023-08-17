import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import talib

class DataHandler:
    
    def __init__(self, tickers:list, start_date:str, end_date:str, timeframe:str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.timeframe = timeframe
        self.df = None
        
        self.download_data()

    def download_data(self):
        try:
            self.df = yf.download(self.tickers, start=self.start_date, end=self.end_date, group_by='tickers', interval=self.timeframe)
            self.df.rename(columns={"Adj Close": "Adj_Close"}, inplace=True)
            return self.df
        except Exception as e:
            print(f"Error occurred while trying to download data: {e}")
            return None

    def null_check_and_clean(self):
        if self.df.isnull().sum().sum() > 0:
            null_columns = self.df.columns[self.df.isnull().any()]
            print("Null value count by column:")
            print(self.df[null_columns].isnull().sum())
            print("\nRows with null values:")
            print(self.df[self.df.isnull().any(axis=1)][null_columns])
            self.df.dropna(inplace=True)
            print("\nNull values dropped.")
        else:
            print("\nNo null values in dataframe.")

    def create_df(self):
        close_prices = pd.DataFrame()
        if isinstance(self.df.columns, pd.MultiIndex):
            for ticker in self.df.columns.levels[0]:
                close_prices[ticker] = self.df[ticker]['Close']
        else:
            if 'Close' in self.df.columns:
                close_prices = self.df[['Close']]
            else:
                raise KeyError("The dataframe doesn't have a 'Close' column.")
        return close_prices

class AnalysisTools:
    
    class Cointegration:
        def __init__(self, series1, series2):
            self.series1 = series1
            self.series2 = series2
            self.signif_value = 0.05
            
        def adf_test(self):
            """Perform ADF test on both series"""
            self._run_adf(self.series1, "Series 1")
            self._run_adf(self.series2, "Series 2")

        def _run_adf(self, series, series_name):
            """Helper function to run ADF test"""
            dftest = adfuller(series, autolag='AIC')
            adf = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '# Lags', '# Observations'])
            for key, value in dftest[4].items():
                adf['Critical Value (%s)' % key] = value
            print(f"Results for {series_name}:")
            print(adf)
            
            p = adf['p-value']
            if p <= self.signif_value:
                print(f"{series_name} is Stationary at {self.signif_value} level of significance\n")
            else:
                print(f"{series_name} is Non-Stationary at {self.signif_value} level of significance\n")

        def engle_granger_test(self):
            """Perform the Engle-Granger test for cointegration"""
            t_stat, p_value, _ = coint(self.series1, self.series2)
            print(f'Engle-Granger Test p-value: {p_value}')
            if p_value < self.signif_value:
                print(f"The series are cointegrated at {self.signif_value} level of significance.")
            else:
                print(f"The series are NOT cointegrated at {self.signif_value} level of significance.")

        def johansen_test(self):
            """Perform the Johansen test for cointegration"""
            dataframe = pd.concat([self.series1, self.series2], axis=1)
            
            # The det_order = -1 specifies no deterministic part in the model (i.e., constant or trend)
            result = coint_johansen(dataframe, det_order=-1, k_ar_diff=1)
            
            for trace, crit_val in zip(result.lr1, result.cvt[:, 1]):  # Using 5% critical values (index 1)
                if trace > crit_val:
                    print(f"The series are cointegrated at {self.signif_value} level of significance.")
                    return
            print(f"The series are NOT cointegrated at {self.signif_value} level of significance.")

    class TechnicalAnalysis:
        def __init__(self, dataframe):
            self.df = dataframe
            self.splitted_dfs = {}

        def split_dataframes(self):
            if isinstance(self.df.columns, pd.MultiIndex):
                for ticker in self.df.columns.levels[0]:
                    self.splitted_dfs[ticker] = self.df[ticker].copy()

        def combine_dataframes(self):
            # Filter each dataframe in the dictionary to only include 'ATR' and 'Close' columns
            filtered_dfs = {key: df[['Close', 'ATR']] for key, df in self.splitted_dfs.items()}
            combined_df = pd.concat(filtered_dfs.values(), axis=1, keys=filtered_dfs.keys())
            return combined_df


        def ATR(self, period=14):
            if isinstance(self.df.columns, pd.MultiIndex):
                self.split_dataframes()
                for key in self.splitted_dfs:
                    self.splitted_dfs[key]['ATR'] = talib.ATR(self.splitted_dfs[key]['High'],  self.splitted_dfs[key]['Low'],  self.splitted_dfs[key]['Close'], timeperiod=period)
                self.df = self.combine_dataframes()
            else:
                self.df['ATR'] = talib.ATR(self.df['High'], self.df['Low'], self.df['Close'], timeperiod=period)
            return self.df


class Visualization:
    
    @staticmethod
    def plot_data(df, title, is_normalized=False):
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot data against a simple range
        for column in df.columns:
            ax.plot(range(df.shape[0]), df[column].values, label=column)
            
        ax.legend(loc='upper left')
        
        # Set axis text color to white
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        ax.set_title(title)
        ax.set_xlabel('Date and Time')
        ax.set_ylabel('Normalized Price' if is_normalized else 'Price ($)')
        
        # Set x-ticks to represent specific dates and times
        start_ticks = df.index[df.index.time == pd.Timestamp("9:30").time()].strftime('%Y-%m-%d %H:%M').tolist()
        end_ticks = df.index[df.index.time == pd.Timestamp("10:30").time()].strftime('%Y-%m-%d %H:%M').tolist()
        date_ticks = start_ticks+end_ticks
        ax.set_xticks([df.index.get_loc(dt) for dt in date_ticks if dt in df.index])
        ax.set_xticklabels(date_ticks, rotation=45, ha='right')

        # Add a grid for the x-ticks
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.7)
        
        plt.tight_layout()
        plt.show()

class Normalization:
    
    @staticmethod
    def zscore(df):
        mean = df.mean()
        std_dev = df.std()
        zscores = (df - mean) / std_dev
        return zscores



def log_return(df,series):
    df['log_returns'] = np.diff(np.log(series))
    return df

def simple_returns(df, series):
    # Calculate simple returns
    df['simple_returns'] = series.pct_change()    
    return df
