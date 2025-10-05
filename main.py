import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''
Dataset used: New York Stock Exchange (Kaggle)

Label:
forecast_col: predicted closing prices

Features:
close: raw close price
pct_change: (close-prev_close)/prev_close
ma5: rolling mean of close over 5 days
ma10: rolling mean of close over 10 days
std5: rolling std over 5 days
volume: raw trading volume
'''

# DATA PREPARATION
def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out) # creating the label column with the last 5 rows are nan
    X = np.array(df[['close', 'pct_change', 'ma5', 'ma10', 'std5']]) # creating the feature arrays
    X = preprocessing.scale(X) # processing the feature arrays
    X_lately = X[-forecast_out:] # creating the column to be used later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True) # dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0) # cross validation

    response = [X_train,X_test , Y_train, Y_test , X_lately]
    return response

# Read data
df = pd.read_csv("prices.csv")
df = df[df.symbol == "GOOG"].copy()

# Add features
df['pct_change'] = df['close'].pct_change()
df['ma5'] = df['close'].rolling(window=5).mean()
df['ma10'] = df['close'].rolling(window=10).mean()
df['std5'] = df['close'].rolling(window=5).std()

df.dropna(inplace=True)

print(df.head())

forecast_col = 'close' # predictions based on past closing prices
forecast_out = 5 # predict 5 days into the future
test_size = 0.2 # using 20% of data for testing

# Split the data and fit into linear regression model
X_train, X_test, Y_train, Y_test , X_lately = prepare_data(df,forecast_col,forecast_out,test_size); # calling for cross validation and data preparation 
learner = LinearRegression() # initializing linear regression model
learner.fit(X_train,Y_train) # training the linear regression model

# Predict output
score = learner.score(X_test,Y_test) # testing the linear regression model
forecast = learner.predict(X_lately) # set that will contain the forecasted data
response = {} # creating json object
response['test_score'] = score
response['forecast_set'] = forecast

print(response)

# Add back to dataframe to visualize
df['forecast'] = np.nan
last_date = df.iloc[-1].name
forecast_dates = pd.date_range(start=last_date, periods=forecast_out+1, freq='D')[1:]
df_forecast = pd.DataFrame({'date': forecast_dates, 'forecast': forecast})

'''
Output: 
            date symbol        open       close         low        high      volume  pct_change         ma5        ma10       std5
4651  2010-01-15   GOOG  593.341025  580.000965  578.041006  593.561024  10909600.0   -0.016699  589.707002  600.366019   7.611211
5119  2010-01-19   GOOG  581.201005  587.620986  576.290999  590.420997   8665700.0    0.013138  587.008995  596.453011   4.172103
5587  2010-01-20   GOOG  585.981009  580.411005  575.290986  585.981009   6525700.0   -0.012270  584.994989  592.095006   4.495083
6055  2010-01-21   GOOG  583.441002  582.980970  572.251003  586.821000  12662600.0    0.004428  584.172985  589.567001   4.390549
6523  2010-01-22   GOOG  564.500980  550.010933  534.860888  570.600979  13651700.0   -0.056554  576.204972  585.157994  14.953825

{'test_score': 0.8925361306008751, 'forecast_set': array([787.41069122, 789.28167047, 782.35384546, 780.00468595, 768.31267392])}
test_score: 89.25% accuracy
forecast_set: Predicted closing prices for the next 5 days
'''

