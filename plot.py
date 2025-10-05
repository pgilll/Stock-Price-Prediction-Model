from main import *
import matplotlib.pyplot as plt
import datetime

# Create copy of data
df_plot = df.copy()
# Create column for forecast
df_plot['forecast'] = np.nan

# Get last data from original data
if 'date' in df_plot.columns:
    last_date = pd.to_datetime(df_plot['date'].iloc[-1])
else:
    df_plot['date'] = pd.date_range(start='2020-01-01', periods=len(df_plot))
    last_date = df_plot.iloc[-1]

# Generate future dates for forecast
forecast_dates = pd.date_range(start=last_date, periods=forecast_out + 1, freq='D')[1:]

# Add forecast data to dataframe
forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast})

# Combine dataframes
df_plot = pd.concat([df_plot[['date', 'close']], forecast_df], ignore_index=True)

# PLOT
plt.figure(figsize=(12,6))
plt.title("GOOG Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Prices (USD)")

df_plot['date'] = pd.to_datetime(df_plot['date'])
plt.plot(df_plot['date'].values, df_plot['close'].values, label="Actual Close", color='blue')
plt.plot(df_plot['date'], df_plot['forecast'], label="Forecast (next 5 days)", color='orange', linestyle='--')

plt.axvline(x=last_date, color='gray', linestyle=':', label="Forecast Start")

plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()