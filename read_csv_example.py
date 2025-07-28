import pandas as pd

df = pd.read_csv("GoogleStock Price.csv")
print(df.head())
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('GoogleStock Price.csv')

# Convert the 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Plot Closing Price over Time
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Close'], label='Close Price', color='blue')
plt.title('Google Stock Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()