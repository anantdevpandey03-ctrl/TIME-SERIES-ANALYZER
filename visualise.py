import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles

#LOADING CSV FILE
data = pd.read_csv("RELIANCE_data.CSV", skiprows=2)
print(data.columns)
print(data.head(10))
# seeting data as date pointer type
data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)
#setting date as index
data.set_index("Date",inplace=True)
#close price ka graph
plt.figure(figsize = (12,6))
plt.plot(data["Close"],color="green",linewidth=2)
plt.title("RELIANCE DATA OF 5 YEARS")
plt.grid(True)
plt.xlabel("Date")
plt.ylabel("Close PRICE")
plt.show()
#plt.figure(figsize = (12,6))