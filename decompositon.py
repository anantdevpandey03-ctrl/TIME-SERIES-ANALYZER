#IN THIS SECTION WE WILL DEAL WITH THREE THINGS OF THE FOLLOWING STOCK
# TREND,SEASONALITY,RESIDUAL(NOISE)
#TREND=A CERATAIN OVERALL PATTERN OF THE STOCK
#SEASONALITY=CERTAIN DOWNFALL OR BOUNCEBACK OF THE STOCK AT PARTICULAR TIME OR AN EVENT
#RESIDUAL=UNCERTAIN EVENTS THAT HAPPENS AT ANY POINT OF ALSO CALLED NOISE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
data=pd.read_csv("RELIANCE_data.CSV",skiprows=2)
print(data.columns)
data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)
data.set_index('Date',inplace=True)
#print(data.head(10))
series=data["Close"]#ffill=forward fill

#converting it to daily frequency
series=series.asfreq("D")
print(series)
#missing values fill
series=series.ffill().bfill()
decomposed=seasonal_decompose(series,model="additive",period=365)
decomposed.plot()
plt.show()
