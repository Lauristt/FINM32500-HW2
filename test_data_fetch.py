import pandas as pd
df =pd.read_parquet("/Users/laurisli/Desktop/FINM32500/HW2/Data/A.parquet")
print(df.shape,df.columns)
vol_series = df['Volume']
print(vol_series.head(5))