import pandas as pd
from pandas_profiling import ProfileReport
# Read the HCC Dataset
df = pd.read_csv("hcc-data.csv")
# Produce the data profiling report
original_report = ProfileReport(df, title='Original Data')
original_report.to_file("original_report.html")