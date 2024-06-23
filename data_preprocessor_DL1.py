import requests
from datetime import date, timedelta
import json
import pandas as pd

current_date = date.today()

daily_start_date = current_date - timedelta(days=367) # 1 year
daily_end_date = current_date - timedelta(days=1)

daily_records = None
for i in range(10):
    daily_url = f"https://marketdata.tradermade.com/api/v1/timeseries?api_key=aWbM1qDWUTAMn6beRiL_&currency=XAUUSD&format=records&start_date={daily_start_date}&end_date={daily_end_date}&interval=daily&period=1"
    response = requests.get(daily_url).text
    dr = json.loads(response)['quotes']
    dr = pd.DataFrame(dr)
    if daily_records is None:
        daily_records = dr
    else:
        daily_records = dr.append(daily_records, ignore_index=True)
    daily_end_date = daily_start_date - timedelta(days=1)
    daily_start_date = daily_end_date - timedelta(days=366)

daily_records = daily_records.dropna()
daily_records.to_csv('C:/Users/User/Desktop/Trading Research/program/gold_data.csv', index=False)