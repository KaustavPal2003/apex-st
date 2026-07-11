import numpy as np, json

# Check what dates look like in the npy file
dates = np.load('ADANIENT_apex_dates_train.npy')
print("NPY sample:", [str(d)[:10] for d in dates[:3]])

# Check what dates look like in the headlines CSV
import csv
with open('ADANIENT_headlines.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print("CSV sample:", [r['date'] for r in rows[:3]])