import json, collections

with open('news_cache/APOLLOHOSP.jsonl', encoding='utf-8') as f:
    lines = f.readlines()

by_date = collections.Counter(json.loads(l)['date'] for l in lines)

print(f'Total: {len(lines)} articles, {len(by_date)} unique dates')
print('Range:', min(by_date), '->', max(by_date))