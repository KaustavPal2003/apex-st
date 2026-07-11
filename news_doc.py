# save as news_doc.py
import json, csv, collections
from pathlib import Path

with open('logs/news_coverage.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['Symbol','Total Articles','Unique Dates','Earliest','Latest','Coverage Density %'])
    
    symbols = json.load(open('watchlist.json'))['watchlist']
    for sym in symbols:
        cache = Path(f'news_cache/{sym}.jsonl')
        if cache.exists():
            lines = cache.read_text(encoding='utf-8').splitlines()
            articles = [json.loads(l) for l in lines if l.strip()]
            dates = collections.Counter(a['date'] for a in articles)
            if dates:
                earliest = min(dates)
                latest = max(dates)
                # Approximate trading days between earliest and latest
                from datetime import datetime
                d1 = datetime.strptime(earliest, '%Y-%m-%d')
                d2 = datetime.strptime(latest, '%Y-%m-%d')
                total_days = (d2 - d1).days * 5 / 7  # rough trading day estimate
                density = round(len(dates) / max(total_days, 1) * 100, 1)
                w.writerow([sym, len(articles), len(dates), earliest, latest, density])
            else:
                w.writerow([sym, 0, 0, '-', '-', '0'])
        else:
            w.writerow([sym, 0, 0, '-', '-', '0'])

print("Generated: logs/news_coverage.csv")