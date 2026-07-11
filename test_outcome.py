# test_outcome.py — run this while uvicorn is running
#test_outcome.py is a manual integration test, not pytest
import requests, json

BASE = "http://localhost:8000"

# 1. Check current CUSUM state for BAJFINANCE
r = requests.get(f"{BASE}/drift/BAJFINANCE")
print("Before:", json.dumps(r.json(), indent=2))

# 2. Log a realistic outcome — prediction was +0.8% log-return, actual was -1.2%
r = requests.post(f"{BASE}/outcome/BAJFINANCE", json={
    "predicted_return": 0.008,
    "actual_return": -0.012
})
print("\nOutcome 1:", json.dumps(r.json(), indent=2))

# 3. Log several more bad predictions in the same direction to trigger drift
for i in range(8):
    r = requests.post(f"{BASE}/outcome/BAJFINANCE", json={
        "predicted_return":  0.010,
        "actual_return":    -0.015   # consistently wrong upward bias
    })
    resp = r.json()
    print(f"Outcome {i+2}: S+={resp['s_pos']:.3f}  S-={resp['s_neg']:.3f}  "
          f"drift={resp['drift_flagged']}  msg={resp['message']}")

# 4. Check final state
r = requests.get(f"{BASE}/drift/BAJFINANCE")
print("\nAfter:", json.dumps(r.json(), indent=2))