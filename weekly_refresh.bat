@echo off
cd E:\TradingProjects_new\apex_st_clean
powercfg /change standby-timeout-ac 0
python fetch_nse_data.py --start 2010-01-01 --screen --skip-fundamentals
python apex_synth_runner_v2.py --epochs 18
python sprint3_finbert.py
python sprint3_gat.py
python sprint4_fusion.py
python sprint5_ensemble.py
powercfg /change standby-timeout-ac 30
:: Restart API to pick up fresh artefacts
taskkill /f /im uvicorn.exe
timeout /t 3 /nobreak > nul
start "APEX API" cmd /k "uvicorn apex_inference:app --host 0.0.0.0 --port 8000"
echo Syncing fresh artefacts to AWS...
foreach ($s in @("ADANIENT","ADANIPORTS","APOLLOHOSP","BAJAJ-AUTO","BAJFINANCE","CIPLA","DIVISLAB","EICHERMOT","GRASIM","INDUSINDBK","JSWSTEEL","NESTLEIND","SUNPHARMA","TATASTEEL","TITAN")) {
    scp -i C:\Users\Admin\.ssh\apex-st-key.pem `
        "${s}_ensemble_result.json" "${s}_conformal.json" "${s}_cusum_state.json" `
        ubuntu@13.233.140.171:/opt/apex_st/
}
scp -i C:\Users\Admin\.ssh\apex-st-key.pem sprint5_summary.json ubuntu@13.233.140.171:/opt/apex_st/
ssh -i C:\Users\Admin\.ssh\apex-st-key.pem ubuntu@13.233.140.171 "screen -S apex-api -X quit; sleep 3; cd /opt/apex_st && source venv/bin/activate && screen -dmS apex-api uvicorn apex_inference:app --host 0.0.0.0 --port 8000"
echo AWS sync complete.