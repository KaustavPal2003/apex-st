@echo off
cd /d E:\TradingProjects_new\apex_st_clean
start "APEX API" cmd /k "uvicorn apex_inference:app --host 0.0.0.0 --port 8000"
timeout /t 8 /nobreak > nul
start "APEX Dashboard" cmd /k "streamlit run dashboard.py --server.port 8501 --server.headless true"