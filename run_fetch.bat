@echo off
setlocal enabledelayedexpansion
echo ============================================================
echo  APEX-ST Sentinel News Backfill — 10 symbols
echo  Started: %DATE% %TIME%
echo ============================================================

set "FAILED="
set "OK_COUNT=0"
set "FAIL_COUNT=0"

REM ── CIPLA ───────────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol CIPLA --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X CIPLA FAILED
    set "FAILED=!FAILED! CIPLA"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK CIPLA done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── DIVISLAB ────────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol DIVISLAB --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X DIVISLAB FAILED
    set "FAILED=!FAILED! DIVISLAB"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK DIVISLAB done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── EICHERMOT ───────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol EICHERMOT --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X EICHERMOT FAILED
    set "FAILED=!FAILED! EICHERMOT"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK EICHERMOT done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── GRASIM ──────────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol GRASIM --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X GRASIM FAILED
    set "FAILED=!FAILED! GRASIM"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK GRASIM done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── INDUSINDBK ──────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol INDUSINDBK --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X INDUSINDBK FAILED
    set "FAILED=!FAILED! INDUSINDBK"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK INDUSINDBK done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── JSWSTEEL ────────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol JSWSTEEL --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X JSWSTEEL FAILED
    set "FAILED=!FAILED! JSWSTEEL"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK JSWSTEEL done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── NESTLEIND ───────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol NESTLEIND --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X NESTLEIND FAILED
    set "FAILED=!FAILED! NESTLEIND"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK NESTLEIND done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── SUNPHARMA ───────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol SUNPHARMA --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X SUNPHARMA FAILED
    set "FAILED=!FAILED! SUNPHARMA"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK SUNPHARMA done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── TATASTEEL ───────────────────────────────────────────────
python sentinel_news_fetcher.py --symbol TATASTEEL --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X TATASTEEL FAILED
    set "FAILED=!FAILED! TATASTEEL"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK TATASTEEL done
    set /a OK_COUNT+=1
)
echo   cooling down 90s...
timeout /t 90 /nobreak > nul

REM ── TITAN (last — no cooldown needed after) ────────────────
python sentinel_news_fetcher.py --symbol TITAN --source gdelt --weeks 156
if errorlevel 1 (
    echo [%DATE% %TIME%] X TITAN FAILED
    set "FAILED=!FAILED! TITAN"
    set /a FAIL_COUNT+=1
) else (
    echo [%DATE% %TIME%] OK TITAN done
    set /a OK_COUNT+=1
)

echo.
echo ============================================================
echo  ALL 10 SYMBOLS PROCESSED
echo  Finished: %DATE% %TIME%
echo  Succeeded: %OK_COUNT%   Failed: %FAIL_COUNT%
if defined FAILED (
    echo  Failed symbols:!FAILED!
) else (
    echo  Failed symbols: none
)
echo ============================================================
pause

endlocal
