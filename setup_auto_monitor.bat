@echo off
REM Windows Task Scheduler Setup for Auto-Monitoring
REM This creates a scheduled task that runs the monitor automatically

echo ======================================================================
echo AUTO-MONITOR SETUP - Windows Task Scheduler
echo ======================================================================
echo.
echo This will create a scheduled task to automatically start monitoring
echo when NBA games are about to begin.
echo.
echo Task will run:
echo   - Every day at 5:00 PM ET (before evening games start)
echo   - Script will check if games are live and monitor them
echo.

set SCRIPT_PATH=%~dp0auto_sms_monitor.py
set PYTHON_PATH=python

echo Creating scheduled task...
echo.

schtasks /create ^
  /tn "IgnitionAI_AutoMonitor" ^
  /tr "%PYTHON_PATH% %SCRIPT_PATH%" ^
  /sc daily ^
  /st 17:00 ^
  /rl highest ^
  /f

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [OK] Scheduled task created successfully!
    echo.
    echo Task Details:
    echo   - Name: IgnitionAI_AutoMonitor
    echo   - Runs: Daily at 5:00 PM
    echo   - Script: %SCRIPT_PATH%
    echo.
    echo To manage the task:
    echo   - View:   schtasks /query /tn "IgnitionAI_AutoMonitor"
    echo   - Run now: schtasks /run /tn "IgnitionAI_AutoMonitor"
    echo   - Delete: schtasks /delete /tn "IgnitionAI_AutoMonitor" /f
    echo.
) else (
    echo.
    echo [!] Failed to create task. Try running as Administrator:
    echo     Right-click this file -^> "Run as administrator"
    echo.
)

pause

