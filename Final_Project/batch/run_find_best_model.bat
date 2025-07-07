@echo off
cd %~dp0\..
python scripts/evaluation/find_best_model.py
echo.
echo Press any key to exit...
pause > nul 