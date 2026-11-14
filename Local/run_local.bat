@echo off
chcp 65001 >nul
REM 本機執行：第一次先安裝套件，之後直接啟動。瀏覽器開 http://127.0.0.1:7860
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
python app.py
pause
