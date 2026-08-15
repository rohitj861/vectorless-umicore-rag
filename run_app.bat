@echo off
REM Launch the Streamlit app with the CORRECT Python.
REM
REM This machine has two Python installations:
REM   Python 3.10  -> Streamlit 1.26  (too old: no st.write_stream)
REM   Python 3.14  -> Streamlit 1.58  (what this app needs)
REM
REM The bare `streamlit` command resolves to the 3.10 one, so always launch
REM through `python -m streamlit` instead. Double-click this file to run.

cd /d "%~dp0"
echo Starting Vectorless RAG on http://localhost:8501 ...
echo (close this window to stop the app)
echo.
python -m streamlit run app.py --server.port 8501 --server.address localhost
pause
