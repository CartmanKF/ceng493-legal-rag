@echo off
cd /d "%~dp0"
".\.conda-envs\legal-rag-gpu\python.exe" -m src.legal_rag.gui
