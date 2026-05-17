"""
Streamlit Cloud entry point.

Streamlit Cloud looks for streamlit_app.py at the repo root.
All app logic lives in frontend/app.py; this file just invokes it.
"""

from frontend.app import main

main()
