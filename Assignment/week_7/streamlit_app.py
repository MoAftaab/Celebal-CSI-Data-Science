import os
import sys
import streamlit as st

# Set up path
app_path = "app.py"  # Use the app.py file in the same directory

# Check if app exists
if not os.path.exists(app_path):
    st.error(f"Could not find app at {app_path}")
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())
else:
    # Add the current directory to Python path
    sys.path.append(os.getcwd())
    
    # Import the app module and run its main function
    with open(app_path, "r") as f:
        app_code = f.read()
    
    # Execute the app code
    exec(app_code) 