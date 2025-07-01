import os
import sys
import streamlit as st

# Set up path
app_path = os.path.join("Assignment", "week 7", "app.py")

# Check if app exists
if not os.path.exists(app_path):
    st.error(f"Could not find app at {app_path}")
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())
    if os.path.exists("Assignment"):
        st.write("Files in Assignment directory:", os.listdir("Assignment"))
        if os.path.exists(os.path.join("Assignment", "week 7")):
            st.write("Files in week 7 directory:", os.listdir(os.path.join("Assignment", "week 7")))
else:
    # Add the directory containing the app to Python path
    sys.path.append(os.path.join("Assignment", "week 7"))
    
    # Import the app module and run its main function
    with open(app_path, "r") as f:
        app_code = f.read()
    
    # Execute the app code
    exec(app_code) 