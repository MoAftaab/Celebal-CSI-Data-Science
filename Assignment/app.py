import os
import sys
import streamlit as st

# Add the week 7 directory to the path
week7_path = os.path.join(os.path.dirname(__file__), "Assignment", "week 7")
sys.path.append(week7_path)

# Check if the app file exists
app_path = os.path.join(week7_path, "app.py")
if not os.path.exists(app_path):
    st.error(f"Could not find app at {app_path}")
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())
    if os.path.exists("Assignment"):
        st.write("Files in Assignment directory:", os.listdir("Assignment"))
        if os.path.exists(os.path.join("Assignment", "week 7")):
            st.write("Files in week 7 directory:", os.listdir(os.path.join("Assignment", "week 7")))
else:
    # Import and execute the app
    try:
        # Change to the week 7 directory to ensure imports work correctly
        original_dir = os.getcwd()
        os.chdir(week7_path)
        
        # Execute the app
        with open("app.py", "r") as f:
            app_code = f.read()
            exec(app_code)
            
        # Change back to original directory
        os.chdir(original_dir)
    except Exception as e:
        st.error(f"Error running app: {str(e)}")
        st.exception(e) 