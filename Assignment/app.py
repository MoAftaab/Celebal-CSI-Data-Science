import os
import sys
import streamlit as st

# Add the week 7 directory to the path
week7_path = os.path.join(os.path.dirname(__file__), "week 7")
sys.path.append(week7_path)

# Make week 7 imports available
try:
    # Change to the week 7 directory to ensure imports work correctly
    original_dir = os.getcwd()
    os.chdir(week7_path)
    
    # Import needed modules
    import utils
    import visualizations
    
    # Change back to original directory
    os.chdir(original_dir)
except Exception as e:
    st.error(f"Error importing modules: {str(e)}")
    st.exception(e)

# Check if the app file exists
app_path = os.path.join(week7_path, "app.py")
if not os.path.exists(app_path):
    st.error(f"Could not find app at {app_path}")
    st.write("Current directory:", os.getcwd())
    st.write("Files in current directory:", os.listdir())
    if os.path.exists("week 7"):
        st.write("Files in week 7 directory:", os.listdir("week 7"))
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