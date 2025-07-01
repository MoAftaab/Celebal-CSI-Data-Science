import os
import sys
import streamlit as st

# This is a simple redirect to the week 7 app
# It's needed because Streamlit Cloud has issues with spaces in folder names

# Add the week 7 directory to the path
week7_path = os.path.join(os.path.dirname(__file__), "week 7")
sys.path.append(week7_path)

# Change to the week 7 directory
original_dir = os.getcwd()
os.chdir(week7_path)

# Import and execute the week 7 app
try:
    import utils
    import visualizations
    
    # Execute the app
    from app import main
    main()
    
except Exception as e:
    st.error(f"Error running app: {str(e)}")
    st.exception(e)
    
    # Debug information
    st.write("Current directory:", os.getcwd())
    st.write("Python path:", sys.path)
    
    if os.path.exists("app.py"):
        st.write("app.py exists in current directory")
    else:
        st.write("app.py does not exist in current directory")
        
    st.write("Files in current directory:", os.listdir()) 