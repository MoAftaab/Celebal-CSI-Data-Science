# PowerShell script to run Streamlit app
Write-Host "Starting Network Anomaly Detection Streamlit App..." -ForegroundColor Green

# Activate Python environment if needed
# Uncomment the following lines if you're using a virtual environment
# $env:Path = "$PSScriptRoot\venv\Scripts;$env:Path"

# Run the Streamlit app
python -m streamlit run app.py

Write-Host "Press any key to continue..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 