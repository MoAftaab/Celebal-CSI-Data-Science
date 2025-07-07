@echo off
echo Starting Network Anomaly Detection Improved Models Pipeline...

echo Step 1: Creating improved DBSCAN model...
python create_dbscan.py

echo Step 2: Creating improved Isolation Forest, One-Class SVM, and LOF models...
python improve_other_models.py

echo All models have been improved successfully!
echo Results are available in the visualizations directory.
pause 