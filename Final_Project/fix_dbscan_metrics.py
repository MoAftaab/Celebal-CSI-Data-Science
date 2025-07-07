import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define paths
VISUALIZATIONS_DIR = "visualizations"

# Load model metrics CSV
csv_path = os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded model metrics from {csv_path}")
    
    # Check if DBSCAN is in the dataframe
    if 'model' in df.columns and 'DBSCAN' in df['model'].values:
        dbscan_idx = df[df['model'] == 'DBSCAN'].index[0]
        
        # Get existing metrics
        accuracy = df.at[dbscan_idx, 'accuracy']
        precision = df.at[dbscan_idx, 'precision']
        recall = df.at[dbscan_idx, 'recall']
        f1_score = df.at[dbscan_idx, 'f1_score']
        
        logger.info(f"DBSCAN metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1_score:.4f}")
        
        # Calculate confusion matrix values
        # We can derive them from precision and recall
        # Assuming we know the total number of test samples
        # Let's load the test data to get the actual counts
        try:
            y_test = np.load(os.path.join("data", "processed", "y_test.npy"))
            total_samples = len(y_test)
            positive_samples = np.sum(y_test == 1)
            negative_samples = total_samples - positive_samples
            
            logger.info(f"Test data: Total={total_samples}, Positive={positive_samples}, Negative={negative_samples}")
            
            # Calculate TP, FP, TN, FN from precision and recall
            # Recall = TP / (TP + FN)
            # Precision = TP / (TP + FP)
            
            # TP = Recall * (TP + FN) = Recall * Positive samples
            true_positive = recall * positive_samples
            
            # FN = Positive samples - TP
            false_negative = positive_samples - true_positive
            
            # FP = TP / Precision - TP
            false_positive = true_positive / precision - true_positive if precision > 0 else 0
            
            # TN = Negative samples - FP
            true_negative = negative_samples - false_positive
            
            # Calculate specificity
            specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            
            logger.info(f"Calculated metrics: TP={true_positive:.1f}, FP={false_positive:.1f}, TN={true_negative:.1f}, FN={false_negative:.1f}")
            logger.info(f"Specificity: {specificity:.4f}")
            
            # Update the dataframe
            df.at[dbscan_idx, 'true_positive'] = true_positive
            df.at[dbscan_idx, 'false_positive'] = false_positive
            df.at[dbscan_idx, 'true_negative'] = true_negative
            df.at[dbscan_idx, 'false_negative'] = false_negative
            df.at[dbscan_idx, 'specificity'] = specificity
            
            # Save the updated dataframe
            df.to_csv(csv_path, index=False)
            logger.info(f"Updated model metrics saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
    else:
        logger.error("DBSCAN not found in model metrics CSV")
else:
    logger.error(f"Model metrics CSV not found at {csv_path}")

logger.info("DBSCAN metrics fix completed") 