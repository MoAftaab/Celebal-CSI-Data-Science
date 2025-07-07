#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Anomaly Detection - Extended Model Comparison Visualizations
-------------------------------------------------------------------
This script generates advanced visualizations comparing the performance
of different anomaly detection models using the NSL-KDD dataset.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Define paths
VISUALIZATIONS_DIR = "visualizations"
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Load metrics data
metrics_file = os.path.join(VISUALIZATIONS_DIR, 'model_metrics.csv')
if not os.path.exists(metrics_file):
    logger.warning(f"Metrics file {metrics_file} not found. Please run model evaluation first.")
    exit(1)
metrics_df = pd.read_csv(metrics_file)
logger.info(f"Loaded metrics for models: {', '.join(metrics_df['model'].tolist())}")

# Define metrics to visualize
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity']

# Basic bar chart
plt.figure(figsize=(14, 8))
metrics_df.set_index('model')[metrics_to_plot].plot(kind='bar')
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'model_comparison_bar.png'))
plt.close()

# Radar chart
def radar_chart(df, metrics, title):
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    fig = plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], metrics)
    ax.set_ylim(0, 1)
    for idx, row in df.iterrows():
        values = row[metrics].tolist() + [row[metrics[0]]]
        ax.plot(angles, values, label=row['model'])
        ax.fill(angles, values, alpha=0.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title)
    return fig
radar_chart(metrics_df, metrics_to_plot, 'Radar Chart of Model Metrics').savefig(
    os.path.join(VISUALIZATIONS_DIR, 'model_comparison_radar.png'))
plt.close()

# Scatter plots
sns.scatterplot(data=metrics_df, x='accuracy', y='f1_score', s=200, hue='model')
plt.title('F1 Score vs. Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'f1_vs_accuracy.png'))
plt.close()

sns.scatterplot(data=metrics_df, x='recall', y='precision', s=200, hue='model')
plt.title('Precision vs. Recall')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'precision_vs_recall.png'))
plt.close()

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(metrics_df.set_index('model')[metrics_to_plot], annot=True, cmap='viridis')
plt.title('Performance Metrics Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'heatmap.png'))
plt.close()

# Boxplot
metrics_long = pd.melt(metrics_df, id_vars=['model'], value_vars=metrics_to_plot,
                       var_name='Metric', value_name='Value')
sns.boxplot(data=metrics_long, x='Metric', y='Value')
plt.title('Metrics Distribution')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'boxplot.png'))
plt.close()

# Parallel coordinates
plt.figure(figsize=(12, 8))
pd.plotting.parallel_coordinates(metrics_df, 'model', cols=metrics_to_plot)
plt.title('Parallel Coordinates of Model Metrics')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'parallel_coordinates.png'))
plt.close()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(metrics_df[metrics_to_plot].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Between Metrics')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, 'correlation_heatmap.png'))
plt.close()

# 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(metrics_df['accuracy'], metrics_df['precision'], metrics_df['recall'], s=60)
for i, row in metrics_df.iterrows():
    ax.text(row['accuracy'], row['precision'], row['recall'], row['model'], size=8)
ax.set_xlabel('Accuracy')
ax.set_ylabel('Precision')
ax.set_zlabel('Recall')
plt.tight_layout()
plt.savefig(os.path.join(VISUALIZATIONS_DIR, '3d_scatter.png'))
plt.close()

# Save all plots to a PDF
with PdfPages(os.path.join(VISUALIZATIONS_DIR, 'all_visuals.pdf')) as pdf:
    for img_file in os.listdir(VISUALIZATIONS_DIR):
        if img_file.endswith('.png'):
            img_path = os.path.join(VISUALIZATIONS_DIR, img_file)
            img = plt.imread(img_path)
            fig = plt.figure()
            plt.imshow(img)
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

logger.info("All visualizations generated and saved.")
