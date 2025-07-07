import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme()

# Read the training dataset
df = pd.read_csv(os.path.join('..', 'data', 'raw', 'Training Dataset.csv'))

# 1. Loan Status Distribution (Pie Chart)
plt.figure(figsize=(10, 8))
loan_status_counts = df['Loan_Status'].value_counts()
plt.pie(loan_status_counts, labels=loan_status_counts.index, autopct='%1.1f%%')
plt.title('Loan Status Distribution')
plt.savefig('loan_status_pie.png', bbox_inches='tight', dpi=300)
plt.close()

# 2. Credit History vs Loan Status
plt.figure(figsize=(12, 6))
credit_loan_status = pd.crosstab(df['Credit_History'], df['Loan_Status'])
credit_loan_status.plot(kind='bar', stacked=True)
plt.title('Credit History vs Loan Status')
plt.xlabel('Credit History')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.savefig('credit_history_vs_loan_status.png', bbox_inches='tight', dpi=300)
plt.close()

# 3. Income Distribution by Loan Status
plt.figure(figsize=(12, 6))
sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)
plt.title('Applicant Income Distribution by Loan Status')
plt.xlabel('Loan Status')
plt.ylabel('Applicant Income')
plt.savefig('income_by_loan_status.png', bbox_inches='tight', dpi=300)
plt.close()

# 4. Additional visualization: Property Area vs Loan Status
plt.figure(figsize=(12, 6))
property_loan_status = pd.crosstab(df['Property_Area'], df['Loan_Status'], normalize='index') * 100
property_loan_status.plot(kind='bar', stacked=True)
plt.title('Loan Approval Rate by Property Area')
plt.xlabel('Property Area')
plt.ylabel('Percentage')
plt.legend(title='Loan Status')
plt.savefig('property_area_loan_status.png', bbox_inches='tight', dpi=300)
plt.close()

# 5. Additional visualization: Education vs Loan Status
plt.figure(figsize=(12, 6))
education_loan_status = pd.crosstab(df['Education'], df['Loan_Status'], normalize='index') * 100
education_loan_status.plot(kind='bar', stacked=True)
plt.title('Loan Approval Rate by Education')
plt.xlabel('Education')
plt.ylabel('Percentage')
plt.legend(title='Loan Status')
plt.savefig('education_loan_status.png', bbox_inches='tight', dpi=300)
plt.close() 