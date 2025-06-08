import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def ensure_dataset_is_csv():
    """Ensures the Titanic dataset is available as a CSV file in 'Assignment/week 3/' folder.
    If not present, it creates the CSV from the Seaborn library.
    Returns the path to the CSV file.
    """
    # Relative path from the script's location
    csv_dir = os.path.join('Assignment', 'week 3') 
    csv_path = os.path.join(csv_dir, 'titanic_dataset.csv')
    
    # Get absolute path for checking existence and creating directory
    base_project_dir = os.path.dirname(os.path.abspath(__file__))
    abs_csv_dir = os.path.join(base_project_dir, csv_dir)
    abs_csv_path = os.path.join(abs_csv_dir, 'titanic_dataset.csv')

    if not os.path.exists(abs_csv_path):
        print(f"'{abs_csv_path}' not found. Creating it from Seaborn dataset...")
        if not os.path.exists(abs_csv_dir):
            os.makedirs(abs_csv_dir)
            print(f"Created directory: '{abs_csv_dir}'")
        
        titanic_df = sns.load_dataset('titanic')
        titanic_df.to_csv(abs_csv_path, index=False)
        print(f"Dataset saved to '{abs_csv_path}'")
    else:
        print(f"Dataset found at '{abs_csv_path}'")
    return abs_csv_path # Return absolute path for pd.read_csv

def create_visualizations():
    """Loads the Titanic dataset from CSV, preprocesses it, and generates visualizations."""
    
    csv_file_path = ensure_dataset_is_csv()
    
    # Load the dataset
    print(f"Loading Titanic dataset from '{csv_file_path}'...")
    df = pd.read_csv(csv_file_path)

    # --- Data Preprocessing (simplified based on common practices) ---
    print("Preprocessing data...")
    # Fill missing 'age' values with the median
    df['age'].fillna(df['age'].median(), inplace=True)

    # Fill missing 'embarked' and 'embark_town' with the mode
    df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)
    df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

    # Drop 'deck' column due to many missing values (as often done)
    # Check if 'deck' column exists before dropping, as it might be absent in some CSV versions
    if 'deck' in df.columns:
        df.drop(columns=['deck'], inplace=True)

    # --- Feature Engineering ---
    print("Performing feature engineering...")
    df['FamilySize'] = df['sibsp'] + df['parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    print("Added 'FamilySize' and 'IsAlone' features.")

    # --- Create visualizations directory ---
    # Visualizations directory relative to the script's location
    viz_dir_name = 'visualizations'
    base_project_dir = os.path.dirname(os.path.abspath(__file__))
    abs_viz_dir = os.path.join(base_project_dir, viz_dir_name)

    if not os.path.exists(abs_viz_dir):
        os.makedirs(abs_viz_dir)
    print(f"Ensured 'visualizations' directory exists at '{abs_viz_dir}'.")

    # --- Generate and Save Visualizations ---
    print("Generating visualizations...")

    # Helper function to save plots to the absolute visualization directory
    def save_plot(filename):
        plt.savefig(os.path.join(abs_viz_dir, filename))
        plt.clf()
        print(f"Saved: {filename} in {viz_dir_name}/")

    # 1. Survival Count
    plt.figure(figsize=(8, 6))
    sns.countplot(x='survived', data=df, palette='viridis')
    plt.title('Survival Count (0 = No, 1 = Yes)', fontsize=15)
    plt.xlabel('Survived', fontsize=12)
    plt.ylabel('Number of Passengers', fontsize=12)
    plt.xticks([0, 1], ['Did not Survive', 'Survived'])
    save_plot('survival_count.png')

    # 2. Survival Rate by Passenger Class (Pclass)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='pclass', y='survived', data=df, palette='viridis', ci=None)
    plt.title('Survival Rate by Passenger Class', fontsize=15)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    save_plot('survival_rate_by_pclass.png')

    # 3. Survival Rate by Sex
    plt.figure(figsize=(8, 6))
    sns.barplot(x='sex', y='survived', data=df, palette='viridis', ci=None)
    plt.title('Survival Rate by Sex', fontsize=15)
    plt.xlabel('Sex', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    save_plot('survival_rate_by_sex.png')

    # 4. Age Distribution by Survival Status
    plt.figure(figsize=(10, 6))
    g = sns.FacetGrid(df, col='survived', height=5, aspect=1, hue='survived', palette={0: 'red', 1: 'green'})
    g.map(sns.histplot, 'age', kde=True, bins=20)
    g.set_axis_labels('Age', 'Number of Passengers')
    g.set_titles(col_template="{col_name} (0=No, 1=Yes)")
    plt.suptitle('Age Distribution by Survival Status', y=1.03, fontsize=15)
    # Save FacetGrid differently
    plt.savefig(os.path.join(abs_viz_dir, 'age_distribution_by_survival.png'))
    plt.clf() # Clear the figure associated with FacetGrid
    print(f"Saved: age_distribution_by_survival.png in {viz_dir_name}/")

    # 5. Fare Distribution by Survival Status (Log scale for better visualization)
    plt.figure(figsize=(10, 6))
    df['fare_log'] = df['fare'].apply(lambda x: x + 0.001) # Avoid log(0)
    g = sns.FacetGrid(df, col='survived', height=5, aspect=1, hue='survived', palette={0: 'red', 1: 'green'})
    g.map(sns.histplot, 'fare_log', kde=True, bins=30, log_scale=True)
    g.set_axis_labels('Fare (Log Scale)', 'Number of Passengers')
    g.set_titles(col_template="{col_name} (0=No, 1=Yes)")
    plt.suptitle('Fare Distribution by Survival Status (Log Scale)', y=1.03, fontsize=15)
    plt.savefig(os.path.join(abs_viz_dir, 'fare_distribution_by_survival.png'))
    plt.clf()
    print(f"Saved: fare_distribution_by_survival.png in {viz_dir_name}/")

    # 6. Correlation Heatmap (only numerical features)
    plt.figure(figsize=(12, 10))
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # Ensure 'fare_log' is excluded if it was added temporarily for plotting and not part of main numerical analysis
    if 'fare_log' in numerical_cols:
        numerical_cols = numerical_cols.drop('fare_log')
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Features', fontsize=15)
    save_plot('correlation_heatmap.png')

    # 7. Survival Rate by Embark Town
    plt.figure(figsize=(8, 6))
    sns.barplot(x='embark_town', y='survived', data=df, palette='viridis', ci=None)
    plt.title('Survival Rate by Embarkation Town', fontsize=15)
    plt.xlabel('Embarkation Town', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    save_plot('survival_rate_by_embark_town.png')

    # 8. Advanced Violin Plot: Age distribution by Pclass and Sex, segmented by Survival
    print("Generating: Age Distribution by Pclass, Sex, and Survival (Violin Catplot)")
    g = sns.catplot(x="pclass", y="age", hue="survived", col="sex", data=df, kind="violin", split=True, height=6, aspect=.8, palette={0: "lightcoral", 1: "lightgreen"}, inner="quartile")
    g.fig.suptitle('Age Distribution by Pclass, Sex, and Survival', y=1.03, fontsize=16)
    g.set_axis_labels("Passenger Class", "Age")
    g.set_titles("{col_name}")
    # Adjust legend
    handles, legend_labels = g.axes[0,0].get_legend_handles_labels()
    if handles: # Check if legend items exist
        new_labels = ['Did not Survive' if label == '0' else 'Survived' for label in legend_labels]
        g.fig.legend(handles, new_labels, title='Survival Status', loc='upper right', bbox_to_anchor=(0.95, 0.95))
    for ax_row in g.axes: # Remove individual plot legends if a figure-level legend is added
        for ax in ax_row:
            if ax.get_legend() is not None:
                ax.get_legend().remove()
    plt.savefig(os.path.join(abs_viz_dir, 'age_pclass_sex_survival_violin_catplot.png'))
    plt.clf() # Clear the current figure
    print(f"Saved: age_pclass_sex_survival_violin_catplot.png in {viz_dir_name}/")

    # 9. FacetGrid: Survival counts by Pclass, Sex, and Embark Town
    print("Generating: Survival Counts by Pclass, Sex, and Embark Town (FacetGrid)")
    g = sns.FacetGrid(df, row='embark_town', col='pclass', hue='survived', margin_titles=True, height=4, aspect=1.2, palette={0: 'salmon', 1: 'skyblue'})
    g.map(sns.countplot, 'sex', order=['male', 'female']).add_legend(title='Survived', labels=['No', 'Yes'])
    g.set_axis_labels("Sex", "Count")
    g.set_titles(row_template="{row_name}", col_template="Class {col_name}")
    g.fig.suptitle('Survival Counts by Pclass, Sex, and Embark Town', y=1.03, fontsize=16)
    plt.savefig(os.path.join(abs_viz_dir, 'survival_by_pclass_sex_embark_facetgrid.png'))
    plt.clf()
    print(f"Saved: survival_by_pclass_sex_embark_facetgrid.png in {viz_dir_name}/")

    # 10. Survival Rate by FamilySize
    print("Generating: Survival Rate by Family Size")
    plt.figure(figsize=(12, 7))
    sns.barplot(x='FamilySize', y='survived', data=df, palette='YlGnBu', errorbar=None) # ci is deprecated, use errorbar=None
    plt.title('Survival Rate by Family Size', fontsize=16)
    plt.xlabel('Family Size (Passenger + Siblings/Spouses + Parents/Children)', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    save_plot('survival_rate_by_family_size.png') # Uses existing save_plot helper

    # 11. Point Plot: Survival Rate by Pclass and Embark Town
    print("Generating: Survival Rate by Pclass and Embark Town (Point Plot)")
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='pclass', y='survived', hue='embark_town', data=df, palette='Set2', dodge=True, errorbar=None)
    plt.title('Survival Rate by Passenger Class and Embarkation Town', fontsize=16)
    plt.xlabel('Passenger Class', fontsize=12)
    plt.ylabel('Survival Rate', fontsize=12)
    plt.legend(title='Embarkation Town')
    save_plot('survival_rate_by_pclass_embark_town_pointplot.png')

    # 12. Pair Plot of Key Numerical Features by Survival
    print("Generating: Pair Plot of Key Features by Survival Status")
    pair_plot_features = ['survived', 'pclass', 'age', 'fare', 'FamilySize']
    df_pairplot = df[pair_plot_features].copy()
    
    plt.figure() # Ensure clean state for pairplot
    pair_plot_obj = sns.pairplot(df_pairplot, hue='survived', diag_kind='kde',
                                 palette={0: 'orangered', 1: 'mediumseagreen'},
                                 plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'w'},
                                 height=2.2, aspect=1.1)
    pair_plot_obj.fig.suptitle('Pair Plot of Key Features by Survival Status', y=1.02, fontsize=16)
    # Update legend for pairplot
    handles = pair_plot_obj._legend_data.values()
    new_legend_labels = ['Did not Survive', 'Survived']
    pair_plot_obj.fig.legend(handles=handles, labels=new_legend_labels, loc='upper right', title='Survival Status', bbox_to_anchor=(0.95, 0.95))
    if pair_plot_obj.legend is not None: # Remove default legend if fig legend is added
        pair_plot_obj.legend.remove()

    plt.savefig(os.path.join(abs_viz_dir, 'pairplot_key_features_by_survival.png'))
    plt.clf()
    print(f"Saved: pairplot_key_features_by_survival.png in {viz_dir_name}/")

    print(f"\nAll visualizations generated and saved in the '{viz_dir_name}' directory.")

if __name__ == '__main__':
    create_visualizations()
