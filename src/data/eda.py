import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.load_data import load_data

def basic_eda(df):
    print("First Five Rows")
    print(df.head())
    print("Last Five Rows")
    print(df.tail())
    print("\n25 to 30 records:")
    print(df.iloc[24:30])
    print("Datatypes")
    print(df.dtypes)
    print("Complete Information")
    print(df.info())
    print("Duplicates")
    print(df.duplicated().sum())
    print("Null Values")
    print(df.isnull().sum())
    print(df["PlacementStatus"].value_counts())
    
    # Setup results directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Figure 1: Count of Placement Status
    count = df["PlacementStatus"].value_counts()
    plt.figure(1, figsize=(5, 6))
    plt.bar(count.index, count.values)
    plt.title("Count of Placement Status")
    plt.xlabel("Placement Status")
    plt.ylabel("Count")
    save_path_placement = os.path.join(results_dir, "images.png")
    plt.savefig(save_path_placement)
    print(f"Placement Status chart saved to {save_path_placement}")
    
    # Figure 2: Histogram of CGPA
    plt.figure(2, figsize=(6, 6))
    plt.hist(df["CGPA"], bins=10, edgecolor='black', alpha=0.85)
    plt.title("Histogram of CGPA")
    plt.xlabel("CGPA")
    plt.ylabel("Frequency")
    save_path_cgpa = os.path.join(results_dir, "cgpa_histogram.png")
    plt.savefig(save_path_cgpa)
    print(f"CGPA histogram chart saved to {save_path_cgpa}")
    
    # Figure 3: Distribution of Gender
    gender_counts = df["Gender"].value_counts()
    plt.figure(3, figsize=(6, 6))
    plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title("Distribution of Gender")
    plt.xlabel("Gender")
    save_path_gender = os.path.join(results_dir, "gender_distribution.png")
    plt.savefig(save_path_gender)
    print(f"Gender distribution chart saved to {save_path_gender}")
    
    # Figure 4: Correlation Heatmap of key metrics
    corr_cols = [
        'CGPA', 'Internships', 'Projects', 'PlacementStatus',
        'CodingTestScore', 'MockInterviewScore', 'AptitudeTestScore', 
        'SoftSkillsRating', 'ExtraCurricular'
    ]
    plt.figure(4, figsize=(10, 8))
    # Make sure we only correlate numerical columns
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Placement Metrics")
    plt.tight_layout()
    save_path_corr = os.path.join(results_dir, "correlation_heatmap.png")
    plt.savefig(save_path_corr)
    print(f"Correlation heatmap saved to {save_path_corr}")
    
    plt.show()

if __name__ == "__main__":
    df = load_data()
    basic_eda(df)