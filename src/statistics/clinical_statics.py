import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class ClinicalFeatureAnalyzer:
    def __init__(self, data_path=None, df=None):
        """
        Initialize the analyzer with clinical data
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif df is not None:
            self.df = df.copy()
        else:
            raise ValueError("Either data_path or df must be provided")
        
        # Handle missing values
        self.df = self.df.replace('', np.nan)
        
    def data_overview(self):
        """Provide comprehensive data overview"""
        print("=" * 60)
        print("DATA OVERVIEW")
        print("=" * 60)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Number of patients: {self.df['Patient'].nunique()}")
        print(f"Number of images per patient:")
        print(self.df['Patient'].value_counts().describe())
        
        # Huvosnew distribution
        if 'Huvosnew' in self.df.columns:
            print(f"\nHuvosnew Distribution (0=good, 1=bad response):")
            huvos_counts = self.df['Huvosnew'].value_counts().sort_index()
            print(huvos_counts)
            print(f"Percentage of bad responders: {self.df['Huvosnew'].mean():.2%}")
        
    def p_value_analysis_huvosnew(self):
        """Comprehensive p-value analysis for Huvosnew prediction"""
        print("\n" + "=" * 60)
        print("P-VALUE ANALYSIS FOR HUVOSNEW PREDICTION")
        print("=" * 60)
        
        if 'Huvosnew' not in self.df.columns:
            print("Huvosnew column not found in dataset!")
            return
        
        results = []
        
        # Analyze each feature against Huvosnew
        features = ['Age_Start', 'geslacht', 'pres_sympt', 'path_fract', 
                   'Tumor_location', 'Soft_Tissue_Exp']
        
        for feature in features:
            if feature not in self.df.columns:
                continue
                
            print(f"\n--- {feature} vs Huvosnew ---")
            
            # Remove missing values for this analysis
            temp_df = self.df[[feature, 'Huvosnew']].dropna()
            
            if len(temp_df) == 0:
                print(f"No data available for {feature}")
                continue
            
            # Check if feature is numeric or categorical
            if temp_df[feature].dtype in [np.int64, np.float64] and temp_df[feature].nunique() > 2:
                # Numeric feature - use t-test or Mann-Whitney U
                group_0 = temp_df[temp_df['Huvosnew'] == 0][feature]
                group_1 = temp_df[temp_df['Huvosnew'] == 1][feature]
                
                if len(group_0) > 0 and len(group_1) > 0:
                    # Test for normality
                    _, p_norm_0 = stats.normaltest(group_0)
                    _, p_norm_1 = stats.normaltest(group_1)
                    
                    if p_norm_0 > 0.05 and p_norm_1 > 0.05:
                        # Use t-test for normal distributions
                        stat, p_value = stats.ttest_ind(group_0, group_1)
                        test_used = "T-test"
                    else:
                        # Use Mann-Whitney U for non-normal distributions
                        stat, p_value = stats.mannwhitneyu(group_0, group_1)
                        test_used = "Mann-Whitney U"
                    
                    print(f"Good response (0): n={len(group_0)}, mean={group_0.mean():.2f} ± {group_0.std():.2f}")
                    print(f"Bad response (1): n={len(group_1)}, mean={group_1.mean():.2f} ± {group_1.std():.2f}")
                    print(f"{test_used}: p-value = {p_value:.6f}")
                    
                    results.append({
                        'Feature': feature,
                        'Test': test_used,
                        'P_Value': p_value,
                        'Significant': p_value < 0.05,
                        'Effect_Size': group_1.mean() - group_0.mean(),
                        'Group_0_Mean': group_0.mean(),
                        'Group_1_Mean': group_1.mean()
                    })
                    
            else:
                # Categorical feature - use Chi-square test
                contingency_table = pd.crosstab(temp_df[feature], temp_df['Huvosnew'])
                
                if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    print("Contingency Table:")
                    print(contingency_table)
                    print(f"Chi-square test: p-value = {p_value:.6f}")
                    
                    results.append({
                        'Feature': feature,
                        'Test': 'Chi-square',
                        'P_Value': p_value,
                        'Significant': p_value < 0.05,
                        'Effect_Size': chi2
                    })
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            print("\n" + "=" * 60)
            print("SUMMARY OF SIGNIFICANT FEATURES")
            print("=" * 60)
            
            # Sort by p-value
            results_df = results_df.sort_values('P_Value')
            
            significant_features = results_df[results_df['Significant'] == True]
            non_significant_features = results_df[results_df['Significant'] == False]
            
            if len(significant_features) > 0:
                print("SIGNIFICANT FEATURES (p < 0.05):")
                print("-" * 40)
                for _, row in significant_features.iterrows():
                    if row['Test'] in ['T-test', 'Mann-Whitney U']:
                        print(f"✓ {row['Feature']}: p = {row['P_Value']:.6f}")
                        print(f"  Good response: {row['Group_0_Mean']:.2f}, Bad response: {row['Group_1_Mean']:.2f}")
                        print(f"  Difference: {row['Effect_Size']:.2f} ({row['Test']})")
                    else:
                        print(f"✓ {row['Feature']}: p = {row['P_Value']:.6f} (Chi-square)")
                    print()
            else:
                print("No significant features found at p < 0.05 level")
            
            if len(non_significant_features) > 0:
                print("NON-SIGNIFICANT FEATURES:")
                print("-" * 30)
                for _, row in non_significant_features.iterrows():
                    print(f"  {row['Feature']}: p = {row['P_Value']:.4f} ({row['Test']})")
        
        return results_df
    
    def visualize_significant_features(self, results_df):
        """Create visualizations for significant features"""
        if results_df.empty or 'Huvosnew' not in self.df.columns:
            return
            
        significant_features = results_df[results_df['Significant'] == True]['Feature'].tolist()
        
        if not significant_features:
            print("No significant features to visualize")
            return
            
        print(f"\nCreating visualizations for {len(significant_features)} significant features...")
        
        n_features = len(significant_features)
        n_cols = min(2, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        
        for i, feature in enumerate(significant_features):
            if i < len(axes):
                ax = axes[i]
                
                if self.df[feature].dtype in [np.int64, np.float64] and self.df[feature].nunique() > 2:
                    # Box plot for numeric features
                    self.df.boxplot(column=feature, by='Huvosnew', ax=ax)
                    ax.set_title(f'{feature} by Chemo Response\n(p={results_df[results_df["Feature"]==feature]["P_Value"].values[0]:.4f})')
                    ax.set_xlabel('Huvosnew (0=good, 1=bad)')
                else:
                    # Bar plot for categorical features
                    pd.crosstab(self.df[feature], self.df['Huvosnew']).plot(kind='bar', ax=ax)
                    ax.set_title(f'{feature} by Chemo Response\n(p={results_df[results_df["Feature"]==feature]["P_Value"].values[0]:.4f})')
                    ax.legend(['Good response', 'Bad response'])
                    plt.sca(ax)
                    plt.xticks(rotation=45)
                
        # Hide empty subplots
        for i in range(len(significant_features), len(axes)):
            axes[i].set_visible(False)
            
        plt.suptitle('Significant Clinical Features for Chemotherapy Response')
        plt.tight_layout()
        plt.show()

    def plot_p_value_summary(self, results_df):
        """Create a summary plot of p-values"""
        if results_df.empty:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Create bar plot of -log10(p-values)
        results_df = results_df.sort_values('P_Value', ascending=False)
        results_df['neg_log_p'] = -np.log10(results_df['P_Value'])
        
        colors = ['red' if sig else 'blue' for sig in results_df['Significant']]
        
        plt.barh(results_df['Feature'], results_df['neg_log_p'], color=colors, alpha=0.7)
        plt.axvline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05 threshold')
        
        plt.xlabel('-log10(p-value)')
        plt.title('Statistical Significance of Clinical Features\n(Red = Significant, Blue = Not Significant)')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def comprehensive_analysis(self):
        """Run all analyses"""
        self.data_overview()
        results_df = self.p_value_analysis_huvosnew()
        
        if not results_df.empty:
            self.visualize_significant_features(results_df)
            self.plot_p_value_summary(results_df)
        
        return results_df

# Example usage with your data
def example_usage():
    # Create sample data matching your format
    sample_data = {
        'Patient': ['OS_000001_01', 'OS_000001_02', 'OS_000001_03', 'OS_000002_01', 'OS_000003_01', 'OS_000004_01'],
        'Age_Start': [18, 18, 18, 48, 35, 62],
        'geslacht': [0, 0, 0, 1, 0, 1],
        'pres_sympt': [0.0, 0.0, 0.0, 1.0, 0.5, 1.0],
        'path_fract': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        'Tumor_location': [0, 0, 0, 1, 2, 1],
        'Soft_Tissue_Exp': [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        'Huvosnew': [0, 0, 0, 0, 1, 1]  # Added Huvosnew labels
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analyzer
    analyzer = ClinicalFeatureAnalyzer(df=df)
    
    # Run comprehensive analysis
    results = analyzer.comprehensive_analysis()
    
    return results

# Function to analyze your actual data
def analyze_your_data(csv_path):
    """Analyze your actual clinical data"""
    analyzer = ClinicalFeatureAnalyzer(data_path=csv_path)
    results = analyzer.comprehensive_analysis()
    return results

if __name__ == "__main__":
    # Run example
    # results = example_usage()
    
    # To use with your actual CSV file, uncomment and modify:
    results = analyze_your_data('/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/T1W/v0/clinical_features_with_Huvos.csv')