import pandas as pd
import numpy as np
from scipy import stats

def get_lists_from_performance(performance_exp):
    """Get lists of metrics from a performance file.

    Parameters
    ----------
    performance_path : str
        Path to the performance file.

    Returns
    -------
    list 
        List of all always right pids.
    list
        List of all always wrong pids.
    """

    performance_df = f'/projects/0/prjs1425/Osteosarcoma_WORC/WORC_COM_OS_res/WORC_{performance_exp}/performance_all_0.json'
    performance_df = pd.read_json(performance_df)

    always_right_pids = list(performance_df['Rankings']['Always right'].keys())
    always_wrong_pids = list(performance_df['Rankings']['Always wrong'].keys())

    return always_right_pids, always_wrong_pids

def statistical_comparison(list1, list2, performance_exp):
    """Perform statistical comparison between two lists of patient IDs.

    Parameters
    ----------
    list1 : list
        First list of patient IDs (e.g., always right).
    list2 : list
        Second list of patient IDs (e.g., always wrong).
    performance_exp : str
        Experiment identifier to construct clinical data path.

    Returns
    -------
    dict
        Dictionary containing comparison results for all clinical features.
    """
    
    # Load clinical data
    if not 'wir' in performance_exp and not 'cli' in performance_exp:
        clinical_info_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/{performance_exp[:-3]}/{performance_exp[-2:]}/clinical_features_with_Huvos.csv'
    else:
        clinical_info_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/exp_data/WIR/{performance_exp[4:-3]}/{performance_exp[-2:]}/clinical_features_with_WIR.csv'
    clinical_df = pd.read_csv(clinical_info_path)
    
    # Filter clinical data for both groups (direct match on Patient column)
    group1_data = clinical_df[clinical_df['Patient'].isin(list1)]
    group2_data = clinical_df[clinical_df['Patient'].isin(list2)]
    
    print(f"Group 1 (Always Right): {len(group1_data)} patients found")
    print(f"Group 2 (Always Wrong): {len(group2_data)} patients found")
    
    # Clinical features to analyze
    clinical_features = ['Age_Start', 'sex',
                         'pres_sympt', 'path_fract','location',
                         'diagnosis', 'metastasis','tumor_size','NAC']
    
    results = {}
    
    for feature in clinical_features:
        if feature in clinical_df.columns:
            # Extract data for both groups, removing NaN values
            data1 = group1_data[feature].dropna()
            data2 = group2_data[feature].dropna()
            
            # Skip if not enough data for statistical test
            if len(data1) < 2 or len(data2) < 2:
                results[feature] = {
                    'test': 'insufficient_data',
                    'group1_mean': np.mean(data1) if len(data1) > 0 else np.nan,
                    'group2_mean': np.mean(data2) if len(data2) > 0 else np.nan,
                    'group1_std': np.std(data1, ddof=1) if len(data1) > 0 else np.nan,
                    'group2_std': np.std(data2, ddof=1) if len(data2) > 0 else np.nan,
                    'group1_count': len(data1),
                    'group2_count': len(data2)
                }
                continue
            
            # Perform t-test for continuous variables
            if pd.api.types.is_numeric_dtype(clinical_df[feature]):
                t_stat, p_value = stats.ttest_ind(data1, data2, nan_policy='omit')
                results[feature] = {
                    'test': 't-test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'group1_mean': np.mean(data1),
                    'group2_mean': np.mean(data2),
                    'group1_std': np.std(data1, ddof=1),
                    'group2_std': np.std(data2, ddof=1),
                    'group1_count': len(data1),
                    'group2_count': len(data2)
                }
            # Perform chi-square test for categorical variables
            else:
                # Create contingency table
                from scipy.stats import chi2_contingency
                contingency_table = pd.crosstab(
                    pd.concat([pd.Series(data1), pd.Series(data2)]),
                    ['group1'] * len(data1) + ['group2'] * len(data2)
                )
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                results[feature] = {
                    'test': 'chi-square',
                    'statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'group1_counts': group1_data[feature].value_counts().to_dict(),
                    'group2_counts': group2_data[feature].value_counts().to_dict(),
                    'group1_count': len(data1),
                    'group2_count': len(data2)
                }
    
    return results

def print_statistical_results(results):
    """Print statistical results in a readable format."""
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON RESULTS")
    print("="*80)
    
    for feature, result in results.items():
        print(f"\n--- {feature} ---")
        
        if result['test'] == 'insufficient_data':
            print(f"  Insufficient data for statistical test")
            print(f"  Group 1 (Always Right): n={result['group1_count']}, mean={result['group1_mean']:.3f}")
            print(f"  Group 2 (Always Wrong): n={result['group2_count']}, mean={result['group2_mean']:.3f}")
        
        elif result['test'] == 't-test':
            significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            print(f"  T-test: t={result['statistic']:.3f}, p={result['p_value']:.4f} {significance}")
            print(f"  Group 1 (Always Right): n={result['group1_count']}, mean±std={result['group1_mean']:.2f}±{result['group1_std']:.2f}")
            print(f"  Group 2 (Always Wrong): n={result['group2_count']}, mean±std={result['group2_mean']:.2f}±{result['group2_std']:.2f}")
        
        elif result['test'] == 'chi-square':
            significance = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else "ns"
            print(f"  Chi-square: χ²={result['statistic']:.3f}, p={result['p_value']:.4f} {significance}")
            print(f"  Group 1 (Always Right) counts: {result['group1_counts']}")
            print(f"  Group 2 (Always Wrong) counts: {result['group2_counts']}")

if __name__ == "__main__":
    performance_exp = 'cli_T1W_v0'  # Example experiment identifier
    always_right, always_wrong = get_lists_from_performance(performance_exp)
    
    print("Always Right PIDs:", len(always_right))
    print("Always Wrong PIDs:", len(always_wrong))
    
    # Perform statistical analysis
    results = statistical_comparison(always_right, always_wrong, performance_exp)
    
    # Print results
    print_statistical_results(results)
    
    # Additional analysis: check if there are significant differences
    significant_features = [feature for feature, result in results.items() 
                          if result['test'] in ['t-test', 'chi-square'] and result['p_value'] < 0.05]
    
    print(f"\n{'='*50}")
    print(f"Significant differences (p < 0.05) found in {len(significant_features)} features:")
    for feature in significant_features:
        p_val = results[feature]['p_value']
        print(f"  - {feature}: p = {p_val:.4f}")