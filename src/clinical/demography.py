import pandas as pd 
import numpy as np 
from scipy.stats import chi2_contingency
from clean import get_clean_data, categorize

# Categorize ages into groups
def categorize_age(age):
    if age < 16:
        return "Children"
    elif 16 <= age < 40:
        return "AYA"
    else:
        return "Older Adults"

# Generate summary tables by age group
def generate_summary_table_by_age(df, variables):
    
    # Add age categories
    df['Age_Start'] = df['Age_Start'].replace(',','.', regex=True).astype(float)
    df['Age_group'] = df['Age_Start'].apply(categorize_age)

    # Initialize results dictionary
    results = {'Characteristic': [], 'N (%)': [], 'Children': [], 'AYA': [], 'Older Adults': [], 'p-Value': []}

    for variable in variables[1:]:

        # Clean the variable
        df[variable] = df[variable].replace(r'^\s*$', 'unknown', regex=True).fillna('unknown')
        df_variable = df

        # Overall count and percentage
        sum_variables = len(df_variable)
        results['Characteristic'].append(variable)  # Summary row for variable
        results['N (%)'].append(f"{sum_variables} (100.0)")
        results['Children'].append(f"{len(df_variable[df_variable['Age_group'] == 'Children'])} ({round(len(df_variable[df_variable['Age_group'] == 'Children']) / sum_variables * 100, 1)})")
        results['AYA'].append(f"{len(df_variable[df_variable['Age_group'] == 'AYA'])} ({round(len(df_variable[df_variable['Age_group'] == 'AYA']) / sum_variables * 100, 1)})")
        results['Older Adults'].append(f"{len(df_variable[df_variable['Age_group'] == 'Older Adults'])} ({round(len(df_variable[df_variable['Age_group'] == 'Older Adults']) / sum_variables * 100, 1)})")
        results['p-Value'].append("")  # Empty p-value for summary row

        # Count overall patients with this variable
        total_counts = df_variable[variable].value_counts().sort_index()
        total_percentages = (total_counts / len(df_variable) * 100).round(1)

        for level, count in total_counts.items():

            results['Characteristic'].append(level)
            results['N (%)'].append(f"{count} ({total_percentages[level]})")
            
            # Get age-group counts and percentages
            age_group_counts = df_variable.groupby('Age_group')[variable].value_counts().unstack().fillna(0)
            age_group_percentages = (age_group_counts.T / age_group_counts.sum(axis=1) * 100).round(1)
            
            # Children, AYA, and Older Adult values
            results['Children'].append(f"{int(age_group_counts.at['Children', level])} ({age_group_percentages.at[level, 'Children']})")
            results['AYA'].append(f"{int(age_group_counts.at['AYA', level])} ({age_group_percentages.at[level, 'AYA']})")
            results['Older Adults'].append(f"{int(age_group_counts.at['Older Adults', level])} ({age_group_percentages.at[level, 'Older Adults']})")
            
        # Calculate p-value for the variable across age groups
        chi2, p_value, _, _ = chi2_contingency(age_group_counts.T.values)
        results['p-Value'].append(round(p_value, 3))

        # Fill empty p-value cells for the remaining levels
        results['p-Value'].extend([''] * (len(total_counts) - 1))
    
    # Create a DataFrame from the results dictionary
    summary_df = pd.DataFrame(results)
    summary_df['p-Value'] = summary_df['p-Value'].fillna('-')
    
    return summary_df

# Generate summary tables by response group
from scipy.stats import ttest_ind
import pandas as pd
import numpy as np

def generate_summary_table_by_response(df, variables, response_column='Huvos'):
    """
    Generate summary table by poor and good response groups
    
    Parameters:
    df: pandas DataFrame
    variables: list of variables to analyze
    response_column: column name containing response data (default 'Huvos')
    """
    
    # Clean response column and categorize
    df[response_column] = df[response_column].replace(r'^\s*$', 'unknown', regex=True).fillna('unknown')
    
    # Define response groups (modify based on your Huvos grading system)
    def categorize_response(response):
        if 'Good' in str(response):  # Modified to handle string conversion
            return "Good Response"
        elif 'poor' in str(response).lower():  # Modified to handle case and string conversion
            return "Poor Response"
        else:
            return "Unknown"
    
    df['Response_group'] = df[response_column].apply(categorize_response)
    
    # Filter out unknown responses for analysis
    df_response = df[df['Response_group'] != 'Unknown']
    
    # Initialize results dictionary
    results = {
        'Characteristic': [], 
        'N (%)': [], 
        'Good Response': [], 
        'Poor Response': [], 
        'p-Value': []
    }

    for variable in variables:
        # Skip the response column itself to avoid circular analysis
        if variable == response_column:
            continue
            
        # Handle Age variable separately (continuous variable)
        if variable == 'Age_Start':
            # Clean age data
            df_response['Age_Start'] = df_response['Age_Start'].replace(',','.', regex=True).astype(float)
            
            # Calculate statistics for age
            good_response_ages = df_response[df_response['Response_group'] == 'Good Response']['Age_Start']
            poor_response_ages = df_response[df_response['Response_group'] == 'Poor Response']['Age_Start']
            
            # Overall statistics
            total_patients = len(df_response)
            mean_age = df_response['Age_Start'].mean()
            std_age = df_response['Age_Start'].std()
            
            # Group statistics
            good_mean = good_response_ages.mean()
            good_std = good_response_ages.std()
            poor_mean = poor_response_ages.mean()
            poor_std = poor_response_ages.std()
            
            # Add to results
            results['Characteristic'].append('Age_Start')
            results['N (%)'].append(f"{total_patients} (100.0)")
            results['Good Response'].append(f"{good_mean:.1f} ± {good_std:.1f}")
            results['Poor Response'].append(f"{poor_mean:.1f} ± {poor_std:.1f}")
            
            # Calculate p-value using t-test for continuous variable
            try:
                t_stat, p_value = ttest_ind(good_response_ages, poor_response_ages, nan_policy='omit')
                results['p-Value'].append(round(p_value, 3))
            except:
                results['p-Value'].append("N/A")
                
            continue  # Skip the rest of the loop for age variable
        
        # For categorical variables
        # Clean the variable
        df_response[variable] = df_response[variable].replace(r'^\s*$', 'unknown', regex=True).fillna('unknown')
        df_variable = df_response

        # Overall count and percentage
        sum_variables = len(df_variable)
        results['Characteristic'].append(variable)  # Summary row for variable
        results['N (%)'].append(f"{sum_variables} (100.0)")
        results['Good Response'].append(f"{len(df_variable[df_variable['Response_group'] == 'Good Response'])} ({round(len(df_variable[df_variable['Response_group'] == 'Good Response']) / sum_variables * 100, 1)})")
        results['Poor Response'].append(f"{len(df_variable[df_variable['Response_group'] == 'Poor Response'])} ({round(len(df_variable[df_variable['Response_group'] == 'Poor Response']) / sum_variables * 100, 1)})")
        results['p-Value'].append("")  # Empty p-value for summary row

        # Count overall patients with this variable
        total_counts = df_variable[variable].value_counts().sort_index()
        total_percentages = (total_counts / len(df_variable) * 100).round(1)

        for level, count in total_counts.items():
            results['Characteristic'].append(f"  {level}")  # Indent subcategories
            results['N (%)'].append(f"{count} ({total_percentages[level]})")
            
            # Get response-group counts and percentages
            response_group_counts = df_variable.groupby('Response_group')[variable].value_counts().unstack().fillna(0)
            response_group_percentages = (response_group_counts.T / response_group_counts.sum(axis=1) * 100).round(1)
            
            # Good and Poor Response values
            good_count = int(response_group_counts.at['Good Response', level]) if level in response_group_counts.columns else 0
            poor_count = int(response_group_counts.at['Poor Response', level]) if level in response_group_counts.columns else 0
            
            good_percent = response_group_percentages.at[level, 'Good Response'] if level in response_group_percentages.index else 0
            poor_percent = response_group_percentages.at[level, 'Poor Response'] if level in response_group_percentages.index else 0
            
            results['Good Response'].append(f"{good_count} ({good_percent})")
            results['Poor Response'].append(f"{poor_count} ({poor_percent})")
            
        # Calculate p-value for the variable across response groups
        try:
            chi2, p_value, _, _ = chi2_contingency(response_group_counts.T.values)
            results['p-Value'].append(round(p_value, 3))
        except:
            results['p-Value'].append("N/A")  # In case of calculation error

        # Fill empty p-value cells for the remaining levels
        results['p-Value'].extend([''] * (len(total_counts) - 1))
    
    # Create a DataFrame from the results dictionary
    summary_df = pd.DataFrame(results)
    summary_df['p-Value'] = summary_df['p-Value'].fillna('-')
    
    return summary_df

def main():
    data_dir = '/gpfs/work1/0/prjs1425/shark/clinical_features/osteosarcoma_t.csv'
    included_df = get_clean_data(data_dir)
    
    variables = ['Age_Start', 'sex', 'pres_sympt', 
                 'Location_extremity_no_extremity', 
                 'Diagnosis_high',
                #  'path_fract', 
                 'Distant_meta_pres',
                 'Size_primary_tumor', 
                 'CTX_pre_op_new',
                 'Huvos']
        
    data = included_df[variables]

    # Combine ‘<1 MAP or < 2AP’ with ‘1 MAP or 2 AP’. 
    data['CTX_pre_op_new'] = data['CTX_pre_op_new'].replace({'<1 MAP or <2 AP': '<=1 MAP or <=2 AP'})
    data['CTX_pre_op_new'] = data['CTX_pre_op_new'].replace({'1 MAP or 2 AP': '<=1 MAP or <=2 AP'})

    # Generate summary table by age
    summary_table_age = generate_summary_table_by_age(data, variables)
    print("Summary by Age Groups:")
    print(summary_table_age)
    save_path_age = '/exports/lkeb-hpc/xwan/osteosarcoma/preprocessing/clinical_analysis/demo/demo_by_age.csv'
    summary_table_age.to_csv(save_path_age, index=False)

    
    # Generate summary table by response
    summary_table_response = generate_summary_table_by_response(data, variables)
    print("\nSummary by Response Groups:")
    print(summary_table_response)
    save_path_response = '/exports/lkeb-hpc/xwan/osteosarcoma/preprocessing/clinical_analysis/demo/demo_by_response.csv'
    summary_table_response.to_csv(save_path_response, index=False)

if __name__ == "__main__":
    main()