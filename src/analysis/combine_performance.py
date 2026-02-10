import pandas as pd
import os
import re

def get_performance(json_path):
    try:
        # Read the JSON file
        data = pd.read_json(json_path)
        
        # Extract the Statistics series
        statistics = data['Statistics']
        
        # Extract the key metrics with 95% CI
        performance_metrics = {}
        for key, value in statistics.items():
            if key.endswith('95%:') and pd.notna(value):
                # Extract the metric name (remove " 95%:")
                metric_name = key.replace(' 95%:', '')
                # Format the value to 3 decimal places
                formatted_value = format_performance_value(value)
                performance_metrics[metric_name] = formatted_value
        
        return performance_metrics
    
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}
def get_combined_performance(exp_name, method='majority'):
    exp_name = exp_name.replace('WORC_', '')
    combined_json_path = f'/projects/0/prjs1425/Osteosarcoma_WORC/res_analysis/{exp_name}/subject_level_performance_{method}.json'
    try:
        data = pd.read_json(combined_json_path)
        statistics = data['Statistics']
        
        performance_metrics = {}
        for key, value in statistics.items():
            if key.endswith('95%:') and pd.notna(value):
                metric_name = key.replace(' 95%:', '')
                formatted_value = format_performance_value(value)
                performance_metrics[metric_name] = formatted_value
        
        return performance_metrics      
    except Exception as e:
        print(f"Error reading {combined_json_path}: {e}")
        return {}
    

def get_numbers(json_path):
    try:
        # Read the JSON file
        features_path = os.path.join(os.path.dirname(json_path), 'Features')
        
        # get number of files under features_path
        feature_files = [f for f in os.listdir(features_path) if os.path.isfile(os.path.join(features_path, f))]
        num_features = int(len(feature_files) / 4)
        
        return num_features
    
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return {}

def format_performance_value(value):
    """Format performance value to extract numbers and format to 3 decimal places"""
    if pd.isna(value):
        return ""
    
    value_str = str(value)
    
    # Pattern to match numbers like 0.49886843938424735 (0.3597529228855517, 0.637...)
    pattern = r'(\d+\.\d+)'
    matches = re.findall(pattern, value_str)
    
    if matches:
        # Format each number to 3 decimal places
        formatted_numbers = []
        for match in matches:
            try:
                num = float(match)
                formatted_num = f"{num:.3f}"
                formatted_numbers.append(formatted_num)
            except ValueError:
                formatted_numbers.append(match)
        
        # Reconstruct the formatted string
        if len(formatted_numbers) >= 3:
            return f"{formatted_numbers[0]} ({formatted_numbers[1]}, {formatted_numbers[2]})"
        elif len(formatted_numbers) == 1:
            return formatted_numbers[0]
        else:
            return " ".join(formatted_numbers)
    
    return value_str

if __name__ == "__main__":
    exp_names = os.listdir('/exports/lkeb-hpc/xwan/osteosarcoma/OS_res/results')
    
    included_exps = []

    # conditions 
    for exp in exp_names:
        # if 'cli' in exp or 'AG' in exp:
        included_exps.append(exp)
    # Create a dictionary to store all performance data
    performance_data = {}
    
    for exp_name in exp_names:
        json_path = f'/exports/lkeb-hpc/xwan/osteosarcoma/OS_res/results/{exp_name}/performance_all_0.json'
        performance = get_performance(json_path)
        majority_performance = get_combined_performance(exp_name, method = 'majority')
        mean_performance = get_combined_performance(exp_name, method = 'average')
        max_prob_performance = get_combined_performance(exp_name, method = 'max_prob')
        numbers = get_numbers(json_path)
        
        def add_performance_data(exp_name, performance, numbers):
            if performance:
                performance_data[exp_name] = {
                    'AUC': performance.get('AUC', ''),
                    'Accuracy': performance.get('Accuracy', ''),
                    'Sensitivity': performance.get('Sensitivity', ''),
                    'Specificity': performance.get('Specificity', ''),
                    'Numbers': numbers
                }
                
                print(f'Experiment: {exp_name} Performance:')
                print(f'  AUC: {performance.get("AUC", "N/A")}')
                print(f'  Accuracy: {performance.get("Accuracy", "N/A")}')
                print(f'  Sensitivity: {performance.get("Sensitivity", "N/A")}')
                print(f'  Specificity: {performance.get("Specificity", "N/A")}')
            else:
                print(f'Experiment: {exp_name} - No performance data found')
        
        add_performance_data(exp_name, performance, numbers)
        add_performance_data(f'{exp_name}_majority', majority_performance, numbers)
        add_performance_data(f'{exp_name}_mean', mean_performance, numbers)
        add_performance_data(f'{exp_name}_max_prob', max_prob_performance, numbers)        
    
    # Create the final table
    metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
    table_data = []
    
    for exp_name in sorted(performance_data.keys()):
        row_data = {'Experiment': exp_name}
        for metric in metrics:
            row_data[metric] = performance_data[exp_name].get(metric, '')
        row_data['Numbers'] = performance_data[exp_name].get('Numbers', 0)
        table_data.append(row_data)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(table_data)
    
    # Save to CSV
    df.to_csv('/exports/lkeb-hpc/xwan/osteosarcoma/OS_res/performance_table.csv', index=False)
    print(f"\nSaved performance table to 'performance_table.csv'")
    
    # Also create a more formatted version for display
    print("\nPerformance Table:")
    print("=" * 100)
    print(f"{'Experiment':<40} {'AUC':<25} {'Accuracy':<25} {'Sensitivity':<25} {'Specificity':<25}")
    print("-" * 100)
    
    for row in table_data:
        print(f"{row['Experiment']:<40} {row['AUC']:<25} {row['Accuracy']:<25} {row['Sensitivity']:<25} {row['Specificity']:<25}")