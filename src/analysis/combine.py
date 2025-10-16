import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import argparse
import os

def combine_to_subject_level(y_score, y_truth, pids, method='average'):
    """
    Combine image-level predictions to subject level by different methods
    method can be 'majority', 'average', or 'max_prob'
    """
    subject_data = defaultdict(lambda: {'scores': [], 'truth': None})
    
    for fold_scores, fold_truths, fold_pids in zip(y_score, y_truth, pids):
        for score, truth, pid in zip(fold_scores, fold_truths, fold_pids):
            # Extract subject ID by removing the last 3 characters (_XX)
            subject_id = pid[:-3]
            
            subject_data[subject_id]['scores'].append(score)
            subject_data[subject_id]['truth'] = truth  # Same for all images of subject
    
    # Process scores for each subject based on method
    subject_scores = []
    subject_truths = []
    subject_ids = []
    
    for subject_id, data in subject_data.items():
        scores = np.array(data['scores'])
        
        if method == 'average':
            # Average the probability scores
            subject_scores.append(np.mean(scores))
        elif method == 'majority':
            # Majority vote based on threshold 0.5
            binary_predictions = (scores >= 0.5).astype(int)
            majority_class = 1 if np.mean(binary_predictions) >= 0.5 else 0
            
            # Get the average probability of the majority class predictions
            if majority_class == 1:
                majority_scores = scores[scores >= 0.5]
                subject_score = np.mean(majority_scores) if len(majority_scores) > 0 else 0.5
            else:
                majority_scores = scores[scores < 0.5]
                subject_score = np.mean(majority_scores) if len(majority_scores) > 0 else 0.5
            
            subject_scores.append(subject_score)
        elif method == 'max_prob':
            # Choose the prediction with highest confidence (farthest from 0.5)
            confidences = np.abs(scores - 0.5)
            max_confidence_idx = np.argmax(confidences)
            subject_scores.append(scores[max_confidence_idx])
        else:
            raise ValueError("Method must be 'majority', 'average', or 'max_prob'")
        
        subject_truths.append(data['truth'])
        subject_ids.append(subject_id)
    
    return np.array(subject_scores), np.array(subject_truths), subject_ids

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate all performance metrics
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # AUC
    auc = roc_auc_score(y_true, y_pred_proba)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)  # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Specificity (True Negative Rate)
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # NPV (Negative Predictive Value)
    fn = np.sum((y_pred == 0) & (y_true == 1))
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # BCA (Balanced Classification Accuracy)
    bca = (recall + specificity) / 2
    
    return {
        'AUC': auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity': recall,
        'Specificity': specificity,
        'F1-score': f1,
        'NPV': npv,
        'BCA': bca
    }

def bootstrap_confidence_interval(y_true, y_pred_proba, metric_func, n_bootstrap=1000, alpha=0.05):
    """
    Calculate bootstrap confidence intervals for a metric
    """
    n_samples = len(y_true)
    bootstrap_metrics = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrap_true = y_true[indices]
        bootstrap_pred = y_pred_proba[indices]
        
        try:
            metric_value = metric_func(bootstrap_true, bootstrap_pred)
            bootstrap_metrics.append(metric_value)
        except:
            continue
    
    if not bootstrap_metrics:
        return 0, (0, 0)
    
    # Calculate confidence interval
    lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    mean_metric = np.mean(bootstrap_metrics)
    
    return mean_metric, (lower, upper)

def metric_func_generator(metric_name, threshold=0.5):
    """Generate metric functions for bootstrap"""
    def metric_func(y_true, y_pred_proba):
        metrics = calculate_metrics(y_true, y_pred_proba, threshold)
        return metrics[metric_name]
    return metric_func

def evaluate_performance(y_score, y_truth, pids, threshold=0.5, n_bootstrap=1000, method='majority'):
    """
    Main function to evaluate performance at subject level with confidence intervals
    """
    # Combine to subject level
    subject_scores, subject_truths, subject_ids = combine_to_subject_level(y_score, y_truth, pids, method=method)
    
    # Calculate point estimates
    point_metrics = calculate_metrics(subject_truths, subject_scores, threshold)
    
    # Calculate confidence intervals for each metric
    metrics_with_ci = {}
    
    for metric_name in point_metrics.keys():
        metric_func = metric_func_generator(metric_name, threshold)
        mean_metric, ci = bootstrap_confidence_interval(
            subject_truths, subject_scores, metric_func, n_bootstrap
        )
        
        metrics_with_ci[metric_name] = {
            'mean': mean_metric,
            'ci_lower': ci[0],
            'ci_upper': ci[1]
        }
    
    return metrics_with_ci, subject_scores, subject_truths, subject_ids

def format_results(metrics_with_ci):
    """
    Format results in the desired output structure
    """
    statistics = {}
    
    for metric_name, values in metrics_with_ci.items():
        mean = values['mean']
        ci_lower = values['ci_lower']
        ci_upper = values['ci_upper']
        
        statistics[f"{metric_name} 95%:"] = f"{mean} ({ci_lower}, {ci_upper})"
    
    return {"Statistics": statistics}

# Example usage
if __name__ == "__main__":
    
    argparser = argparse.ArgumentParser(description="Evaluate subject-level performance metrics with confidence intervals.")
    argparser.add_argument('--input_json', type=str, 
                           default='/projects/0/prjs1425/Osteosarcoma_WORC/res_analysis/T1W_v0/results.json',
                           help='Path to the input JSON file containing y_score, y_truth, and pids.')
    argparser.add_argument('--method', type=str, choices=['majority', 'average', 'max_prob'], default='majority',
                           help='Method to combine image-level predictions to subject level.')
    args = argparser.parse_args()

    # Load your JSON data
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    
    # Extract data
    y_score = data['y_predictions']
    y_truth = data['y_truths']
    pids = data['pids']
    
    # Evaluate performance
    metrics_with_ci, subject_scores, subject_truths, subject_ids = evaluate_performance(
        y_score, y_truth, pids, threshold=0.5, n_bootstrap=1000, method=args.method
    )
    
    # Format results
    results = format_results(metrics_with_ci)
    
    # Print results
    print("Subject-level Performance Metrics with 95% Confidence Intervals:")
    for metric, value in results['Statistics'].items():
        print(f"  {metric}: {value}")
    
    # Print subject information
    print(f"\nTotal subjects: {len(subject_ids)}")
    print(f"Subjects: {subject_ids[:10]}...")  # Show first 10 subjects
    
    save_path = os.path.join(os.path.dirname(args.input_json), f'subject_level_performance_{args.method}.json')
    print(f"Results saved to: {save_path}")
    # You can also save the results
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)