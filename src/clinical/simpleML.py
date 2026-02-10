import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os

def run_simple_ML(data, label, result_path, exp_name='scores'):
    # List of classifiers and their names
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, random_state=42),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=42)

    # Initialize DataFrame to store classifier scores
    df = pd.DataFrame()
    classifiers_list, accuracy_scores = [], []

    # Iterate over classifiers
    for name, clf in zip(names, classifiers):
        # Create a pipeline with StandardScaler and classifier
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        classifiers_list.append(name)
        accuracy_scores.append(round(score, 2))
        print(f"Classifier: {name}, Score: {score}")

    # Save results to a CSV
    df['classifier'] = classifiers_list
    df['score'] = accuracy_scores

    # print average score
    print(f"Average score across classifiers: {df['score'].mean()}")
    df.to_csv(os.path.join(result_path, f'{exp_name}.csv'), index=False)

