import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import matplotlib.pyplot as plt
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
from joblib import Parallel, delayed
import os
import sys
from datetime import datetime
from scipy.stats import gaussian_kde
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="The objective has been evaluated at point.*")
warnings.filterwarnings("ignore", category=Warning, message="Stochastic Optimizer: Maximum iterations.*")
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=Warning, module="sklearn.neural_network._multilayer_perceptron", message="Stochastic Optimizer: Maximum iterations.*")
warnings.filterwarnings("ignore", category=UserWarning, module="skopt.optimizer.optimizer", message="The objective has been evaluated at point.*")


### Outlier detection (Tukey method)
def remove_outliers(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    LF = Q1 - 1.5 * IQR
    UF = Q3 + 1.5 * IQR
    data = data[~((data < LF) | (data > UF)).any(axis=1)]
    return data


### Model Evaluation Metrics
def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
       Evaluate model performance using cross-validation and regression metrics.

       Parameters:
           model: Scikit-learn regressor object.
           X_train (np.ndarray): Training features.
           X_test (np.ndarray): Test features.
           y_train (np.ndarray): Training target.
           y_test (np.ndarray): Test target.

       Returns:
           tuple: K-Fold R², Train R², Test R², Train RMSE, Test RMSE,
                  Train MAPE, Test MAPE, and training time.
       """
    start_time = time.time()
    kfold_r2_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=1)
    kfold_r2 = np.mean(kfold_r2_scores)

    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    return kfold_r2, train_r2, test_r2, rmse_train, rmse_test, mape_train, mape_test, train_time


### Conformal Prediction via Cross-Validation
def cv_conformal_prediction(model, X, y, X_test, significance=0.1, n_splits=5):
    kf = KFold(n_splits=n_splits)
    nonconformity_scores = []

    for train_index, val_index in kf.split(X):
        X_train_cv, X_val_cv = X[train_index], X[val_index]
        y_train_cv, y_val_cv = y[train_index], y[val_index]

        model.fit(X_train_cv, y_train_cv)
        y_pred_val = model.predict(X_val_cv)
        scores = np.abs(y_pred_val - y_val_cv)
        nonconformity_scores.extend(scores)

    nonconformity_scores = np.array(nonconformity_scores)
    alpha = np.ceil((1 - significance) * (len(nonconformity_scores) + 1)) / len(nonconformity_scores)
    threshold = np.quantile(nonconformity_scores, alpha)

    y_pred_test = model.predict(X_test)
    lower_bounds = y_pred_test - threshold
    upper_bounds = y_pred_test + threshold

    return np.column_stack((lower_bounds, upper_bounds))


### Coverage and Interval Size Calculation
def calculate_coverage_and_size(predictions, y_test):
    lower_bounds = predictions[:, 0]
    upper_bounds = predictions[:, 1]
    coverage = np.mean((y_test >= lower_bounds) & (y_test <= upper_bounds))
    interval_size = np.mean(upper_bounds - lower_bounds)
    return coverage, interval_size

# Develop and evaluate each model
def process_model(model_name, model_info):
    """
    Execute hyperparameter tuning, model evaluation, and conformal prediction.

    Parameters:
        model_name (str): Name of the model.
        model_config (dict): Dictionary containing model instance and parameter space.

    Returns:
        tuple: Model name, best model, best hyperparameters, and evaluation metrics.
    """
    print(f"Starting {model_name}...")
    bayes_search = BayesSearchCV(
        model_info['model'],
        model_info['params'],
        n_iter=50,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=0
    )
    bayes_search.fit(X_train, y_train)
    best_model = bayes_search.best_estimator_
    best_params = bayes_search.best_params_
    print(f"Completed {model_name}")

    metrics = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    cv_predictions = cv_conformal_prediction(best_model, X_train, y_train, X_test)
    coverage, interval_size = calculate_coverage_and_size(cv_predictions, y_test)

    y_test_pred = best_model.predict(X_test)
    residuals = y_test - y_test_pred

    kde = gaussian_kde(residuals)
    x_grid = np.linspace(min(residuals), max(residuals), 1000)
    pdf = kde(x_grid)

    hist, bin_edges = np.histogram(residuals, bins=20)

    return (model_name, best_model, best_params, {
        'K-Fold R²': metrics[0],
        'Train R²': metrics[1],
        'Test R²': metrics[2],
        'Train RMSE': metrics[3],
        'Test RMSE': metrics[4],
        'Train MAPE': metrics[5],
        'Test MAPE': metrics[6],
        'Time': metrics[7],
        'Coverage': coverage,
        'Average Interval Size': interval_size,
        'Residuals': residuals.tolist(),
        'KDE_PDF': pdf.tolist(),
        'X_Grid': x_grid.tolist(),
        'Histogram': hist.tolist(),
        'Bin_Edges': bin_edges.tolist()
    })


### Main Execution Pipeline
if __name__ == "__main__":
    # ----------------------
    # 1. Data Loading & Prep
    # ----------------------
    file_path = "simulation_data.xlsx"
    df = pd.read_excel(file_path)

    # Create results directory
    data_file_name = os.path.splitext(os.path.basename(file_path))[0]
    current_time = datetime.now().strftime("%m%d_%H%M%S")
    folder_path = f"ML_model_{data_file_name}_{current_time}"
    os.makedirs(folder_path, exist_ok=True)

    # Preprocess data
    df = remove_outliers(df)
    features = ['Pressure', 'Power', 'Spacing', 'Depo_time', 'SiH4', 'NH3', 'N2', 'H2']
    target = 'rate'
    X = df[features].values
    y = df[target].values

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=123
    )

    # ----------------------
    # 2. Model Definitions
    # ----------------------
    models = {
        "Random Forest": {
            "model": RandomForestRegressor(random_state=0),
            "params": {
                "n_estimators": Integer(50, 300),
                "max_depth": Integer(2, 8)
            }
        },
        "CatBoost": {
            "model": CatBoostRegressor(random_state=0, silent=True),
            "params": {
                "iterations": Integer(50, 300),
                "depth": Integer(2, 8),
                "learning_rate": Real(0.005, 0.5, prior="log-uniform")
            }
        },
        "Decision Tree": {
            "model": DecisionTreeRegressor(random_state=0),
            "params": {
                "max_depth": Integer(2, 8)
            }
        },
        "KNN": {
            "model": KNeighborsRegressor(),
            "params": {
                "n_neighbors": Integer(1, 50),
                "weights": Categorical(["uniform", "distance"])
            }
        },
        "BP Neural Network": {
            "model": MLPRegressor(random_state=0, max_iter=500),
            "params": {
                "hidden_layer_sizes": Integer(50, 200),
                "learning_rate_init": Real(0.0001, 0.5, prior="log-uniform"),
                "alpha": Real(0.0001, 0.1, prior="log-uniform"),
                "activation": Categorical(["relu", "tanh", "logistic"])
            }
        },
        "SVR": {
            "model": SVR(),
            "params": {
                "C": Real(0.1, 100, prior="log-uniform"),
                "gamma": Real(0.001, 1, prior="log-uniform"),
                "kernel": Categorical(["linear", "rbf", "poly"])
            }
        }
    }

    # ----------------------
    # 3. Parallel Model Execution
    # ----------------------
    start_time = time.time()

    results_list = Parallel(n_jobs=-1)(
        delayed(process_model)(model_name, model_info)
        for model_name, model_info in models.items()
    )

    results = {model_name: (best_model, best_params, metrics) for model_name, best_model, best_params, metrics in
               results_list}
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

    # ----------------------
    # 4. Console Output
    # ----------------------
    for model_name, (_, best_params, metrics) in results.items():
        print(f"{model_name}:\n")
        print(f"  K-Fold R²: {metrics['K-Fold R²']:.4f}")
        print(f"  Train R²: {metrics['Train R²']:.4f}")
        print(f"  Test R²: {metrics['Test R²']:.4f}")
        print(f"  Train RMSE: {metrics['Train RMSE']:.4f}")
        print(f"  Test RMSE: {metrics['Test RMSE']:.4f}")
        print(f"  Train MAPE: {metrics['Train MAPE']:.4f}")
        print(f"  Test MAPE: {metrics['Test MAPE']:.4f}")
        print(f"  Time: {metrics['Time']:.4f} seconds")
        print(f"  Coverage: {metrics['Coverage']:.4f}")
        print(f"  Average Interval Size: {metrics['Average Interval Size']:.4f}")
        print(f"  Best Hyperparameters: {best_params}\n")

    # ----------------------
    # 5. Scatter Plot Visualization
    # ----------------------
    n_models = len(models)
    fig, axes = plt.subplots(
        nrows=(n_models + 2) // 3,  # Max 3 columns
        ncols=3,
        figsize=(15, 5 * ((n_models + 2) // 3)),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    for i, (model_name, (best_model, _, metrics)) in enumerate(results.items()):
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        axes[i].scatter(y_train, y_train_pred, color='royalblue', alpha=0.6, label='Train')
        axes[i].scatter(y_test, y_test_pred, color='darkorange', alpha=0.6, label='Test')
        min_val = min(y.min(), y_train_pred.min(), y_test_pred.min())
        max_val = max(y.max(), y_train_pred.max(), y_test_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1, label='y = x')
        axes[i].set_title(model_name, fontsize=14)
        axes[i].set_xlabel('True Value', fontsize=12)
        axes[i].set_ylabel('Predicted Value', fontsize=12)
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.5)

        train_scatter_data = {
            'Train_True': y_train,
            'Train_Pred': y_train_pred
        }
        train_scatter_df = pd.DataFrame(train_scatter_data)
        train_scatter_csv_path = os.path.join(folder_path, f"{model_name}_train_scatter.csv")
        train_scatter_df.to_csv(train_scatter_csv_path, index=False)
        test_scatter_data = {
            'Test_True': y_test,
            'Test_Pred': y_test_pred
        }
        test_scatter_df = pd.DataFrame(test_scatter_data)
        test_scatter_csv_path = os.path.join(folder_path, f"{model_name}_test_scatter.csv")
        test_scatter_df.to_csv(test_scatter_csv_path, index=False)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.savefig(os.path.join(folder_path, "scatter_diagram.png"))
    plt.clf()

    # ----------------------
    # 6. Residual Distribution Plot
    # ----------------------
    plt.figure(figsize=(10, 6))

    for model_name, (_, _, metrics) in results.items():
        x_grid = np.array(metrics['X_Grid'])
        pdf = np.array(metrics['KDE_PDF'])
        plt.plot(x_grid, pdf, label=model_name)

    plt.title('Residual Distribution of All Models')
    plt.xlabel('Residuals')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(os.path.join(folder_path, "all_residual_distribution.png"))
    plt.clf()

    """
    # ----------------------
    # 7. FAB Data Prediction 
    # ----------------------
    test_file_path = r"fab_data.xlsx"
    test_data = pd.read_excel(test_file_path)
    test_data = remove_outliers(data)
    
    X_test_fab = test_data[['Pressure', 'Power', 'Spacing', 'Depo_time', 'SiH4', 'NH3', 'N2', 'H2']].values
    y_test_fab = test_data['rate'].values
    
    X_test_fab = scaler.transform(X_test_fab)
    
    for model_name, (best_model, _, _) in results.items():
        pred_test = best_model.predict(X_test_fab)
        r2 = r2_score(y_test_fab, pred_test)
        print(f"{model_name} - R2 on test data: {r2:.4f}")
        plt.plot(range(len(pred_test)), pred_test, color='blue', label='Predicted')
        plt.plot(range(len(pred_test)), y_test_fab, color='red', label='Actual')
        plt.title(f"{model_name} - Testing Data Prediction")
        plt.legend()
        plt.savefig(os.path.join(folder_path, f"{model_name}_Testing_Data_Prediction.png"))
        plt.clf()  
    """