# -*-coding:utf-8-*-

from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

def load_data_from_excel(file_path):
    """从Excel文件加载数据"""
    df = pd.read_excel(file_path)
    return df

def prepare_data(df):
    """Prepare training data and target variable, drop rows where the target is NaN."""
    X = df.iloc[:, :-1]  # All rows, all columns except the last
    y = df.iloc[:, -1]  # All rows, only the last column

    # Drop rows where the target variable is NaN
    mask = y.notna()  # Create a mask for rows where y is not NaN
    X = X[mask]  # Apply mask to X
    y = y[mask]  # Apply mask to y

    return X, y

# 定义用于多个随机种子评估的函数
def evaluate_with_multiple_seeds(n_estimators, max_depth, learning_rate, min_child_weight, data, targets):
    results = []
    seeds = range(0, 1001, 10)  # Generate seeds from 0 to 1000 with a step of 10

    for seed in seeds:
        train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae = evaluate_rf(
            n_estimators, max_depth, learning_rate, min_child_weight, data, targets, random_seed=seed
        )
        results.append({
            "Random Seed": seed,
            "Train R²": train_r2,
            "Test R²": test_r2,
            "Train RMSE": train_rmse,
            "Test RMSE": test_rmse,
            "Train MAE": train_mae,
            "Test MAE": test_mae
        })

    # Save all performance metrics to an Excel file
    df_results = pd.DataFrame(results)
    file_name = "D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\随机种子\XGBoost_all_seed_performance_metrics_0.2.xlsx"
    df_results.to_excel(file_name, index=False)
    print(f"所有随机种子的模型性能已保存到 Excel 文件中：{file_name}")

    # Additionally, save predictions for seed 290
    evaluate_rf(n_estimators, max_depth, learning_rate, min_child_weight, data, targets, random_seed=290,
                save_predictions=True)

def evaluate_rf(n_estimators, max_depth, learning_rate, min_child_weight, data, targets, random_seed=42,
                save_predictions=False):
    scaler_zscore = preprocessing.StandardScaler()
    t_size = 0.2  # Set the test size for splitting

    # Split the data using the random seed
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=t_size, random_state=random_seed)
    X_train = scaler_zscore.fit_transform(X_train)
    X_test = scaler_zscore.transform(X_test)

    # Train the XGBRegressor with the best parameters
    model = XGBRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        learning_rate=learning_rate,
        min_child_weight=int(min_child_weight),
        random_state=random_seed
    )
    model.fit(X_train, y_train.ravel())

    # Predict for both train and test sets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the model using R² score, RMSE, and MAE for both train and test sets
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # Save predictions and true values if needed
    if save_predictions:
        save_predictions_to_excel(y_train.values, y_pred_train, y_test.values, y_pred_test, random_seed)

    return train_r2, test_r2, train_rmse, test_rmse, train_mae, test_mae

def save_predictions_to_excel(y_train, y_pred_train, y_test, y_pred_test, random_seed):
    # Prepare data for saving
    df_train = pd.DataFrame({
        "True Train Values": y_train.flatten(),  # Ensure it's a flat array
        "Predicted Train Values": y_pred_train.flatten()
    })
    df_test = pd.DataFrame({
        "True Test Values": y_test.flatten(),  # Ensure it's a flat array
        "Predicted Test Values": y_pred_test.flatten()
    })

    # Save to Excel
    file_name = f"D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\随机种子\XGBoost_predictions_seed_290_0.2.xlsx"
    with pd.ExcelWriter(file_name) as writer:
        df_train.to_excel(writer, sheet_name="Train Data", index=False)
        df_test.to_excel(writer, sheet_name="Test Data", index=False)

    print(f"随机种子为 {random_seed} 时的训练集和测试集的预测值已保存到 Excel 文件中：{file_name}")

def optimize_rf(data, targets):
    """Apply Bayesian Optimization to XGBRegressor parameters."""

    def rfc_single_seed(n_estimators, max_depth, learning_rate, min_child_weight):
        return evaluate_rf(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            min_child_weight=int(min_child_weight),
            data=data,
            targets=targets
        )[1]  # We only need the Test R² score during optimization

    params = {
        "n_estimators": (50, 500),
        "max_depth": (2, 10),
        "learning_rate": (0.01, 0.2),
        "min_child_weight": (1, 10)
    }

    optimizer = BayesianOptimization(
        f=rfc_single_seed,
        pbounds=params,
        random_state=1,
        verbose=2
    )
    logger = JSONLogger(path="D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\随机种子\XGBoost_rank12_test.json")
    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    start = time.time()
    optimizer.maximize(init_points=5, n_iter=55)
    with open('D:\pycharm\机器学习\pythonProject\Master3\ML\json_smile\XGBoost_rank12_test.json', 'a', encoding='utf-8', errors='replace') as f:
        f.write(str(optimizer.max))
    print(optimizer.max)
    end = time.time()
    runTime = end - start
    print("运行时间：", runTime)

    # Get the best parameters
    best_params = optimizer.max['params']
    n_estimators = int(best_params['n_estimators'])
    max_depth = int(best_params['max_depth'])
    learning_rate = best_params['learning_rate']
    min_child_weight = int(best_params['min_child_weight'])

    # Evaluate with multiple seeds and save all performance metrics
    evaluate_with_multiple_seeds(n_estimators, max_depth, learning_rate, min_child_weight, data, targets)

def main():
    file_path = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\去重后的数据.xlsx'  # 请更换为你的文件路径
    df = load_data_from_excel(file_path)
    X, y = prepare_data(df)
    optimize_rf(X, y)

if __name__ == "__main__":
    main()
