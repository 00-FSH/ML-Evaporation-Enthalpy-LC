# -*-coding:utf-8-*-

from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump


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


def main():
    file_path = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\去重后的数据.xlsx'  # 请更换为你的文件路径
    df = load_data_from_excel(file_path)
    X, y = prepare_data(df)

    # 数据标准化
    scaler_zscore = preprocessing.StandardScaler()
    t_size = 0.2  # Set the test size for splitting
    random_seed = 440

    # Split the data using the random seed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=random_seed)
    X_train_df = X_train.copy()
    X_test_df = X_test.copy()
    X_train = scaler_zscore.fit_transform(X_train)
    X_test = scaler_zscore.transform(X_test)

    # Train the GBDT with the best parameters
    model = GradientBoostingRegressor(
        n_estimators=242,
        learning_rate=0.09237904284326988,
        max_depth=6,
        subsample=0.5332854875209898,
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

    # 打印模型性能
    print(f"Train R²: {train_r2}, Test R²: {test_r2}")
    print(f"Train RMSE: {train_rmse}, Test RMSE: {test_rmse}")
    print(f"Train MAE: {train_mae}, Test MAE: {test_mae}")

    # 保存模型
    model_save_path = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/best_GBDT_model440.joblib'
    scaler_save_path = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/scaler_GBDT440.joblib'

    # If the save directory does not exist, create it
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model and scaler
    dump(model, model_save_path)
    dump(scaler_zscore, scaler_save_path)
    print(f"模型已保存至 {model_save_path}")
    print(f"数据标准化器已保存至 {scaler_save_path}")


if __name__ == "__main__":
    main()

