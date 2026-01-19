# -*-coding:utf-8-*-
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from joblib import dump
import matplotlib.pyplot as plt

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

def plot_learning_curve(model, scaler, X, y, random_seed=42, save_path=None):
    """绘制学习曲线，并保存数据到 Excel"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    test_scores = []

    # 拆分训练集和测试集
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    for frac in train_sizes:
        n_train = int(len(X_train_full) * frac)
        X_train_frac = X_train_full.iloc[:n_train]
        y_train_frac = y_train_full.iloc[:n_train]

        # 数据标准化
        X_train_scaled = scaler.fit_transform(X_train_frac)
        X_test_scaled = scaler.transform(X_test)

        # 拟合模型
        model.fit(X_train_scaled, y_train_frac.ravel())

        # 计算 R²
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)

        train_scores.append(r2_score(y_train_frac, y_train_pred))
        test_scores.append(r2_score(y_test, y_test_pred))

    # 绘图
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes*100, train_scores, marker='o', label='Training R²')
    plt.plot(train_sizes*100, test_scores, marker='s', label='Test R²')
    plt.xlabel('Training Set Size (%)')
    plt.ylabel('R² Score')
    plt.title('GBDT Learning Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 保存数据到 Excel
    if save_path is not None:
        df_curve = pd.DataFrame({
            'Training Size (%)': train_sizes*100,
            'Train R2': train_scores,
            'Test R2': test_scores
        })
        df_curve.to_excel(save_path, index=False)
        print(f"学习曲线数据已保存到: {save_path}")

def main():
    file_path = r'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\去重后的数据.xlsx'
    df = load_data_from_excel(file_path)
    X, y = prepare_data(df)

    # 数据标准化器
    scaler_zscore = preprocessing.StandardScaler()
    t_size = 0.2  # 测试集比例
    random_seed = 440

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=t_size, random_state=random_seed
    )

    X_train_df = X_train.copy()
    X_test_df = X_test.copy()

    X_train_scaled = scaler_zscore.fit_transform(X_train)
    X_test_scaled = scaler_zscore.transform(X_test)

    # 使用超参数训练 GBDT
    model = GradientBoostingRegressor(
        n_estimators=242,
        learning_rate=0.09237904284326988,
        max_depth=6,
        subsample=0.5332854875209898,
        random_state=random_seed
    )
    model.fit(X_train_scaled, y_train.ravel())

    # 预测并评估
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

    # 保存模型和 scaler
    model_save_path = r'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/best_GBDT_model440.joblib'
    scaler_save_path = r'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/scaler_GBDT440.joblib'
    model_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    dump(model, model_save_path)
    dump(scaler_zscore, scaler_save_path)
    print(f"模型已保存至 {model_save_path}")
    print(f"数据标准化器已保存至 {scaler_save_path}")

    # 绘制学习曲线并保存数据
    learning_curve_path = r'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/GBDT_learning_curve_data.xlsx'
    plot_learning_curve(model, scaler_zscore, X, y, random_seed=random_seed, save_path=learning_curve_path)

if __name__ == "__main__":
    main()
