import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  # 导入 GBDT 模型
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# 1. 读取Excel数据
file_path = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/2474SHAP前20.xlsx'  # 替换为你的Excel文件路径
data = pd.read_excel(file_path)

# 2. 分离特征和目标
X = data.iloc[:, :-1]  # 特征为除最后一列之外的所有列
y = data.iloc[:, -1]   # 目标特性为最后一列

# 3. 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 使用 GBDT 模型进行训练
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)  # GBDT 模型参数
model.fit(X_train, y_train)

# 5. 计算 permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# 6. 获取特征名称和重要性
features = X.columns
importances = result.importances_mean

# 7. 按重要性对特征进行排序
sorted_idx = importances.argsort()

# 8. 打印重要性结果
print("Permutation Importance Results:")
for i in sorted_idx:
    print(f"{features[i]}: {importances[i]:.4f}")

# 9. 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx], importances[sorted_idx], color='skyblue')
plt.xlabel('Permutation Importance')
plt.title('Feature Importance based on Permutation')
plt.tight_layout()
plt.show()
