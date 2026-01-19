

import pandas as pd
from joblib import load
import matplotlib.pyplot as plt

# 步骤 2：读取数据
# 替换以下路径为你的xlsx文件路径
input_file_path = "D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\散热材料1.xlsx"
data_to_predict = pd.read_excel(input_file_path)

if 'Output' in data_to_predict.columns:
    data_to_predict = data_to_predict.drop(columns=['Output'])

# 步骤 3：加载模型和标准化器
# 替换以下路径为你保存模型和标准化器的路径
model_path = "D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/best_GBDT_model440.joblib"
scaler_path = "D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\scaler_GBDT440.joblib"
model = load(model_path)
scaler = load(scaler_path)

# 步骤 4：预处理数据
# 假设输入数据中只有特征列，无需预测列
X_to_predict_scaled = scaler.transform(data_to_predict)

# 步骤 5：进行预测
predictions = model.predict(X_to_predict_scaled)

# 步骤 6：保存预测结果
# 在原数据框中添加预测列
data_to_predict['Predictions'] = predictions
# 替换以下路径为你想保存预测结果的xlsx文件路径
output_file_path = "D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\散热材料1.xlsx"
data_to_predict.to_excel(output_file_path, index=False)

# 步骤 7：画图展示预测结果
plt.figure(figsize=(15, 10))
plt.plot(predictions, label='Predictions', color='blue')
plt.xlabel('Sample Index')
plt.ylabel(' Heats of Vaporization')
plt.title('Prediction Results')
plt.legend()
plt.show()

