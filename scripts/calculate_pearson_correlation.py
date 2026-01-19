import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


# 读取包含描述符和目标特性的 DataFrame
def load_data(descriptor_file):
    # 检查文件路径是否存在
    if not os.path.exists(descriptor_file):
        raise ValueError(f"文件路径无效: {descriptor_file}")

    # 读取 Excel 文件
    try:
        df = pd.read_excel(descriptor_file)
    except Exception as e:
        raise ValueError(f"无法读取文件: {descriptor_file}. 错误: {e}")

    return df


# 计算相关性矩阵，并筛选与目标特性相关性高的描述符
def find_correlated_descriptors(df, target_column, top_n=15):
    # 只保留数值型列（去掉例如 SMILES 等非数值列）
    df_numeric = df.select_dtypes(include=[float, int])

    # 检查目标特性列是否在数值型数据中
    if target_column not in df_numeric.columns:
        raise ValueError(f"目标列 '{target_column}' 不是数值型数据，不能进行相关性分析。")

    # 计算描述符和目标特性之间的皮尔逊相关系数
    correlation_matrix = df_numeric.corr()

    # 取出与目标特性的相关性并排序
    target_corr = correlation_matrix[target_column].drop(target_column).abs().sort_values(ascending=False)

    # 筛选出相关性最高的前 top_n 个描述符
    top_descriptors = target_corr.head(top_n)

    return top_descriptors


# 绘制热力图并保存
def plot_correlation_heatmap(correlation_df, output_image):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_df.to_frame(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation with Latent Heat of Evaporation')
    plt.tight_layout()

    # 保存热力图为图像文件
    plt.savefig(output_image)
    print(f"相关性热力图已保存到 {output_image}")
    plt.show()


# 主函数
def main(descriptor_file, target_column, output_image, top_n=15):
    # 加载数据
    df = load_data(descriptor_file)

    # 查找相关性最高的描述符
    top_correlated_descriptors = find_correlated_descriptors(df, target_column, top_n)

    # 打印相关性最高的前 15 个描述符
    print("相关性最高的前 15 个描述符：")
    print(top_correlated_descriptors)

    # 绘制并保存相关性热力图
    plot_correlation_heatmap(top_correlated_descriptors, output_image)


# 使用示例
descriptor_file = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/2474SHAP前20.xlsx'  # 包含描述符和目标特性的文件
target_column = 'Output'  # 蒸发焓特性的列名
output_image = 'correlation_heatmap.png'  # 输出热力图的图像文件
top_n = 15  # 选择前 15 个相关性最高的描述符

main(descriptor_file, target_column, output_image, top_n)
