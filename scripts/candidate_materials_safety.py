import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import random


def load_smiles_from_text(file_path):
    """从文本文件加载SMILES字符串"""
    with open(file_path, 'r') as file:
        smiles_list = file.read().splitlines()  # 每行一个 SMILES 字符串
    return smiles_list


def calculate_danger_score(smiles):
    """根据毒性和化学稳定性计算危险性分数"""
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None  # 如果无法解析SMILES，则返回None

    # 提取一些分子描述符，作为化学稳定性分析的基础
    mw = Descriptors.MolWt(mol)  # 分子量
    logp = Descriptors.MolLogP(mol)  # LogP（脂水分配系数）
    tpsa = Descriptors.TPSA(mol)  # 极性表面积（Polar Surface Area）

    # 假设危险性评分：毒性 + 化学稳定性
    danger_score = mw / 1000  # 分子量大则可能更危险
    danger_score += logp  # LogP大说明更易通过生物膜
    danger_score -= tpsa / 100  # 极性表面积大的分子通常不易渗透细胞膜

    # 确保分数在 0 到 10 范围内
    danger_score = min(max(danger_score, 0), 10)

    return danger_score


def save_results_to_excel(smiles_list, scores, output_path):
    """将结果保存到新的Excel文件"""
    result_df = pd.DataFrame({
        'SMILES': smiles_list,
        'Danger Score': scores
    })
    result_df.to_excel(output_path, index=False, engine='openpyxl')


# 设置文件路径
input_file = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\材料24.txt'  # 请根据实际路径修改
output_file = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\毒性分数24.xlsx'  # 请根据实际路径修改

# 加载数据
smiles_data = load_smiles_from_text(input_file)

# 计算危险性分数
danger_scores = [calculate_danger_score(smiles) for smiles in smiles_data]

# 保存结果
save_results_to_excel(smiles_data, danger_scores, output_file)

print(f"危险性分数已保存到 {output_file}")
