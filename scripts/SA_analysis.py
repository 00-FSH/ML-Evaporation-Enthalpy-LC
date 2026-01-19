import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import sascorer  # 本地 sascorer 文件
import pandas as pd


# 从文本文件中读取 SMILES 字符串
def read_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles_list = file.readlines()
    return [smiles.strip() for smiles in smiles_list]


# 计算 SA 值
def calculate_sa(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"
        sa_score = sascorer.calculateScore(mol)
        return sa_score, "Success"
    except Exception as e:
        return None, str(e)


# 主函数：计算文件中所有化合物的 SA 值
def main(input_file, output_file):
    smiles_list = read_smiles(input_file)
    results = []

    for smiles in smiles_list:
        sa_score, status = calculate_sa(smiles)
        results.append({'SMILES': smiles, 'SA_Score': sa_score, 'Status': status})

    # 将结果保存到 Excel 文件
    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)
    print(f"Results have been saved to {output_file}")


# 输入 SMILES 文件和输出结果文件
input_file = "D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\散热材料.txt"  # 包含 SMILES 字符串的文件路径
output_file = "D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测/SA得分.xlsx"  # 输出结果的文件路径

main(input_file, output_file)
