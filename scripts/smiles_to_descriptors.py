import pandas as pd
from rdkit import Chem
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors


# 读取 SMILES 文件
def read_smiles(file_path):
    with open(file_path, 'r') as file:
        smiles_list = file.readlines()
    # 去除每行的换行符
    smiles_list = [smiles.strip() for smiles in smiles_list]
    return smiles_list


# 计算分子描述符
def calculate_descriptors(smiles_list):
    descriptor_names = [desc_name[0] for desc_name in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    descriptor_data = []

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            descriptors = calculator.CalcDescriptors(mol)
            descriptor_data.append([smiles] + list(descriptors))
        else:
            print(f"SMILES {smiles} 无法解析，跳过。")
            descriptor_data.append([smiles] + [None] * len(descriptor_names))

    return pd.DataFrame(descriptor_data, columns=['SMILES'] + descriptor_names)


# 将结果保存为 Excel 文件
def save_to_excel(df, output_file):
    df.to_excel(output_file, index=False)


# 主函数
def main(smiles_file, output_file):
    smiles_list = read_smiles(smiles_file)
    descriptors_df = calculate_descriptors(smiles_list)
    save_to_excel(descriptors_df, output_file)
    print(f"描述符已保存为 {output_file}")


# 使用示例
smiles_file = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\散热材料1.txt'  # 输入SMILES的文件路径
output_file = 'D:\pycharm\机器学习\pythonProject\Master3\ML\SMILE\数据预测\散热材料1.xlsx'  # 输出Excel文件的路径
main(smiles_file, output_file)
