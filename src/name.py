import argparse
import os
import os.path as osp
from openpyxl.reader.excel import load_workbook

def parse_args():
    parser = argparse.ArgumentParser(description="Reverse lookup names from serial numbers.")
    parser.add_argument('--projectName', help='project name')
    parser.add_argument('--interactionDatasetName', default="NPInter2", help='raw interactions dataset')
    parser.add_argument('--only_file', help='Path to the file containing serial numbers')
    args = parser.parse_args()
    return args

def read_interaction_dataset(dataset_path, dataset_name):
    lncRNA_name_index_dict = {}
    protein_name_index_dict = {}
    lncRNA_index_name_dict = {}
    protein_index_name_dict = {}
    if not osp.exists(dataset_path):
        raise FileNotFoundError("The dataset path does not exist.")
    wb = load_workbook(dataset_path)
    sheets = wb.worksheets
    sheet = sheets[0]
    rows = sheet.rows

    # 分别为 lncRNA 和蛋白质设置独立的 serial_number 计数器
    lncRNA_serial_number = 0
    protein_serial_number = 0
    flag = 0

    for row in rows:
        if flag == 0:
            flag = flag + 1
            continue
        [lncRNA_name, protein_name, label] = [col.value for col in row]
        if lncRNA_name not in lncRNA_name_index_dict:  # 新的，没创建过的lncRNA
            lncRNA_name_index_dict[lncRNA_name] = lncRNA_serial_number
            lncRNA_index_name_dict[lncRNA_serial_number] = lncRNA_name
            lncRNA_serial_number = lncRNA_serial_number + 1
        if protein_name not in protein_name_index_dict:  # 新的，没创建过的protein
            protein_name_index_dict[protein_name] = protein_serial_number
            protein_index_name_dict[protein_serial_number] = protein_name
            protein_serial_number = protein_serial_number + 1

    return lncRNA_index_name_dict, protein_index_name_dict

def reverse_lookup(only_file, lncRNA_index_name_dict, protein_index_name_dict):
    with open(only_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 2:
                serial_number = int(parts[0])
                node_type = parts[1]
                if node_type == 'lncRNA':
                    if serial_number in lncRNA_index_name_dict:
                        print(f"lncRNA序号 {serial_number} 对应的名字是: {lncRNA_index_name_dict[serial_number]}")
                    else:
                        print(f"未找到lncRNA序号 {serial_number} 对应的名字")
                elif node_type == 'protein':
                    if serial_number in protein_index_name_dict:
                        print(f"蛋白质序号 {serial_number} 对应的名字是: {protein_index_name_dict[serial_number]}")
                    else:
                        print(f"未找到蛋白质序号 {serial_number} 对应的名字")
            else:
                print(f"无效的行: {line.strip()}")

if __name__ == '__main__':
    args = parse_args()
    interaction_dataset_path = 'data/source_database_data/' + args.interactionDatasetName + '.xlsx'
    lncRNA_index_name_dict, protein_index_name_dict = read_interaction_dataset(dataset_path=interaction_dataset_path,
                                                                              dataset_name=args.interactionDatasetName)
    reverse_lookup(args.only_file, lncRNA_index_name_dict, protein_index_name_dict)