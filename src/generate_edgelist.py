import argparse
import os
import os.path as osp
import sys
import random
import torch
from torch_geometric.data import Data

from openpyxl.reader.excel import load_workbook

sys.path.append(os.path.realpath('.'))

from src.classes import LncRNA, RNA2RNA
from src.classes import Protein
from src.classes import LncRNA_Protein_Interaction

def parse_args():
    parser = argparse.ArgumentParser(description="generate_dataset.")
    parser.add_argument('--projectName', help='project name')
    parser.add_argument('--interactionDatasetName', default="NPInter2", help='raw interactions dataset')
    parser.add_argument('--createBalanceDataset', default=1, type=int, help='Create a Balance dataset')
    #parser.add_argument('--reduce', default=0, type=int,
    #                    help='randomly reduce the source database, and also maintain one connected component')
    #parser.add_argument('--reduceRatio', default=0.5, help='reduce Ratio')
    parser.add_argument('--output', default=1, type=int, help='output dataset or not')

    args = parser.parse_args()
    return args


def read_interaction_dataset(dataset_path, dataset_name):
    interaction_list = []
    negative_interaction_list = []
    lncRNA_list = []
    protein_list = []
    lncRNA_name_index_dict = {}
    protein_name_index_dict = {}
    set_interactionKey = set()
    set_negativeInteractionKey = set()
    #列表：正负相互作用，lncRNA，蛋白质
    #字典：rna-id，蛋白质-id
    #集合：正负相互作用
    if not osp.exists(dataset_path):
        raise FileNotFoundError("The dataset path does not exist.")
    wb =load_workbook(dataset_path)
    sheets = wb.worksheets
    sheet = sheets[0]
    rows = sheet.rows

    serial_number = 0
    lncRNA_count = 0
    protein_count = 0
    flag = 0

    for row in rows:
        if flag == 0:
            flag = flag + 1
            continue
        #读出这一行的每个元素,每一行对应一个interaction实例，如果这个interaction对应的lncRNA和protein还没创建，就创建它
        #并在索引词典中加入它在lncRNA_list或者protein_list中的索引
        [lncRNA_name, protein_name, label] = [col.value for col in row]
        label = int(label)
        if lncRNA_name not in lncRNA_name_index_dict:  # 新的，没创建过的lncRNA
            temp_lncRNA = LncRNA(lncRNA_name, serial_number, 'LncRNA')
            lncRNA_list.append(temp_lncRNA)
            lncRNA_name_index_dict[lncRNA_name] = lncRNA_count
            serial_number = serial_number + 1
            lncRNA_count = lncRNA_count + 1
        else:  # 在interaction dataset中已经读到过，已经创建了对象的lncRNA，就存在lncRNA_list中
            temp_lncRNA = lncRNA_list[lncRNA_name_index_dict[lncRNA_name]]
        if protein_name not in protein_name_index_dict:  # 新的，没创建过的protein
            temp_protein = Protein(protein_name, serial_number, 'Protein')
            protein_list.append(temp_protein)
            protein_name_index_dict[protein_name] = protein_count
            serial_number = serial_number + 1
            protein_count = protein_count + 1
        else:  # 在interaction dataset中已经读到过，已经创建了对象的protein，就存在protein_list中
            temp_protein = protein_list[protein_name_index_dict[protein_name]]

        #创建新的相互作用类型
        interaction_key = (temp_lncRNA.serial_number, temp_protein.serial_number)
        temp_interaction = LncRNA_Protein_Interaction(temp_lncRNA, temp_protein, label, interaction_key)
        temp_lncRNA.interaction_list.append(temp_interaction)
        temp_protein.interaction_list.append(temp_interaction)

        #确认是正样本还是负样本
        if label == 1:
            interaction_list.append(temp_interaction)
            set_interactionKey.add(interaction_key)
        elif label == 0:
            negative_interaction_list.append(temp_interaction)
            set_negativeInteractionKey.add(interaction_key)
        else:
            print(label, 'is not supported.')
    lncRNA_Jaccard_dic = {}
    protein_Jaccard_dic = {}
    jaccard_list = []
    jaccard_number = 0
    #获取lncRNA相互作用的蛋白质集合
    for lncRNA in lncRNA_list:
        protein_set = set()
        for interaction in lncRNA.interaction_list:
            protein_set.add(interaction.protein.serial_number)
        lncRNA_Jaccard_dic[lncRNA] = protein_set

    for protein in protein_list:
        rna_set = set()
        for interaction in protein.interaction_list:
            rna_set.add(interaction.lncRNA.serial_number)
        protein_Jaccard_dic[protein] = rna_set

    #计算lncRNA两两之间的jaccard系数
    for i in range(len(lncRNA_list)-1):
        set1 = lncRNA_Jaccard_dic[lncRNA_list[i]]
        for j in range(i+1, len(lncRNA_list)):
            set2 = lncRNA_Jaccard_dic[lncRNA_list[j]]
            jaccard = len(set1.intersection(set2))/len(set1.union(set2))
            if len(set1) > 1 and len(set2) > 1 and jaccard > 0.5:
                jaccard_list.append(RNA2RNA(lncRNA_list[i], lncRNA_list[j],1, jaccard_number))
                jaccard_number = jaccard_number + 1
                #print(lncRNA_list[i].name, lncRNA_list[j].name)
    r2r_num = jaccard_number
    #计算蛋白质两两之间的jaccard系数
    for i in range(len(protein_list)-1):
        set1 = protein_Jaccard_dic[protein_list[i]]
        for j in range(i+1, len(protein_list)):
            set2 = protein_Jaccard_dic[protein_list[j]]
            jaccard = len(set1.intersection(set2))/len(set1.union(set2))
            if jaccard > 0:
                jaccard_list.append(RNA2RNA(protein_list[i], protein_list[j],1, jaccard_number))
                jaccard_number = jaccard_number + 1
                # print(protein_list[i].name, protein_list[j].name)

    print('number of lncRNA：{:d}, number of protein：{:d}, number of node：{:d}'.format(lncRNA_count, protein_count,
                                                                                      lncRNA_count + protein_count))
    print('number of interaction：{:d}'.format(len(interaction_list) + len(negative_interaction_list)))
    print('number of RNA2RNA: {:d}'.format(r2r_num))
    print('number of P2P: {:d}'.format(jaccard_number-r2r_num))
    print('number of Jaccard: {:d}'.format(len(jaccard_list)))
    return interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, \
        protein_name_index_dict, set_interactionKey, set_negativeInteractionKey, jaccard_list


def negative_interaction_generation():
    global lncRNA_list, protein_list, interaction_list, negative_interaction_list, set_interactionKey, set_negativeInteractionKey
    set_negativeInteractionKey = set()
    if len(negative_interaction_list) != 0:
        raise Exception('negative interactions exist')

    num_of_interaction = len(interaction_list)
    num_of_lncRNA = len(lncRNA_list)
    num_of_protein = len(protein_list)
    negative_interaction_count = 0
    while (negative_interaction_count < num_of_interaction):
        random_index_lncRNA = random.randint(0, num_of_lncRNA - 1)
        random_index_protein = random.randint(0, num_of_protein - 1)
        temp_lncRNA = lncRNA_list[random_index_lncRNA]
        temp_protein = protein_list[random_index_protein]
        # 检查随机选出的lncRNA和protein是不是有已知相互作用
        key_negativeInteraction = (temp_lncRNA.serial_number, temp_protein.serial_number)
        if key_negativeInteraction in set_interactionKey:
            continue
        if key_negativeInteraction in set_negativeInteractionKey:
            continue

        # 经过检查，随机选出的lncRNA和protein是可以作为负样本的
        set_negativeInteractionKey.add(key_negativeInteraction)
        temp_interaction = LncRNA_Protein_Interaction(temp_lncRNA, temp_protein, 0, key_negativeInteraction)
        negative_interaction_list.append(temp_interaction)
        temp_lncRNA.interaction_list.append(temp_interaction)
        temp_protein.interaction_list.append(temp_interaction)
        negative_interaction_count = negative_interaction_count + 1
    print('generate ', len(negative_interaction_list), ' negative samples')


def create_pyg_graph_R2R(interaction_list, lncRNA_list, protein_list, jaccard_list):
    edge_index = []
    #获取相互作用边信息
    for interaction in interaction_list:
        edge_index.append([interaction.lncRNA.serial_number, interaction.protein.serial_number])
        edge_index.append([interaction.protein.serial_number,interaction.lncRNA.serial_number])

    for r2r_interaction in jaccard_list:
        edge_index.append([r2r_interaction.lncRNA1.serial_number, r2r_interaction.lncRNA2.serial_number])
        edge_index.append([r2r_interaction.lncRNA2.serial_number, r2r_interaction.lncRNA1.serial_number])

    edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous()

    num_nodes = len(lncRNA_list) + len(protein_list)
    # 初始化节点特征矩阵，第一列全为 0，第二列用于表示节点类型
    node_features = torch.zeros((num_nodes, 2))
    # 为 RNA 节点的类型特征设置为 0
    for lncRNA in lncRNA_list:
        node_features[lncRNA.serial_number, 0] = 0
    # 为蛋白质节点的类型特征设置为 1
    for protein in protein_list:
        node_features[protein.serial_number, 0] = 1

    graph = Data(x=node_features, edge_index=edge_index)
    return graph



if __name__ == '__main__':
    args = parse_args()
    interaction_dataset_path = 'data/source_database_data/' + args.interactionDatasetName + '.xlsx'
    interaction_list, negative_interaction_list, lncRNA_list, protein_list, lncRNA_name_index_dict, protein_name_index_dict, set_interactionKey, \
        set_negativeInteractionKey, jaccard_list = read_interaction_dataset(dataset_path=interaction_dataset_path,
                                                              dataset_name=args.interactionDatasetName)
    
    # if args.createBalanceDataset == 1:
    #     negative_interaction_generation() # 生成负样本

    graph = create_pyg_graph_R2R(interaction_list, lncRNA_list, protein_list, jaccard_list)
    print(graph)