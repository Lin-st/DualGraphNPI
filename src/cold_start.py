import argparse

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from torch.utils.data import DataLoader, Dataset
import os
from sklearn.metrics import matthews_corrcoef, roc_auc_score, auc, precision_recall_curve
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained model on test dataset")
    parser.add_argument('--trained_model', required=True, help='Path to trained model .pth file')
    parser.add_argument('--projectName', required=True, help='Database name for test data (e.g., NPInter5_test)')
    return parser.parse_args()

class SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample.lncRNA.name, sample.protein.name, sample.lncRNA.serial_number, sample.protein.serial_number, sample.y


class HeteroGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        assert out_channels == hidden_channels // 2, "out_channels should be half of hidden_channels for dimension matching"

        # 1. 图卷积层保持不变
        self.jaccard_convs = torch.nn.ModuleList([
            HeteroConv({
                ('lncRNA', 'jaccard_related', 'lncRNA'): GCNConv(in_channels['lncRNA'], hidden_channels),
                ('protein', 'jaccard_related', 'protein'): GCNConv(in_channels['protein'], hidden_channels)
            }, aggr='mean'),
            HeteroConv({
                ('lncRNA', 'jaccard_related', 'lncRNA'): GCNConv(hidden_channels, out_channels),
                ('protein', 'jaccard_related', 'protein'): GCNConv(hidden_channels, out_channels)
            }, aggr='mean')
        ])

        self.blast_convs = torch.nn.ModuleList([
            HeteroConv({
                ('lncRNA', 'blast_related', 'lncRNA'): GCNConv(in_channels['lncRNA'], hidden_channels),
                ('protein', 'blast_related', 'protein'): GCNConv(in_channels['protein'], hidden_channels)
            }, aggr='mean'),
            HeteroConv({
                ('lncRNA', 'blast_related', 'lncRNA'): GCNConv(hidden_channels, out_channels),
                ('protein', 'blast_related', 'protein'): GCNConv(hidden_channels, out_channels)
            }, aggr='mean')
        ])

        # 2. 预测层
        self.bn = torch.nn.BatchNorm1d(2 * out_channels)
        self.dropout = torch.nn.Dropout(0.3)
        self.lin1 = Linear(2 * out_channels, out_channels)
        self.lin2 = Linear(out_channels, 1)

        # 3. 残差连接
        self.residual_lncRNA = Linear(in_channels['lncRNA'], out_channels)
        self.residual_protein = Linear(in_channels['protein'], out_channels)

        # 4. 添加投影头用于对比学习
        self.projection_head = torch.nn.Sequential(
            Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            Linear(out_channels, out_channels)
        )

    def forward(self, x_dict_jaccard, edge_index_dict_jaccard, x_dict_blast, edge_index_dict_blast):
        # 保存原始特征用于残差连接
        x_lncRNA_orig = x_dict_jaccard['lncRNA']
        x_protein_orig = x_dict_jaccard['protein']

        # 处理jaccard图
        x_dict_jaccard_list = []
        for conv in self.jaccard_convs:
            x_dict_jaccard = conv(x_dict_jaccard, edge_index_dict_jaccard)
            x_dict_jaccard = {key: F.leaky_relu(x, negative_slope=0.2) for key, x in x_dict_jaccard.items()}
            x_dict_jaccard_list.append(x_dict_jaccard)

        # 处理blast图
        x_dict_blast_list = []
        for conv in self.blast_convs:
            x_dict_blast = conv(x_dict_blast, edge_index_dict_blast)
            x_dict_blast = {key: F.leaky_relu(x, negative_slope=0.2) for key, x in x_dict_blast.items()}
            x_dict_blast_list.append(x_dict_blast)

        # 获取最后一层的嵌入
        x_dict_jaccard = x_dict_jaccard_list[-1]
        x_dict_blast = x_dict_blast_list[-1]

        # 拼接特征
        combined_embeddings = {}
        for node_type in x_dict_jaccard.keys():
            # 不再拼接两个图的特征，而是取平均
            combined = (x_dict_jaccard[node_type] + x_dict_blast[node_type]) / 2

            # 添加残差连接
            if node_type == 'lncRNA':
                residual = self.residual_lncRNA(x_lncRNA_orig)
            else:
                residual = self.residual_protein(x_protein_orig)

            combined_embeddings[node_type] = combined + residual

        return combined_embeddings, x_dict_jaccard, x_dict_blast

    def project(self, z):
        return self.projection_head(z)

    def predict(self, lncRNA_emb, protein_emb):
        combined = torch.cat([lncRNA_emb, protein_emb], dim=1)
        combined = self.bn(combined)
        combined = self.dropout(combined)
        combined = F.leaky_relu(self.lin1(combined), negative_slope=0.2)
        return self.lin2(combined)

    def contrastive_loss(self, z1, z2, indices):
        """计算视图间的对比损失"""
        # 投影到对比学习空间
        z1 = self.project(z1)
        z2 = self.project(z2)

        # 归一化
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z1, z2.T) / self.temperature

        # 正样本是相同节点的不同视图
        pos_sim = torch.diag(sim_matrix)

        # 对比损失
        numerator = torch.exp(pos_sim)
        denominator = torch.sum(torch.exp(sim_matrix), dim=1)
        loss = -torch.log(numerator / denominator)

        return loss.mean()

def evaluate_model(model, graph_jaccard, graph_blast, samples, batch_size=512, top_n=5):
    model.eval()
    dataset = SampleDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 预加载图数据到GPU
    x_dict_jaccard = {
        'lncRNA': graph_jaccard['lncRNA'].x.to(device),
        'protein': graph_jaccard['protein'].x.to(device)
    }
    edge_index_dict_jaccard = {
        ('lncRNA', 'jaccard_related', 'lncRNA'): graph_jaccard['lncRNA', 'jaccard_related', 'lncRNA'].edge_index.to(
            device),
        ('protein', 'jaccard_related', 'protein'): graph_jaccard['protein', 'jaccard_related', 'protein'].edge_index.to(
            device)
    }
    x_dict_blast = {
        'lncRNA': graph_blast['lncRNA'].x.to(device),
        'protein': graph_blast['protein'].x.to(device)
    }
    edge_index_dict_blast = {
        ('lncRNA', 'blast_related', 'lncRNA'): graph_blast['lncRNA', 'blast_related', 'lncRNA'].edge_index.to(device),
        ('protein', 'blast_related', 'protein'): graph_blast['protein', 'blast_related', 'protein'].edge_index.to(
            device)
    }

    with torch.no_grad():
        combined_embeddings, _, _ = model(x_dict_jaccard, edge_index_dict_jaccard, x_dict_blast, edge_index_dict_blast)

        # 收集所有预测结果和标签用于Top-N计算
        all_scores = []
        all_labels = []
        all_lncRNA_names = []
        all_protein_names = []

        for lncRNA_names, protein_names, lncRNA_indices, protein_indices, labels in loader:
            lncRNA_indices = lncRNA_indices.to(device)
            protein_indices = protein_indices.to(device)
            labels = labels.to(device)

            lncRNA_emb = combined_embeddings['lncRNA'][lncRNA_indices]
            protein_emb = combined_embeddings['protein'][protein_indices]
            scores = model.predict(lncRNA_emb, protein_emb).squeeze()
            # 将 scores 转换为概率
            scores = torch.sigmoid(scores)

            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_lncRNA_names.extend(lncRNA_names)
            all_protein_names.extend(protein_names)

        # 转换为numpy数组
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)

        # 计算Top-N精度
        top_n_accuracy = calculate_top_n_precision(all_scores, all_labels,top_n)

        # 原始评估指标计算
        preds = (all_scores > 0.5).astype(float)
        correct = (preds == all_labels).sum()
        total = len(all_labels)

        true_positives = ((preds == 1) & (all_labels == 1)).sum()
        false_positives = ((preds == 1) & (all_labels == 0)).sum()
        true_negatives = ((preds == 0) & (all_labels == 0)).sum()
        false_negatives = ((preds == 0) & (all_labels == 1)).sum()

        accuracy = correct / total
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
        mcc = matthews_corrcoef(all_labels, preds)

        # 计算AUROC
        auroc = roc_auc_score(all_labels, all_scores)

        # 计算AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_scores)
        auprc = auc(recall_curve, precision_curve)

        # 找出前五十个置信度最高且真实标签为0的组合
        sorted_indices = np.argsort(all_scores)[::-1]
        top_fifty_indices = []
        count = 0
        for idx in sorted_indices:
            if all_labels[idx] == 1 and all_protein_names[idx] == 'A0A090N8E9':
                top_fifty_indices.append(idx)
                count += 1
                if count == 50:
                    break

        top_fifty_combinations = [(all_lncRNA_names[i], all_protein_names[i], all_scores[i]) for i in top_fifty_indices]

    return accuracy, sensitivity, specificity, precision, mcc, top_n_accuracy, auroc, auprc, top_fifty_combinations


def calculate_top_n_precision(scores, labels, top_n):
    """计算全局Top-N精度（Precision@N）"""
    # 将所有样本按预测得分降序排序
    sorted_indices = np.argsort(scores)[::-1]
    top_n_indices = sorted_indices[:top_n]

    # 计算前N个中的正样本比例
    top_n_labels = labels[top_n_indices]
    precision = np.sum(top_n_labels == 1) / top_n
    return precision


if __name__ == "__main__":
    args = parse_args()
    in_channels = {
        'lncRNA': 64 + 640,
        'protein': 49 + 1280
    }
    hidden_channels = 32
    out_channels = 16
    model = HeteroGNN(in_channels, hidden_channels, out_channels)
    model = model.to(device)

    # 加载模型
    model_path = 'saved_models/final_model_' + args.trained_model +'.pth'
    model.load_state_dict(torch.load(model_path))

    # 加载测试集数据
    test_dir = 'data/graph/' + args.projectName
    test_samples = torch.load(os.path.join(test_dir, 'test_samples.pt'))
    graph_jaccard = torch.load(os.path.join(test_dir, 'subgraph_jaccard.pt'))
    graph_blast = torch.load(os.path.join(test_dir, 'subgraph_blast.pt'))
    N = 2200
    # 评估模型，增加Top-n精度计算
    test_accuracy, test_sensitivity, test_specificity, test_precision, test_mcc, test_top5, test_auroc, test_auprc, top_fifty_combinations = evaluate_model(
        model, graph_jaccard, graph_blast, test_samples, top_n=N)
    print(f'Test Accuracy: {test_accuracy:.4f}, Test Sensitivity: {test_sensitivity:.4f}, '
          f'Test Specificity: {test_specificity:.4f}, Test Precision: {test_precision:.4f}, '
          f'Test MCC: {test_mcc:.4f},\nTest Top-{N} Precision: {test_top5:.4f}, '
          f'Test AUROC: {test_auroc:.4f}, Test AUPRC: {test_auprc:.4f}')

    # print("\nTop fifty combinations with highest confidence and true label 1:")
    # for lncRNA_name, protein_name, score in top_fifty_combinations:
    #     print(f"{lncRNA_name}-{protein_name}: {score:.4f}")