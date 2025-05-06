import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from torch.utils.data import DataLoader, Dataset
import os
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import matthews_corrcoef

# 检查GPU可用性并设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample.lncRNA.serial_number, sample.protein.serial_number, sample.y


class HeteroGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()

        # 确保out_channels是hidden_channels的一半，以便维度匹配
        assert out_channels == hidden_channels // 2, "out_channels should be half of hidden_channels for dimension matching"

        # 1. 改进的图卷积层
        self.jaccard_convs = torch.nn.ModuleList([
            HeteroConv({
                ('lncRNA', 'interacts_with', 'protein'): SAGEConv((in_channels['lncRNA'], in_channels['protein']),
                                                                  hidden_channels),
                ('lncRNA', 'jaccard_related', 'lncRNA'): GCNConv(in_channels['lncRNA'], hidden_channels),
                ('protein', 'jaccard_related', 'protein'): GCNConv(in_channels['protein'], hidden_channels)
            }, aggr='mean'),
            HeteroConv({
                ('lncRNA', 'interacts_with', 'protein'): SAGEConv((hidden_channels, hidden_channels), out_channels),
                ('lncRNA', 'jaccard_related', 'lncRNA'): GCNConv(hidden_channels, out_channels),
                ('protein', 'jaccard_related', 'protein'): GCNConv(hidden_channels, out_channels)
            }, aggr='mean')
        ])

        self.blast_convs = torch.nn.ModuleList([
            HeteroConv({
                ('lncRNA', 'interacts_with', 'protein'): SAGEConv((in_channels['lncRNA'], in_channels['protein']),
                                                                  hidden_channels),
                ('lncRNA', 'blast_related', 'lncRNA'): GCNConv(in_channels['lncRNA'], hidden_channels),
                ('protein', 'blast_related', 'protein'): GCNConv(in_channels['protein'], hidden_channels)
            }, aggr='mean'),
            HeteroConv({
                ('lncRNA', 'interacts_with', 'protein'): SAGEConv((hidden_channels, hidden_channels), out_channels),
                ('lncRNA', 'blast_related', 'lncRNA'): GCNConv(hidden_channels, out_channels),
                ('protein', 'blast_related', 'protein'): GCNConv(hidden_channels, out_channels)
            }, aggr='mean')
        ])

        # 2. 预测层
        self.bn = torch.nn.BatchNorm1d(2 * out_channels)
        self.dropout = torch.nn.Dropout(0.3)
        self.lin1 = Linear(2 * out_channels, out_channels)
        self.lin2 = Linear(out_channels, 1)

        # 3. 修改残差连接 - 确保维度匹配
        self.residual_lncRNA = Linear(in_channels['lncRNA'], out_channels)
        self.residual_protein = Linear(in_channels['protein'], out_channels)

    def forward(self, x_dict_jaccard, edge_index_dict_jaccard, x_dict_blast, edge_index_dict_blast):
        # 保存原始特征用于残差连接
        x_lncRNA_orig = x_dict_jaccard['lncRNA']
        x_protein_orig = x_dict_jaccard['protein']

        # 处理jaccard图
        for conv in self.jaccard_convs:
            x_dict_jaccard = conv(x_dict_jaccard, edge_index_dict_jaccard)
            x_dict_jaccard = {key: F.leaky_relu(x, negative_slope=0.2) for key, x in x_dict_jaccard.items()}

        # 处理blast图
        for conv in self.blast_convs:
            x_dict_blast = conv(x_dict_blast, edge_index_dict_blast)
            x_dict_blast = {key: F.leaky_relu(x, negative_slope=0.2) for key, x in x_dict_blast.items()}

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

        return combined_embeddings

    def predict(self, lncRNA_emb, protein_emb):
        # 预测流程
        combined = torch.cat([lncRNA_emb, protein_emb], dim=1)
        combined = self.bn(combined)
        combined = self.dropout(combined)
        combined = F.leaky_relu(self.lin1(combined), negative_slope=0.2)
        return self.lin2(combined)


def evaluate_model(model, graph_jaccard, graph_blast, samples, batch_size=512):
    model.eval()
    dataset = SampleDataset(samples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # 预加载图数据到GPU
    x_dict_jaccard = {
        'lncRNA': graph_jaccard['lncRNA'].x.to(device),
        'protein': graph_jaccard['protein'].x.to(device)
    }
    edge_index_dict_jaccard = {
        ('lncRNA', 'interacts_with', 'protein'): graph_jaccard['lncRNA', 'interacts_with', 'protein'].edge_index.to(
            device),
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
        ('lncRNA', 'interacts_with', 'protein'): graph_blast['lncRNA', 'interacts_with', 'protein'].edge_index.to(
            device),
        ('lncRNA', 'blast_related', 'lncRNA'): graph_blast['lncRNA', 'blast_related', 'lncRNA'].edge_index.to(device),
        ('protein', 'blast_related', 'protein'): graph_blast['protein', 'blast_related', 'protein'].edge_index.to(
            device)
    }

    with torch.no_grad():
        combined_embeddings = model(x_dict_jaccard, edge_index_dict_jaccard, x_dict_blast, edge_index_dict_blast)

        correct = 0
        total = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        all_labels = []
        all_preds = []

        for lncRNA_indices, protein_indices, labels in loader:
            lncRNA_indices = lncRNA_indices.to(device)
            protein_indices = protein_indices.to(device)
            labels = labels.to(device)

            lncRNA_emb = combined_embeddings['lncRNA'][lncRNA_indices]
            protein_emb = combined_embeddings['protein'][protein_indices]
            scores = model.predict(lncRNA_emb, protein_emb).squeeze()
            preds = (torch.sigmoid(scores) > 0.5).float()

            correct += (preds == labels).sum().item()
            total += len(labels)

            true_positives += ((preds == 1) & (labels == 1)).sum().item()
            false_positives += ((preds == 1) & (labels == 0)).sum().item()
            true_negatives += ((preds == 0) & (labels == 0)).sum().item()
            false_negatives += ((preds == 0) & (labels == 1)).sum().item()

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    accuracy = correct / total
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    mcc = matthews_corrcoef(all_labels, all_preds)

    return accuracy, sensitivity, specificity, precision, mcc


if __name__ == "__main__":
    # 假设 RNA 嵌入向量是 64 维，蛋白质是 49 维
    in_channels = {
        'lncRNA': 64,
        'protein': 49
    }
    hidden_channels = 32
    out_channels = 16
    model = HeteroGNN(in_channels, hidden_channels, out_channels)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scaler = GradScaler()  # 用于混合精度训练

    # 加载数据
    fold = 0
    fold_dir = '../data/graph/lst/fold_4'  # 替换为实际的 project_name
    train_samples = torch.load(os.path.join(fold_dir, 'train_samples.pt'))
    test_samples = torch.load(os.path.join(fold_dir, 'test_samples.pt'))
    graph_jaccard = torch.load(os.path.join(fold_dir, 'subgraph_jaccard.pt'))
    graph_blast = torch.load(os.path.join(fold_dir, 'subgraph_blast.pt'))

    # 预加载图数据到GPU
    x_dict_jaccard = {
        'lncRNA': graph_jaccard['lncRNA'].x.to(device),
        'protein': graph_jaccard['protein'].x.to(device)
    }
    edge_index_dict_jaccard = {
        ('lncRNA', 'interacts_with', 'protein'): graph_jaccard['lncRNA', 'interacts_with', 'protein'].edge_index.to(
            device),
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
        ('lncRNA', 'interacts_with', 'protein'): graph_blast['lncRNA', 'interacts_with', 'protein'].edge_index.to(
            device),
        ('lncRNA', 'blast_related', 'lncRNA'): graph_blast['lncRNA', 'blast_related', 'lncRNA'].edge_index.to(device),
        ('protein', 'blast_related', 'protein'): graph_blast['protein', 'blast_related', 'protein'].edge_index.to(
            device)
    }

    # 创建 DataLoader
    train_dataset = SampleDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # 训练模型
    model.train()
    for epoch in range(200):
        total_loss = 0
        for lncRNA_indices, protein_indices, labels in train_loader:
            lncRNA_indices = lncRNA_indices.to(device)
            protein_indices = protein_indices.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            # 使用混合精度训练
            with autocast():
                combined_embeddings = model(x_dict_jaccard, edge_index_dict_jaccard, x_dict_blast,
                                            edge_index_dict_blast)

                lncRNA_emb = combined_embeddings['lncRNA'][lncRNA_indices]
                protein_emb = combined_embeddings['protein'][protein_indices]
                combined_emb = torch.cat([lncRNA_emb, protein_emb], dim=1)
                scores = model.predict(lncRNA_emb, protein_emb).squeeze()

                loss = F.binary_cross_entropy_with_logits(scores, labels)

            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}')

    # 评估模型
    train_accuracy, train_sensitivity, train_specificity, train_precision, train_mcc = evaluate_model(model, graph_jaccard, graph_blast, train_samples)
    test_accuracy, test_sensitivity, test_specificity, test_precision, test_mcc = evaluate_model(model, graph_jaccard, graph_blast, test_samples)
    print(f'Train Accuracy: {train_accuracy:.4f}, Train Sensitivity: {train_sensitivity:.4f}, Train Specificity: {train_specificity:.4f}, Train Precision: {train_precision:.4f}, Train MCC: {train_mcc:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}, Test Sensitivity: {test_sensitivity:.4f}, Test Specificity: {test_specificity:.4f}, Test Precision: {test_precision:.4f}, Test MCC: {test_mcc:.4f}')