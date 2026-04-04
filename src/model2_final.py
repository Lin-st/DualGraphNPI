import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, Linear
from torch.utils.data import DataLoader, Dataset
import os
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import matthews_corrcoef

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


def evaluate_model(model, graph_jaccard, graph_blast, samples, batch_size=512):
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

            # 检查preds是否为单个元素
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)

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
    in_channels = {
        'lncRNA': 64 + 640,
        'protein': 49 + 1280
    }
    hidden_channels = 32
    out_channels = 16
    model = HeteroGNN(in_channels, hidden_channels, out_channels)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    scaler = GradScaler()

    # 对比学习权重
    contrastive_weight = 0.08

    # 加载数据
    fold = 0
    fold_dir = '../data/graph/7317more/fold_4'
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

    # 创建 DataLoader
    train_dataset = SampleDataset(train_samples)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, drop_last=False)

    # 训练模型
    model.train()
    for epoch in range(200):
        total_loss = 0
        total_cl_loss = 0
        total_bce_loss = 0

        for lncRNA_indices, protein_indices, labels in train_loader:
            lncRNA_indices = lncRNA_indices.to(device)
            protein_indices = protein_indices.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            with autocast():
                # 获取嵌入和两个视图的表示
                combined_embeddings, x_dict_jaccard_emb, x_dict_blast_emb = model(
                    x_dict_jaccard, edge_index_dict_jaccard, x_dict_blast, edge_index_dict_blast
                )

                # 获取当前batch的节点嵌入
                lncRNA_jaccard = x_dict_jaccard_emb['lncRNA'][lncRNA_indices]
                protein_jaccard = x_dict_jaccard_emb['protein'][protein_indices]
                lncRNA_blast = x_dict_blast_emb['lncRNA'][lncRNA_indices]
                protein_blast = x_dict_blast_emb['protein'][protein_indices]

                # 计算对比损失
                cl_loss_lncRNA = model.contrastive_loss(lncRNA_jaccard, lncRNA_blast, lncRNA_indices)
                cl_loss_protein = model.contrastive_loss(protein_jaccard, protein_blast, protein_indices)
                cl_loss = (cl_loss_lncRNA + cl_loss_protein) / 2

                # 计算分类损失
                lncRNA_emb = combined_embeddings['lncRNA'][lncRNA_indices]
                protein_emb = combined_embeddings['protein'][protein_indices]
                scores = model.predict(lncRNA_emb, protein_emb).squeeze()
                bce_loss = F.binary_cross_entropy_with_logits(scores, labels)

                # 组合损失
                loss = bce_loss + contrastive_weight * cl_loss

            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_cl_loss += cl_loss.item()
            total_bce_loss += bce_loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            avg_cl_loss = total_cl_loss / len(train_loader)
            avg_bce_loss = total_bce_loss / len(train_loader)
            print(f'Epoch: {epoch + 1}, Loss: {avg_loss:.4f}, CL Loss: {avg_cl_loss:.4f}, BCE Loss: {avg_bce_loss:.4f}')

    # 评估模型
    train_accuracy, train_sensitivity, train_specificity, train_precision, train_mcc = evaluate_model(
        model, graph_jaccard, graph_blast, train_samples)
    test_accuracy, test_sensitivity, test_specificity, test_precision, test_mcc = evaluate_model(
        model, graph_jaccard, graph_blast, test_samples)
    print(f'Train Accuracy: {train_accuracy:.4f}, Train Sensitivity: {train_sensitivity:.4f}, '
          f'Train Specificity: {train_specificity:.4f}, Train Precision: {train_precision:.4f}, '
          f'Train MCC: {train_mcc:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}, Test Sensitivity: {test_sensitivity:.4f}, '
          f'Test Specificity: {test_specificity:.4f}, Test Precision: {test_precision:.4f}, '
          f'Test MCC: {test_mcc:.4f}')

    # # 创建保存模型的文件夹
    # save_dir = '../saved_models'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    #
    # # 保存模型的状态字典
    # model_path = os.path.join(save_dir, 'final_model_2241.pth')
    # torch.save(model.state_dict(), model_path)
    # print(f'Model saved to {model_path}')