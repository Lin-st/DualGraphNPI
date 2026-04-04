import os
# os.environ["http_proxy"] = "http://127.0.0.1:10809"  # 替换为您的V2Ray端口
# os.environ["https_proxy"] = "http://127.0.0.1:10809"
import torch
from Bio import SeqIO
from tqdm import tqdm  # 进度条库
from multimolecule import RnaTokenizer, RnaFmModel


def create_output_dir(path):
    """创建输出目录（如果不存在）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_sequences(fasta_path, strict_mode=False):
    sequences = []
    sequence_ids = []  # 新增：保存序列ID
    invalid_seqs = 0
    invalid_examples = []

    with tqdm(SeqIO.parse(fasta_path, "fasta"), desc="加载序列", unit="seq") as pbar:
        for record in pbar:
            seq = str(record.seq).upper().replace("T", "U")

            if strict_mode:
                if all(c in "ACGU" for c in seq):
                    sequences.append(seq)
                    sequence_ids.append(record.id)  # 保存有效序列ID
                else:
                    invalid_seqs += 1
                    invalid_chars = {c for c in seq if c not in "ACGU"}
                    invalid_examples.append((record.id, invalid_chars))
            else:
                sequences.append(seq)
                sequence_ids.append(record.id)  # 保存所有序列ID
                if any(c not in "ACGU" for c in seq):
                    invalid_seqs += 1
                    invalid_chars = {c for c in seq if c not in "ACGU"}
                    invalid_examples.append((record.id, invalid_chars))

            pbar.set_postfix({"总序列": len(sequences), "含非法字符序列": invalid_seqs})

    if invalid_examples:
        print("\n警告：发现以下序列包含非标准字符（仍将被保留）:")
        for seq_id, bad_chars in invalid_examples[:5]:
            print(f" - {seq_id}: 包含非标准字符 {bad_chars}")
        if len(invalid_examples) > 5:
            print(f"（共 {len(invalid_examples)} 条序列含非标准字符，仅显示前5条）")

    return sequences, sequence_ids  # 返回序列和对应的ID列表


def generate_embeddings(model, tokenizer, sequences, sequence_ids, output_path, batch_size=32):
    """批量生成嵌入并保存ID-向量映射"""
    # Tokenize所有序列
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    # 分批次处理
    embeddings = []
    device = model.device
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size),
                      desc="生成嵌入",
                      unit="batch",
                      total=total_batches):
            batch = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
            outputs = model(**batch, output_hidden_states=True)
            embeddings.append(outputs.hidden_states[-1].cpu())

    # 合并结果
    embeddings = torch.cat(embeddings, dim=0)

    # 创建ID-向量映射字典
    id_to_embedding = {
        seq_id: embedding
        for seq_id, embedding in zip(sequence_ids, embeddings)
    }

    # 保存两个文件：原始嵌入和映射字典
    torch.save(id_to_embedding, output_path)

    print(f"\nID-嵌入映射字典已保存至 {output_path} (形状: {embeddings.shape})")


def main():
    # 配置路径
    input_fasta = "../data/lncRNA_sequence/NPInter5_test/lncRNA_sequence.fasta"
    output_path = "../data/RNA-FM/NPInter5/lncRNA_embeddings.pt"

    # 检查输入文件
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"输入文件不存在: {input_fasta}")

    # 创建输出目录
    create_output_dir(output_path)

    # 加载序列和ID
    print("=" * 50)
    sequences, sequence_ids = load_sequences(input_fasta, strict_mode=False)
    if not sequences:
        raise ValueError("未找到有效序列！请检查FASTA文件内容。")
    print(f"✅ 共加载 {len(sequences)} 条序列")

    # 初始化模型
    print("=" * 50)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"正在加载RNA-FM模型到 {device}...")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/rnafm")
    model = RnaFmModel.from_pretrained("multimolecule/rnafm").to(device)
    model.eval()

    # 生成嵌入和映射字典
    print("=" * 50)
    generate_embeddings(model, tokenizer, sequences, sequence_ids, output_path, batch_size=16)


if __name__ == "__main__":
    main()