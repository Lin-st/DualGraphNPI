import os
# os.environ["http_proxy"] = "http://127.0.0.1:10809"  # 替换为您的V2Ray端口
# os.environ["https_proxy"] = "http://127.0.0.1:10809"
import torch
from Bio import SeqIO
from tqdm import tqdm
from transformers import AutoTokenizer, EsmModel


def create_output_dir(path):
    """创建输出目录（如果不存在）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_protein_sequences(fasta_path, strict_mode=True):
    """
    加载蛋白质序列并保留ID
    :param strict_mode: 严格模式将跳过包含非标准氨基酸的序列
    """
    sequences = []
    sequence_ids = []
    invalid_seqs = 0
    invalid_examples = []

    # 标准氨基酸字符集（20种常见氨基酸+终止符）
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY*")

    with tqdm(SeqIO.parse(fasta_path, "fasta"), desc="Loading proteins", unit="seq") as pbar:
        for record in pbar:
            seq = str(record.seq).upper()

            if strict_mode:
                if all(aa in valid_aas for aa in seq):
                    sequences.append(seq)
                    sequence_ids.append(record.id)
                else:
                    invalid_seqs += 1
                    invalid_aas = {aa for aa in seq if aa not in valid_aas}
                    invalid_examples.append((record.id, invalid_aas))
            else:
                sequences.append(seq)
                sequence_ids.append(record.id)
                if any(aa not in valid_aas for aa in seq):
                    invalid_seqs += 1
                    invalid_aas = {aa for aa in seq if aa not in valid_aas}
                    invalid_examples.append((record.id, invalid_aas))

            pbar.set_postfix({"Total": len(sequences), "Invalid": invalid_seqs})

    if invalid_examples:
        print("\nWarning: Found sequences with non-standard AAs:")
        for seq_id, bad_aas in invalid_examples[:5]:
            print(f" - {seq_id}: Contains {bad_aas}")
        if len(invalid_examples) > 5:
            print(f"(Total {len(invalid_examples)} invalid sequences, showing first 5)")

    return sequences, sequence_ids


def generate_protein_embeddings(model, tokenizer, sequences, sequence_ids, output_dir, batch_size=8):
    """生成蛋白质嵌入并保存结果"""
    device = model.device

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # Tokenization
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    # 分批处理
    embeddings = []
    total_batches = (len(sequences) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in tqdm(range(0, len(sequences), batch_size),
                      desc="Generating embeddings",
                      unit="batch",
                      total=total_batches):
            batch = {k: v[i:i + batch_size].to(device) for k, v in inputs.items()}
            outputs = model(**batch)
            embeddings.append(outputs.last_hidden_state.cpu())

    # 合并结果
    embeddings = torch.cat(embeddings, dim=0)

    # ID到嵌入的映射字典
    id_to_embedding = {seq_id: emb for seq_id, emb in zip(sequence_ids, embeddings)}
    torch.save(id_to_embedding, f"{output_dir}/protein_embeddings.pt")

    print(f"\nSaved embeddings to {output_dir}:")
    print(f"- protein_embeddings.pt (shape: {embeddings.shape})")
    print(f"- embeddings_with_ids.pt (with sequence IDs)")
    print(f"- id_to_embedding.pt (dictionary mapping)")


def main():
    # 配置路径
    input_fasta = "../data/protein_sequence/NPInter5_test/protein_sequence.fasta"
    output_dir = "../data/esm/NPInter5"

    # 检查输入文件
    if not os.path.exists(input_fasta):
        raise FileNotFoundError(f"Input file not found: {input_fasta}")

    # 加载序列
    print("=" * 50)
    sequences, sequence_ids = load_protein_sequences(input_fasta, strict_mode=False)
    if not sequences:
        raise ValueError("No valid sequences found!")
    print(f"✅ Loaded {len(sequences)} protein sequences")

    # 初始化ESM模型
    print("=" * 50)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ESM model to {device}...")

    model_name = "facebook/esm2_t33_650M_UR50D"  # 中等规模ESM2模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()

    # 生成嵌入
    print("=" * 50)
    generate_protein_embeddings(
        model,
        tokenizer,
        sequences,
        sequence_ids,
        output_dir,
        batch_size=4  # 根据GPU内存调整
    )


if __name__ == "__main__":
    main()