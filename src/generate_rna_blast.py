import subprocess
import os


def create_blast_database(fasta_file):
    """
    创建 BLAST 数据库
    """
    makeblastdb_cmd = f"makeblastdb -in {fasta_file} -dbtype nucl -out rna_db"
    try:
        subprocess.run(makeblastdb_cmd, shell=True, check=True)
        print("BLAST 数据库创建成功")
    except subprocess.CalledProcessError as e:
        print(f"创建 BLAST 数据库时出错: {e}")


def perform_blast(query_file, db_name, evalue_threshold=1e-5, perc_identity_threshold=90):
    """
    执行 BLAST 搜索，并设置相似性阈值
    """
    blastn_cline = f"blastn -query {query_file} -db {db_name} -out blast_results.txt -outfmt 6 -evalue {evalue_threshold} -perc_identity {perc_identity_threshold} -num_threads 2"
    try:
        subprocess.run(blastn_cline, shell=True, check=True)
        print("BLAST 搜索完成")
    except subprocess.CalledProcessError as e:
        print(f"执行 BLAST 搜索时出错: {e}")


def get_all_hits(blast_results_file, query_id):
    """
    从 BLAST 结果中获取所有符合条件的匹配序列 ID（排除自身）
    """
    hits = []
    try:
        with open(blast_results_file) as f:
            lines = f.readlines()
            for line in lines:
                parts = line.split('\t')
                if parts[0] == query_id and parts[1] != query_id:
                    hits.append(parts[1])
        if not hits:
            print(f"未找到 {query_id} 的匹配结果（排除自身）")
        return hits
    except FileNotFoundError:
        print("BLAST 结果文件未找到")
        return []


def extract_sequence(fasta_file, seq_id):
    """
    从 FASTA 文件中提取指定 ID 的序列，处理序列换行问题
    """
    current_id = None
    current_seq = ""
    collecting_seq = False
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                if collecting_seq and current_id == seq_id:
                    return current_seq
                current_id = line.strip()[1:]
                collecting_seq = (current_id == seq_id)
                current_seq = ""
            else:
                if collecting_seq:
                    current_seq += line.strip()
    if collecting_seq and current_id == seq_id:
        return current_seq
    return None


def create_rna_pair_list(query_id, all_hit_ids, output_file):
    """
    将查询 ID 和匹配 ID 按指定格式写入列表文件
    """
    with open(output_file, "a", encoding='utf-8') as outfile:
        outfile.write(f"查询 ID: {query_id}\n")
        for hit_id in all_hit_ids:
            outfile.write(f"匹配 ID: {hit_id}\n")
        outfile.write("\n")


if __name__ == "__main__":
    # fasta_file = "data/lncRNA_sequence/NPInter2/lncRNA_sequence.fasta"  # 替换为你的 RNA FASTA 文件路径
    # output_file = "rna_pair_list.txt"
    fasta_file = "../data/lncRNA_sequence/RPI369/lncRNA_sequence.fasta"  # 替换为你的 RNA FASTA 文件路径
    output_file = "../data/blast/RPI369/rna_pair_list.txt"
    # 创建 BLAST 数据库
    create_blast_database(fasta_file)
    # 提取所有序列 ID
    seq_ids = []
    current_id = None
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                current_id = line.strip()[1:]
                seq_ids.append(current_id)
    for query_id in seq_ids:
        # 提取目标 RNA 序列到一个单独的文件
        query_seq = extract_sequence(fasta_file, query_id)
        if query_seq:
            with open("query.fasta", "w") as query_file:
                query_file.write(f">{query_id}\n{query_seq}\n")
            # 执行 BLAST 搜索，并设置相似性阈值
            perform_blast("query.fasta", "rna_db", evalue_threshold=1e-5, perc_identity_threshold=70)
            # 获取所有符合条件的匹配序列 ID（排除自身）
            all_hit_ids = get_all_hits("blast_results.txt", query_id)
            # 创建 RNA-RNA 列表文件
            create_rna_pair_list(query_id, all_hit_ids, output_file)
    # 删除临时文件
    if os.path.exists("query.fasta"):
        os.remove("query.fasta")
    if os.path.exists("blast_results.txt"):
        os.remove("blast_results.txt")
    if os.path.exists("rna_db.nhr"):
        os.remove("rna_db.nhr")
    if os.path.exists("rna_db.nin"):
        os.remove("rna_db.nin")
    if os.path.exists("rna_db.nsq"):
        os.remove("rna_db.nsq")