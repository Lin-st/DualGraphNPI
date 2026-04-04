import subprocess
import os

def create_blast_database(fasta_file):
    """
    创建 BLAST 蛋白质数据库
    """
    makeblastdb_cmd = f"makeblastdb -in {fasta_file} -dbtype prot -out protein_db"
    try:
        subprocess.run(makeblastdb_cmd, shell=True, check=True)
        print("BLAST 蛋白质数据库创建成功")
    except subprocess.CalledProcessError as e:
        print(f"创建 BLAST 蛋白质数据库时出错: {e}")


def perform_blast(query_file, db_name, evalue_threshold=1e-5, identity_threshold=90, coverage_threshold=90):
    """
    执行 BLAST 蛋白质搜索，并设置相似性阈值
    """
    blastp_cline = f"blastp -query {query_file} -db {db_name} -out blast_results.txt -outfmt \"6 qseqid sseqid pident qcovhsp evalue\" -evalue {evalue_threshold} -qcov_hsp_perc {coverage_threshold}"
    try:
        subprocess.run(blastp_cline, shell=True, check=True)
        filtered_results = []
        with open('blast_results.txt', 'r') as result_file:
            for line in result_file:
                parts = line.strip().split('\t')
                if len(parts) >= 4:
                    pident = float(parts[2])
                    if pident >= identity_threshold:
                        filtered_results.append(line)
        with open('blast_results.txt', 'w') as result_file:
            result_file.writelines(filtered_results)
        print("BLAST 蛋白质搜索完成，结果已过滤")
    except subprocess.CalledProcessError as e:
        print(f"执行 BLAST 蛋白质搜索时出错: {e}")


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
                # 处理BLAST结果中的ID格式
                processed_qid = process_protein_id(parts[0])
                processed_sid = process_protein_id(parts[1])
                if processed_qid == query_id and processed_sid != query_id:
                    hits.append(processed_sid)
        if not hits:
            print(f"未找到 {query_id} 的匹配结果（排除自身）")
        return hits
    except FileNotFoundError:
        print("BLAST 结果文件未找到")
        return []


def extract_sequence(fasta_file, seq_id):
    """
    从 FASTA 文件中提取指定 ID 的序列
    """
    current_id = None
    current_seq = ""
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                if current_id == seq_id:
                    return current_seq
                current_id = process_protein_id(line.strip()[1:])
                current_seq = ""
            else:
                if current_id == seq_id:
                    current_seq += line.strip()
    if current_id == seq_id:
        return current_seq
    return None


def create_protein_pair_list(query_id, all_hit_ids, output_file):
    """
    将查询 ID 和匹配 ID 按指定格式写入列表文件
    """
    with open(output_file, "a", encoding='utf-8') as outfile:
        outfile.write(f"查询 ID: {query_id}\n")
        for hit_id in all_hit_ids:
            outfile.write(f"匹配 ID: {hit_id}\n")
        outfile.write("\n")


def process_protein_id(raw_id):
    """
    处理蛋白质ID，提取sp|或tr|后面的ID部分
    """
    if '|' in raw_id:
        parts = raw_id.split('|')
        if len(parts) >= 3 and parts[0] in ['sp', 'tr']:
            return parts[1]
    return raw_id


if __name__ == "__main__":
    fasta_file = "../data/protein_sequence/RPI369/protein_sequence.fasta"
    output_file = "../data/blast/RPI369/protein_pair_list.txt"
    # 创建 BLAST 蛋白质数据库
    create_blast_database(fasta_file)

    # 提取所有序列 ID
    seq_ids = []
    current_id = None
    with open(fasta_file) as f:
        for line in f:
            if line.startswith('>'):
                current_id = process_protein_id(line.strip()[1:])
                seq_ids.append(current_id)

    for query_id in seq_ids:
        # 提取目标蛋白质序列到一个单独的文件
        query_seq = extract_sequence(fasta_file, query_id)
        if query_seq:
            with open("query.fasta", "w") as query_file:
                query_file.write(f">{query_id}\n{query_seq}\n")

            # 执行 BLAST 蛋白质搜索，并设置相似性阈值
            perform_blast("query.fasta", "protein_db", evalue_threshold=1e-5, identity_threshold=50, coverage_threshold=10)

            # 获取所有符合条件的匹配序列 ID（排除自身）
            all_hit_ids = get_all_hits("blast_results.txt", query_id)

            # 创建蛋白质 - 蛋白质列表文件
            create_protein_pair_list(query_id, all_hit_ids, output_file)

    # 删除临时文件
    if os.path.exists("query.fasta"):
        os.remove("query.fasta")
    if os.path.exists("blast_results.txt"):
        os.remove("blast_results.txt")
    if os.path.exists("protein_db.phr"):
        os.remove("protein_db.phr")
    if os.path.exists("protein_db.pin"):
        os.remove("protein_db.pin")
    if os.path.exists("protein_db.psq"):
        os.remove("protein_db.psq")