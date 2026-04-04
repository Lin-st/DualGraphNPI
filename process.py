def process_fasta(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        current_id = None
        current_seq = ""
        for line in infile:
            if line.startswith('>'):
                if current_id is not None:
                    # 将处理后的序列写入输出文件
                    outfile.write(f">{current_id}\n{current_seq}\n")
                # 提取sp|后面的字符作为新的ID
                parts = line.strip().split('|')
                if len(parts) >= 3:
                    current_id = parts[1]
                else:
                    current_id = line.strip()[1:]
                current_seq = ""
            else:
                # 去除序列行的换行符并拼接
                current_seq += line.strip()
        # 处理最后一个序列
        if current_id is not None:
            outfile.write(f">{current_id}\n{current_seq}\n")

if __name__ == "__main__":
    input_file = "data/protein_sequence/RPI7317/protein_sequence.fasta"  # 请替换为实际输入文件名
    output_file = "data/protein_sequence/RPI7317/protein_sequence.fasta"  # 请替换为实际输出文件名
    process_fasta(input_file, output_file)