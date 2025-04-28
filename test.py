import openpyxl


def process_text_file(file_path):
    pairs = set()
    query_with_matches = set()
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        current_query_id = None
        for line in lines:
            line = line.strip()
            if line.startswith('查询 ID:'):
                current_query_id = line.split(': ')[1]
            elif line.startswith('匹配 ID:') and current_query_id:
                match_id = line.split(': ')[1]
                # 确保查询ID和匹配ID按字母顺序排序，避免重复对
                pair = tuple(sorted([current_query_id, match_id]))
                pairs.add(pair)
                query_with_matches.add(current_query_id)

    # 去除没有匹配ID的查询ID对应的对
    valid_pairs = [pair for pair in pairs if pair[0] in query_with_matches and pair[1] in query_with_matches]

    return valid_pairs


def write_to_excel(pairs, output_file_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(['查询ID', '匹配ID'])
    for pair in pairs:
        sheet.append(list(pair))
    workbook.save(output_file_path)


if __name__ == "__main__":
    input_file = 'protein_pair_list.txt'
    output_file = 'protein_protein_pairs.xlsx'
    rna_rna_pairs = process_text_file(input_file)
    write_to_excel(rna_rna_pairs, output_file)