class Node:
    def __init__(self, name, serial_number, node_type):
        self.name = name
        self.interaction_list = []
        self.serial_number = serial_number  # 从0开始的序号
        self.embedded_vector = []
        self.attributes_vector = []
        self.node_type = node_type


class LncRNA:
    def __init__(self, lncRNA_name, serial_number, node_type):
        Node.__init__(self, lncRNA_name, serial_number, node_type)


class Protein:
    def __init__(self, protein_name, serial_number, node_type):
        Node.__init__(self, protein_name, serial_number, node_type)


class LncRNA_Protein_Interaction:
    def __init__(self, lncRNA, protein, y: int, key=None):
        self.lncRNA = lncRNA
        self.protein = protein
        self.y = y  # y=1代表真的连接，y=0代表假的连接
        self.key = key

class RNA2RNA:
    def __init__(self, lncRNA1, lncRNA2, y: int, key=None):
        self.lncRNA1 = lncRNA1
        self.lncRNA2 = lncRNA2
        self.y = y  # y=1代表lncRNA，y=0代表蛋白质
        self.key = key