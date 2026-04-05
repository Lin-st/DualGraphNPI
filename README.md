# DualGraphNPI: A Contrastive Dual-Graph Method for RNA-Protein Interaction Prediction

This document provides instructions for running the DualGraphNPI pipeline using Python module execution (`python -m src.xxx`). All commands must be executed from the project root directory `DualGraphNPI`.

## Data Organization (Important)

Input data must be placed in the following fixed directory structure. File names and extensions are case‑sensitive.

DualGraphNPI/
└── data/
    ├── lncRNA_sequence/
    │   └── {db_name}/          # e.g., RPI369, NPInter5
    │       └── lncRNA_sequence.fasta  # lncRNA sequences in FASTA format
    ├── protein_sequence/
    │   └── {db_name}/
    │       └── protein_sequence.fasta  # Protein sequences in FASTA format
    ├── source_database_data/
    │   └── {db_name}.xlsx  # Interaction data (3 columns: lncRNA name, protein name, label 0/1)
    ├── lncRNA_3_mer/
    │   └── {db_name}/
    │       └── lncRNA_3_mer.txt  # lncRNA 3‑mer features (">ID" line followed by tab‑separated vector)
    ├── protein_2_mer/
    │   └── {db_name}/
    │       └── protein_2_mer.txt  # Protein 2‑mer features (same format as above)
    ├── RNA-FM/        # Created automatically
    ├── esm/           # Created automatically
    ├── blast/         # Created automatically
    └── graph/         # Created automatically
