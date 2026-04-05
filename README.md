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

- `{db_name}` is the database name (e.g., `RPI369`, `NPInter5`). All scripts accept the `--db_name` argument and automatically construct paths.
- **All input files must exist**; otherwise the scripts will raise `FileNotFoundError`.
- **Output directories are created automatically**.

## 1. Generate Feature Embeddings

### 1.1 ESM Protein Embeddings

```bash
python -m src.ESM --db_name <database_name>
```

### 1.2 RNA‑FM lncRNA Embeddings

```bash
python -m src.RNA-FM --db_name <database_name>
```

## 2. BLAST Homology Analysis

### 2.1 Run BLAST Searches

```bash
python -m src.generate_rna_blast --db_name <database_name>
python -m src.generate_protein_blast --db_name <database_name>
```

### 2.2 Convert to Excel Format

```bash
python -m src.lncRNA_blast_dataset --db_name <database_name>
python -m src.protein_blast_dataset --db_name <database_name>
```

### 3. Build Heterogeneous Graphs and Generate 5‑Fold Cross‑Validation Data
```bush
python -m src.generate_edgelist --projectName <database_name> --interactionDatasetName <database_name> --createBalanceDataset 1
```
- projectName: output directory name (usually the database name)

- interactionDatasetName: prefix of the interaction dataset file (usually the database name)

- Output: data/graph/{projectName}/fold_0/ – fold_4/

### 4. Train the Model
Example for fold 0:
```bush
python -m src.train_save_model --projectName <database_name> --fold 0
```
