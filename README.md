# GateSolv
A structure-guided protein solubility prediction framework integrating global descriptors, residue-graph learning, and gated feature fusion.

Installation
Requirements

This project requires Python 3.10 or later and depends on both standard machine-learning packages and structural-biology toolkits.

Core dependencies include:

numpy
pandas
scikit-learn
matplotlib
joblib
torch
torch-geometric
biopython
freesasa
esm
optionally:
xgboost
lightgbm
catboost

Input data
Required input for inference

The inference pipeline starts from a directory containing single-chain PDB files:
your_pdb_dir/
├── proteinA.pdb
├── proteinB.pdb
└── proteinC.pdb

Training data source
The present project was trained using protein-solubility benchmark data derived from the ProtSolM resource.

End-to-end training workflow
The training procedure is staged and should be executed in the following order.
1. Extract global features
Run the five global-feature scripts separately for each split.
Example for the training split:
python 1_global_feature/1-sequence_global.py \
    --pdb_dir data/train/pdb \
    --out_csv data/train/global/1_sequence_full.csv \
    --out_csv_reduced data/train/global/1_sequence_train_reduced.csv

python 1_global_feature/2-structure_confidence_global.py \
    --pdb_dir data/train/pdb \
    --out_csv data/train/global/2_confidence_full.csv \
    --out_csv_reduced data/train/global/2_confidence_train_reduced.csv

python 1_global_feature/3-surface_physchem_aggregation_risk_global.py \
    --pdb_dir data/train/pdb \
    --out_csv data/train/global/3_surface_full.csv \
    --out_csv_reduced data/train/global/3_surface_train_reduced.csv

python 1_global_feature/4-compactness_shape_global.py \
    --pdb_dir data/train/pdb \
    --out_csv data/train/global/4_shape_full.csv \
    --out_csv_reduced data/train/global/4_shape_train_reduced.csv

python 1_global_feature/5-interaction_network_global.py \
    --pdb_dir data/train/pdb \
    --out_csv data/train/global/5_interaction_full.csv \
    --out_csv_reduced data/train/global/5_interaction_train_reduced.csv

Merge reduced global features:
python 1_global_feature/6-merge_all_reduced_csv.py \
    --root_dir data/train/global \
    --out_dir data/train/global

Repeat the same procedure for validation and test splits.


2. Train the global encoder
python 1_global_feature/7_train_global_encoder.py \
    --train_global_csv data/train/global/global_features_train_merged.csv \
    --train_label_csv data/train/train_labels.csv \
    --val_global_csv data/val/global/global_features_valid_merged.csv \
    --val_label_csv data/val/val_labels.csv \
    --test_global_csv data/test/global/global_features_test_merged.csv \
    --test_label_csv data/test/test_labels.csv \
    --out_dir results/global_encoder
Export learned global embeddings:
python 1_global_feature/8-export_h_global_from_ckpt.py \
    --global_csv data/train/global/global_features_train_merged.csv \
    --label_csv data/train/train_labels.csv \
    --ckpt results/global_encoder/global_encoder_best.pt \
    --out_csv results/global_encoder/h_global_train.csv

3.Build graph features
Construct residue-graph edges:
python 2_graph_feature/1-1-build_edges.py \
    --pdb_dir data/train/pdb \
    --out_dir data/train/edges \
    --index_csv data/train/edges/index.csv

Construct node features:
python 2_graph_feature/1-2-build_node_features_full.py \
    --pdb_dir data/train/pdb \
    --out_dir data/train/nodes \
    --index_csv data/train/nodes/index.csv \
    --device cuda \
    --fp16

Repeat for validation and test splits.

4. Train the graph encoder
python 2_graph_feature/2_train_graph_encoder_with_test_eval.py \
    --train_node_dir data/train/nodes \
    --train_edge_dir data/train/edges \
    --train_label_csv data/train/train_labels.csv \
    --val_node_dir data/val/nodes \
    --val_edge_dir data/val/edges \
    --val_label_csv data/val/val_labels.csv \
    --test_node_dir data/test/nodes \
    --test_edge_dir data/test/edges \
    --test_label_csv data/test/test_labels.csv \
    --out_dir results/graph_encoder

   Export learned graph embeddings:
   
   python 2_graph_feature/3_export_h_graph_from_ckpt.py \
    --node_dir data/train/nodes \
    --edge_dir data/train/edges \
    --label_csv data/train/train_labels.csv \
    --ckpt results/graph_encoder/graph_encoder_best.pt \
    --out_csv results/graph_encoder/h_graph_train.csv

   Repeat for validation and test splits.


5. Run fusion ablation and export fused embeddings

   python 3_fusion/step3_fuse_embed.py \
    --h_graph_train results/graph_encoder/h_graph_train.csv \
    --h_global_train results/global_encoder/h_global_train.csv \
    --h_graph_val results/graph_encoder/h_graph_val.csv \
    --h_global_val results/global_encoder/h_global_val.csv \
    --h_graph_test results/graph_encoder/h_graph_test.csv \
    --h_global_test results/global_encoder/h_global_test.csv \
    --out_dir results/fusion \
    --export_all_variants

    The best fusion variant will be selected automatically according to the chosen validation metric, and exported fused representations will be written to the exports/ directory.

6. Train the final classifier and meta-learning stack
   python 4_classifier/train-v6-formal.revised.py \
    --train_csv results/fusion/exports/vector_noln.h_fused_train.csv \
    --val_csv results/fusion/exports/vector_noln.h_fused_val.csv \
    --test_csv results/fusion/exports/vector_noln.h_fused_test.csv \
    --out_dir results/formal_training \
    --device cuda

   This script benchmarks classical learners, trains MLP baselines, performs family-constrained base-model selection, builds out-of-fold meta-learning inputs, evaluates meta learners, and exports a deployable inference package.



End-to-end inference from raw PDB files

After training, inference on new proteins can be executed directly from raw PDB files using the unified inference script.

python infer.py \
    --pdb_dir demo_pdbs \
    --work_dir demo_work \
    --out_csv demo_predictions.csv \
    --graph_ckpt results/graph_encoder/graph_encoder_best.pt \
    --global_ckpt results/global_encoder/global_encoder_best.pt \
    --fuse_ckpt results/fusion/models/vector_noln/vector_noln.pt \
    --train_run_dir results/formal_training/run_YYYYMMDD_HHMMSS \
    --device cuda:0

Pretrained single-model files
The pretrained single ML models used in the final classification stage are distributed separately via Baidu Netdisk.

Link: https://pan.baidu.com/s/1kyTmC6j-JyfieSykpKHM0A?pwd=8qgu
Extraction code: 8qgu

Please download these files and place them in the expected model directory before running inference or reproducing the final classification stage.
