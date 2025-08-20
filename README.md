# Contacts-classification
Predict residue–residue contact types in protein structures (RING labels: HBOND, VDW, IONIC, PIPISTACK, PICATION, PIHBOND, SSBOND) using supervised learning.

The repo includes a Colab-friendly notebook that:

- Loads per-PDB feature tables (features_ring/*.tsv) or a merged fallback dataset,

- Builds features (numerics + one-hot categoricals, optional 3Di letters),

- Trains one binary model per class (one-vs-rest),

- Evaluates on a hold-out set split by PDB to avoid leakage,

- Tests on a chosen PDB or a random PDB.

# Requirements

- Python 3.10+

- Packages: numpy, pandas, scikit-learn, xgboost

- Colab works out of the box.

# Quickstart (with Colab)

1. Upload the project folder to your Drive under:
  /MyDrive/progetto_bioinfo_finale/project/
  optionally add complete_training_dataset to the project folder, downloadable from this link:
  https://drive.google.com/file/d/1qwxPwq1lBrWXs5Q4dn4A84bNS4V5zc3X/view?usp=sharing
  extract the zip file inside the project folder

2. Run the contacts_classification_structural_bioinfo.ipynb file on google colab

3. Login in google drive following the instructions given by Colab
