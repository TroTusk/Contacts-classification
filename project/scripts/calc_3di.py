import numpy as np
import logging
from pathlib import Path, PurePath
import argparse
import os
import pandas as pd
import torch
from Bio.SeqUtils import seq1

# Robust imports
from Bio.PDB import FastMMCIFParser, PDBParser, is_aa
# This module is resolved via PYTHONPATH
import foldseek_extract_pdb_features


# 50 letters (X/x are missing)
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz'

# Robust path handling:
# Try to locate the 3Di model folder relative to this script.
# Assumes this file lives in 'project/scripts' and models in 'project/classification_ring/3di_model'.
_script_dir = str(PurePath(os.path.realpath(__file__)).parent)
model_dir = os.path.abspath(os.path.join(_script_dir, '..', 'classification_ring', '3di_model'))
if not os.path.isdir(model_dir):
    # If the layout is different, warn and fall back to a relative path.
    logging.warning(f"3Di model folder not found at {model_dir}. Falling back to '3di_model' relative path.")
    model_dir = '3di_model'


def encoder_features(residues, virt_cb=(270, 0, 2), full_backbone=True):
    """
    Compute 3D descriptors for each residue in a structure.
    """
    coords, valid_mask = foldseek_extract_pdb_features.get_atom_coordinates(residues, full_backbone=full_backbone)
    coords = foldseek_extract_pdb_features.move_CB(coords, virt_cb=virt_cb)
    partner_idx = foldseek_extract_pdb_features.find_nearest_residues(coords, valid_mask)
    features, valid_mask2 = foldseek_extract_pdb_features.calc_angles_forloop(coords, partner_idx, valid_mask)
    seq_dist = (partner_idx - np.arange(len(partner_idx)))[:, np.newaxis]
    log_dist = np.sign(seq_dist) * np.log(np.abs(seq_dist) + 1)
    vae_features = np.hstack([features, log_dist])
    return vae_features, valid_mask2

def discretize(centroids, x):
    return np.argmin(foldseek_extract_pdb_features.distance_matrix(x, centroids), axis=1)

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_file', help='Input structure file (.mmCIF or .PDB)')
    parser.add_argument('-out_dir', '--out_dir', help='Output directory', default='.')
    # Accept (but ignore) a conf_file for compatibility with external callers (e.g., predict.py)
    parser.add_argument('-conf_file', '--conf_file', help='(Ignored) Configuration file placeholder.', default=None)
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    # Logger setup
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    if not rootLogger.hasHandlers():
        rootLogger.setLevel(logging.INFO)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    # Start
    pdb_id = Path(args.pdb_file).stem
    logging.info(f"{pdb_id} 3Di processing")

    # Load model parameters
    try:
        # Torch compatibility: allow full object load (older checkpoints)
        encoder = torch.load(os.path.join(model_dir, 'encoder.pt'), weights_only=False)
        centroids = np.loadtxt(os.path.join(model_dir, 'states.txt'))
        encoder.eval()
    except FileNotFoundError as e:
        logging.error(f"Error loading 3Di model: {e}. Check that the path '{model_dir}' is correct.")
        raise

    # Select appropriate parser based on file extension
    if args.pdb_file.lower().endswith(('.cif', '.mmcif')):
        parser = FastMMCIFParser(QUIET=True)
    elif args.pdb_file.lower().endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format: {args.pdb_file}. Expected .pdb, .cif, or .mmcif.")
    
    structure = parser.get_structure(pdb_id, args.pdb_file)

    data = []
    # Iterate over valid amino-acid residues only
    all_valid_residues = {(res.get_parent().id, res.id): res
                          for model in structure
                          for chain in model
                          for res in chain if is_aa(res)}

    for chain in structure[0]:
        # Keep residues that are valid at the global structure scope
        residues = [res for res in chain.get_residues()
                    if (res.get_parent().id, res.id) in all_valid_residues]
        
        if not residues:
            continue
            
        feat, mask = encoder_features(residues)
        res_features = feat[mask]
        
        # Skip chains where no features can be computed
        valid_residues = [res for i, res in enumerate(residues) if mask[i]]
        if not valid_residues:
            continue
            
        with torch.no_grad():
            res = encoder(torch.tensor(res_features, dtype=torch.float32)).detach().numpy()

        valid_states = discretize(centroids, res)

        for i, state in enumerate(valid_states):
            current_residue = valid_residues[i]
            data.append((pdb_id, chain.id,
                         *current_residue.id[1:],  # (resi, ins)
                         seq1(current_residue.get_resname()),
                         state, LETTERS[state]))
    
    if not data:
        logging.warning(f"{pdb_id}: could not compute 3Di features for any residue.")
        # Write an empty file to keep downstream steps from failing
        df = pd.DataFrame(columns=['pdb_id', 'ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter'])
    else:
        # Build DataFrame and save
        df = pd.DataFrame(data, columns=['pdb_id', 'ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter'])

    # Ensure output directory exists and write TSV
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, f"{pdb_id}.tsv")
    df.to_csv(output_path, sep="\t", index=False)

    logging.info(f"3Di file written to: {output_path}")
