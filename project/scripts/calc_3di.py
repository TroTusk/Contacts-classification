import numpy as np
import logging
from pathlib import Path, PurePath
import argparse
import os
import pandas as pd
import torch
from Bio.SeqUtils import seq1

# --- MODIFICA 1: Import robusti ---
from Bio.PDB import FastMMCIFParser, PDBParser, is_aa
# Lo script viene importato ma viene trovato tramite PYTHONPATH
import foldseek_extract_pdb_features


# 50 letters (X/x are missing)
LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWYZabcdefghijklmnopqrstuvwyz'

# --- MODIFICA 2: Gestione robusta dei percorsi ---
# Troviamo il percorso alla cartella che contiene tutti gli strumenti (es. classification_ring)
# Questo è più robusto di un percorso hardcoded.
# Si assume che questo script sia in 'project/scripts' e i dati in 'project/classification_ring/3di_model'
_script_dir = str(PurePath(os.path.realpath(__file__)).parent)
# Percorso corretto alla cartella del modello
model_dir = os.path.abspath(os.path.join(_script_dir, '..', 'classification_ring', '3di_model'))
if not os.path.isdir(model_dir):
    # Se la struttura è diversa, logga un avviso ma continua
    logging.warning(f"La cartella del modello 3di non è stata trovata in {model_dir}. Si tenterà con un percorso relativo.")
    model_dir = '3di_model'


def encoder_features(residues, virt_cb=(270, 0, 2), full_backbone=True):
    """
    Calculate 3D descriptors for each residue of a PDB file.
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
    parser.add_argument('pdb_file', help='mmCIF or PDB file')
    parser.add_argument('-out_dir', '--out_dir', help='Output directory', default='.')
    # --- MODIFICA 3: Aggiunto conf_file fittizio per robustezza ---
    # Questo script non usa conf_file, ma lo accettiamo come argomento per non fallire se predict.py glielo passa.
    parser.add_argument('-conf_file', '--conf_file', help='(Ignored) Configuration file.', default=None)
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    # Setup del logger
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    if not rootLogger.hasHandlers():
        rootLogger.setLevel(logging.INFO)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

    # Start
    pdb_id = Path(args.pdb_file).stem
    logging.info("{} 3Di processing".format(pdb_id))

    # Load model parameters
    try:
        # --- MODIFICA 4: Correzione per la compatibilità di PyTorch ---
        encoder = torch.load(os.path.join(model_dir, 'encoder.pt'), weights_only=False)
        centroids = np.loadtxt(os.path.join(model_dir, 'states.txt'))
        encoder.eval()
    except FileNotFoundError as e:
        logging.error(f"Errore nel caricamento del modello 3Di: {e}. Controlla che il percorso '{model_dir}' sia corretto.")
        raise

    # --- MODIFICA 5: Scelta dinamica del parser PDB/mmCIF ---
    if args.pdb_file.lower().endswith(('.cif', '.mmcif')):
        parser = FastMMCIFParser(QUIET=True)
    elif args.pdb_file.lower().endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Formato file non supportato: {args.pdb_file}. Deve essere .pdb, .cif, o .mmcif.")
    
    structure = parser.get_structure(pdb_id, args.pdb_file)

    data = []
    # Usiamo un approccio che itera direttamente sui residui validi
    all_valid_residues = { (res.get_parent().id, res.id): res for model in structure for chain in model for res in chain if is_aa(res) }

    for chain in structure[0]:
        # Prendi i residui della catena corrente che sono anche validi globalmente
        residues = [res for res in chain.get_residues() if (res.get_parent().id, res.id) in all_valid_residues]
        
        if not residues:
            continue
            
        feat, mask = encoder_features(residues)
        res_features = feat[mask]
        
        # Filtra i residui che non hanno feature calcolabili
        valid_residues = [res for i, res in enumerate(residues) if mask[i]]
        if not valid_residues:
            continue
            
        with torch.no_grad():
            res = encoder(torch.tensor(res_features, dtype=torch.float32)).detach().numpy()

        valid_states = discretize(centroids, res)

        for i, state in enumerate(valid_states):
            current_residue = valid_residues[i]
            data.append((pdb_id, chain.id, *current_residue.id[1:], seq1(current_residue.get_resname()), state, LETTERS[state]))
    
    if not data:
        logging.warning(f"{pdb_id}: non è stato possibile calcolare le feature 3Di per nessun residuo.")
        # Crea comunque un file vuoto per non far fallire la pipeline
        df = pd.DataFrame(columns=['pdb_id', 'ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter'])
    else:
        # Crea un DataFrame e salva su file
        df = pd.DataFrame(data, columns=['pdb_id', 'ch', 'resi', 'ins', 'resn', '3di_state', '3di_letter'])

    # --- MODIFICA 6: Creazione sicura della directory di output ---
    os.makedirs(args.out_dir, exist_ok=True)
    output_path = os.path.join(args.out_dir, f"{pdb_id}.tsv")
    df.to_csv(output_path, sep="\t", index=False)

    logging.info(f"File 3Di salvato in: {output_path}")