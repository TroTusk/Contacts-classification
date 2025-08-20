from Bio.PDB import DSSP, HSExposureCB, PPBuilder, is_aa, NeighborSearch
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBParser import PDBParser
from Bio.SeqUtils import seq1

import pandas as pd
from pathlib import Path, PurePath
import json
import argparse
import logging
import os


def arg_parser():
    # Parse CLI arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('pdb_file', help='Input structure file (.mmCIF or .PDB)')
    parser.add_argument('-conf_file', '--conf_file', help='Configuration JSON with parameters', default=None)
    parser.add_argument('-out_dir', '--out_dir', help='Output directory', default='.')
    return parser.parse_args()


if __name__ == '__main__':

    args = arg_parser()

    # Logger setup
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Robust handling of configuration paths:
    # - If args.conf_file is relative, resolve it against this script's folder.
    src_dir = str(PurePath(os.path.realpath(__file__)).parent)
    config_file = args.conf_file if args.conf_file is not None else os.path.join(src_dir, "configuration.json")

    with open(config_file) as f:
        config = json.load(f)

    # Normalize any relative paths found inside the JSON (e.g., *_file, *_dir).
    config_dir = str(PurePath(os.path.realpath(config_file)).parent)
    for k in config:
        if (k.endswith('_file') or k.endswith('_dir')) and not os.path.isabs(config[k]):
            config[k] = os.path.join(config_dir, config[k])

    # Run
    pdb_id = Path(args.pdb_file).stem
    logging.info(f"{pdb_id} processing")

    # Load Ramachandran regions (matrix of bins/labels)
    regions_matrix = []
    with open(config["rama_file"]) as f:
        for line in f:
            if line:
                regions_matrix.append([int(ele) for ele in line.strip().split()])

    # Load Atchley scales (per-residue physico-chemical factors)
    atchley_scale = {}
    with open(config["atchley_file"]) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip().split("\t")
            atchley_scale[line[0]] = line[1:]

    # Choose the appropriate structure parser based on file extension
    if args.pdb_file.lower().endswith(('.cif', '.mmcif')):
        parser = MMCIFParser(QUIET=True)
    elif args.pdb_file.lower().endswith('.pdb'):
        parser = PDBParser(QUIET=True)
    else:
        raise ValueError(f"Unsupported file format for {args.pdb_file}. Must be .pdb, .cif, or .mmcif")

    structure = parser.get_structure(pdb_id, args.pdb_file)

    # Collect valid amino-acid residues (exclude hetero/water; residue.id[0] == ' ')
    residues = [residue for residue in structure[0].get_residues() if is_aa(residue) and residue.id[0] == ' ']
    if not residues:
        logging.warning(f"{pdb_id} no valid residues (skipping prediction)")
        raise ValueError("no valid residues")

    # DSSP (secondary structure + RSA). If it fails, keep empty dict and leave fields as None.
    dssp = {}
    try:
        dssp = dict(DSSP(structure[0], os.path.abspath(args.pdb_file), dssp=config["dssp_file"]))
    except Exception as e:
        logging.warning(f"{pdb_id} DSSP error: {e}")

    # Half Sphere Exposure (HSE); if it fails, keep empty dict.
    hse = {}
    try:
        hse = dict(HSExposureCB(structure[0]))
    except Exception as e:
        logging.warning(f"{pdb_id} HSE error: {e}")

    # Ramachandran φ/ψ and a coarse SS class from config ranges
    rama_dict = {}  # {(chain_id, residue_id): [phi, psi, ss_class], ...}
    ppb = PPBuilder()
    for chain in structure[0]:
        for pp in ppb.build_peptides(chain):
            phi_psi = pp.get_phi_psi_list()
            for i, residue in enumerate(pp):
                phi, psi = phi_psi[i]
                ss_class = None
                if phi is not None and psi is not None:
                    for x, y, width, height, ss_c, color in config["rama_ss_ranges"]:
                        if x <= phi < x + width and y <= psi < y + height:
                            ss_class = ss_c
                            break
                rama_dict[(chain.id, residue.id)] = [phi, psi, ss_class]

    # Build residue–residue contacts with a distance cutoff and add per-residue features
    data = []
    ns = NeighborSearch([atom for residue in residues for atom in residue])
    for residue_1, residue_2 in ns.search_all(config["distance_threshold"], level="R"):
        index_1 = residues.index(residue_1)
        index_2 = residues.index(residue_2)

        # Skip close sequence neighbors (keep only contacts with enough sequence separation)
        if abs(index_1 - index_2) >= config["sequence_separation"]:

            aa_1 = seq1(residue_1.get_resname())
            aa_2 = seq1(residue_2.get_resname())
            chain_1 = residue_1.get_parent().id
            chain_2 = residue_2.get_parent().id

            data.append((
                pdb_id,
                chain_1,
                *residue_1.id[1:],  # (resi, ins)
                aa_1,
                *dssp.get((chain_1, residue_1.id), [None, None, None, None])[2:4],  # ss8, rsa
                *hse.get((chain_1, residue_1.id), [None, None])[:2],                # up, down
                *rama_dict.get((chain_1, residue_1.id), [None, None, None]),        # phi, psi, ss3
                *atchley_scale[aa_1],
                chain_2,
                *residue_2.id[1:],  # (resi, ins)
                aa_2,
                *dssp.get((chain_2, residue_2.id), [None, None, None, None])[2:4],
                *hse.get((chain_2, residue_2.id), [None, None])[:2],
                *rama_dict.get((chain_2, residue_2.id), [None, None, None]),
                *atchley_scale[aa_2]
            ))

    if not data:
        logging.warning(f"{pdb_id} no contacts found (skipping prediction)")
        raise ValueError("no contacts error (skipping prediction)")

    # Assemble DataFrame and write TSV
    df = pd.DataFrame(
        data,
        columns=[
            'pdb_id',
            's_ch', 's_resi', 's_ins', 's_resn', 's_ss8', 's_rsa', 's_up', 's_down', 's_phi', 's_psi', 's_ss3',
            's_a1', 's_a2', 's_a3', 's_a4', 's_a5',
            't_ch', 't_resi', 't_ins', 't_resn', 't_ss8', 't_rsa', 't_up', 't_down', 't_phi', 't_psi', 't_ss3',
            't_a1', 't_a2', 't_a3', 't_a4', 't_a5'
        ]
    ).round(3)

    os.makedirs(args.out_dir, exist_ok=True)
    output_file_path = os.path.join(args.out_dir, f"{pdb_id}.tsv")
    df.to_csv(output_file_path, sep="\t", index=False)
