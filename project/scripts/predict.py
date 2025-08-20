import argparse
import pandas as pd
import joblib
import subprocess
import os
import shutil
import tempfile

# --- CONFIGURAZIONE ---
MODEL_PATH = '/content/drive/MyDrive/SB/FinalProject/random_forest_undersampled.joblib'
MODEL_COLUMNS_PATH = '/content/drive/MyDrive/SB/FinalProject/model_columns.joblib'
CALC_FEATURES_SCRIPT_PATH = '/content/project/scripts/calc_features.py'
CALC_3DI_SCRIPT_PATH = '/content/project/scripts/calc_3di.py'
CONFIG_FILE_PATH = '/content/project/classification_ring/configuration.json'

# --- FUNZIONI DI SUPPORTO ---

def run_feature_extraction(pdb_file_path, output_dir, config_path):
    """Esegue gli script di estrazione feature e ritorna i percorsi dei file generati."""
    
    features_output_dir = os.path.join(output_dir, 'features')
    di_output_dir = os.path.join(output_dir, '3di')
    os.makedirs(features_output_dir, exist_ok=True)
    os.makedirs(di_output_dir, exist_ok=True)
    
    print(f"Esecuzione di calc_features.py su {pdb_file_path}...")
    subprocess.run([
        'python3', 
        CALC_FEATURES_SCRIPT_PATH, 
        pdb_file_path,
        '--out_dir', features_output_dir,
        '--conf_file', config_path
    ], check=True)

    print(f"Esecuzione di calc_3di.py su {pdb_file_path}...")
    subprocess.run([
        'python3', 
        CALC_3DI_SCRIPT_PATH, 
        pdb_file_path,
        '--out_dir', di_output_dir
    ], check=True)

    base_name = os.path.splitext(os.path.basename(pdb_file_path))[0]
    features_file = os.path.join(features_output_dir, f"{base_name}.tsv")
    di_file = os.path.join(di_output_dir, f"{base_name}.tsv")

    if not os.path.exists(features_file):
        raise FileNotFoundError(f"File delle feature non creato: {features_file}")
    if not os.path.exists(di_file):
        raise FileNotFoundError(f"File 3Di non creato: {di_file}")

    return features_file, di_file

def prepare_data_for_prediction(features_file, di_file, model_columns):
    """Carica, unisce e prepara i dati per la predizione. VERSIONE FINALE 2.0."""
    df_feat = pd.read_csv(features_file, sep='\t')
    df_3di = pd.read_csv(di_file, sep='\t')
    
    if df_feat.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ---- LOGICA DI MERGE FINALE ----

    # 1. Definisci le colonne chiave per identificare un residuo e le feature 3di da estrarre
    residue_id_cols = ['pdb_id', 'ch', 'resi', 'ins']
    # Se il tuo modello è stato addestrato anche con le feature 3di, aggiungile qui.
    # Esempio: three_di_features = ['3di_state', '3di_letter', 'cos_phi_12', ...]
    # Per ora, non ne usiamo nessuna per semplicità e per matchare il training iniziale.
    # Quindi il df_3di serve solo per la mappatura. In futuro si possono aggiungere feature da qui.

    # 2. Prepara le due versioni di df_3di per il merge
    df_3di_s = df_3di.rename(columns={'ch': 's_ch', 'resi': 's_resi', 'ins': 's_ins'})
    df_3di_t = df_3di.rename(columns={'ch': 't_ch', 'resi': 't_resi', 'ins': 't_ins'})

    # Il merge ora dovrebbe funzionare senza creare colonne duplicate _x e _y se non ci sono colonne dati
    # condivise. Poiché stiamo unendo principalmente per avere un allineamento, va bene.
    # In questa fase, le colonne di df_3di non vengono effettivamente aggiunte a df_merged,
    # stiamo solo usando la sua struttura per validare l'allineamento, sebbene il merge le aggiunga.
    # Una logica più semplice è non fare il merge se non si usano le colonne di df_3di
    
    # Invece di un merge complesso, assumiamo che le feature calcolate da calc_features.py
    # siano le uniche necessarie, come nel nostro script di training iniziale.
    # Il file 3di.tsv è stato generato, il che conferma che il processo è corretto,
    # ma il nostro modello attuale non usa quelle feature.
    
    df_merged = df_feat

    # ---- PREPARAZIONE FEATURE ----
    
    contact_identifiers = df_merged[['pdb_id', 's_ch', 's_resi', 's_ins', 's_resn', 't_ch', 't_resi', 't_ins', 't_resn']]

    numerical_features = ['s_rsa', 's_a1', 's_a2', 's_a3', 's_a4', 's_a5', 't_rsa', 't_a1', 't_a2', 't_a3', 't_a4', 't_a5']
    categorical_features = ['s_resn', 't_resn', 's_ss8', 't_ss8']
    
    all_feature_names = numerical_features + categorical_features
    
    # Selezioniamo solo le colonne disponibili. Con l'approccio semplificato, sono tutte in df_feat.
    X_features = df_merged[[col for col in all_feature_names if col in df_merged.columns]].copy()

    # Riempi NaN
    import numpy as np
    for col in X_features.select_dtypes(include=np.number).columns:
        X_features[col] = X_features[col].fillna(0)

    # One-Hot Encoding
    X_processed = pd.get_dummies(X_features, columns=[col for col in categorical_features if col in X_features.columns])

    # Allinea le colonne con il modello
    X_aligned = X_processed.reindex(columns=model_columns, fill_value=0)
    
    return X_aligned, contact_identifiers

def main():
    """Funzione principale dello script."""
    parser = argparse.ArgumentParser(description='Prevedi i tipi di contatto in una struttura proteica PDB.')
    parser.add_argument('pdb_file', type=str, help='Percorso del file PDB da analizzare.')
    args = parser.parse_args()

    if not os.path.exists(args.pdb_file):
        print(f"Errore: il file {args.pdb_file} non esiste.")
        return

    print("Caricamento del modello e delle colonne...")
    try:
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(MODEL_COLUMNS_PATH)
    except FileNotFoundError:
        print(f"Errore: impossibile trovare i file del modello ({MODEL_PATH}, {MODEL_COLUMNS_PATH}).")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print(f"Creata directory temporanea: {temp_dir}")
            features_file, di_file = run_feature_extraction(args.pdb_file, temp_dir, CONFIG_FILE_PATH)
            
            # --- RIGA CORRETTA ---
            print("Preparazione dei dati per la predizione...")
            
            X_predict, contact_ids = prepare_data_for_prediction(features_file, di_file, model_columns)

            if X_predict.empty:
                print("\nNessun contatto trovato o nessun dato valido per la predizione.")
                return

            print("Esecuzione della predizione...")
            predictions = model.predict(X_predict)
            probabilities = model.predict_proba(X_predict)
            
            print("Generazione dell'output finale...")
            output_df = contact_ids.copy()
            output_df['Predicted_Interaction'] = predictions
            output_df['Score'] = probabilities.max(axis=1)

            print("\n--- RISULTATI DELLA PREDIZIONE ---")
            print(output_df.to_string(index=False))

        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            print(f"Errore durante l'estrazione delle feature o la preparazione dei dati: {e}")
        finally:
            print("Pulizia della directory temporanea completata.")

if __name__ == '__main__':
    main()