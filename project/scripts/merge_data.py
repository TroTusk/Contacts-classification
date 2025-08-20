import pandas as pd
import glob
import os

# --- INIZIO CONFIGURAZIONE ---

# 1. Definisci il percorso della cartella dove si trovano TUTTI i tuoi file di training.
training_data_folder = '/content/project/features_ring'

# 2. Definisci come vuoi chiamare il file finale che conterrà tutti i dati uniti.
output_filename = 'training_dataset_completo.tsv'

# --- FINE CONFIGURAZIONE ---


# 3. Trova tutti i file nella cartella di training.
#    L'asterisco `*` è un carattere jolly. `*.tsv` cerca tutti i file che finiscono con `.tsv`.
#    Se i tuoi file avessero un'altra estensione, ad esempio .txt, dovresti scrivere '*.txt'.
search_path = os.path.join(training_data_folder, "*.tsv") # Crea il percorso di ricerca corretto
all_files = glob.glob(search_path)

if not all_files:
    print(f"ERRORE: Nessun file .tsv è stato trovato nella cartella '{training_data_folder}'.")
    print("Controlla che il percorso sia corretto e che i file siano presenti.")
else:
    print(f"Trovati {len(all_files)} file. Inizio la procedura di unione...")

    # 4. Crea una lista vuota dove metteremo ogni file dopo averlo letto.
    dataframe_list = []
    
    # 5. Itera su ogni file trovato.
    for file_path in all_files:
        # Leggi il singolo file .tsv in un DataFrame di pandas.
        df = pd.read_csv(file_path, sep='\t')
        # Aggiungi il DataFrame appena letto alla nostra lista.
        dataframe_list.append(df)

    # 6. Concatena (unisci) tutti i DataFrame presenti nella lista in un unico DataFrame.
    #    `ignore_index=True` resetta l'indice del nuovo DataFrame da 0 fino alla fine.
    print("Unione dei file in corso... (potrebbe richiedere un po' di tempo)")
    combined_df = pd.concat(dataframe_list, ignore_index=True)

    # 7. Salva il DataFrame finale in un nuovo file .tsv.
    #    `index=False` è fondamentale per evitare di scrivere una colonna di indice extra nel file.
    print(f"Salvataggio del dataset unito nel file '{output_filename}'...")
    combined_df.to_csv(output_filename, sep='\t', index=False)

    print("\n--- Operazione Completata! ---")
    print(f"File '{output_filename}' creato con successo.")
    print(f"Dimensioni del dataset combinato: {combined_df.shape[0]} righe, {combined_df.shape[1]} colonne.")
    print("\nOra puoi usare questo file per il Passo 3 (addestrare il modello di machine learning).")