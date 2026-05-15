# =========================================================================
# 1. IMPORTAZIONE LIBRERIE E MODULI
# =========================================================================
import numpy as np
import pandas as pd
import os
import json
import optuna
import warnings
from sklearn.mixture import GaussianMixture
from skimage import io, color, measure, morphology
from skimage.filters import threshold_otsu
from stardist.models import StarDist2D
from csbdeep.utils import normalize

warnings.filterwarnings("ignore")

# Stampa intestazione personalizzata
print("\n" * 2) 
print("=" * 57)
print("  OTTIMIZZAZIONE PARAMETRI PER KI-67 ")
print("=" * 57)
print("\n")

# =========================================================================
# 2. CONFIGURAZIONI BASE E PATH
# =========================================================================
cartella_base = r"C:\Users\rocca\Desktop\corso progetto\PYTHON_TEST"
cartella_immagini = os.path.join(cartella_base, "Vetrini")
dimensione_tile = 2000
bianco_puro = np.array([255, 255, 255])

# =========================================================================
# 3. LETTURA GROUND TRUTH QUPATH E PREPARAZIONE IMMAGINI
# =========================================================================
percorso_csv_misure = os.path.join(cartella_base, "misure.csv")
try:
    df_misure = pd.read_csv(percorso_csv_misure, sep=';')
    if 'Image' not in df_misure.columns:
        df_misure = pd.read_csv(percorso_csv_misure, sep=',')
    valori_reali_qupath = dict(zip(df_misure['Image'], df_misure['Positive %']))
except Exception as e:
    print(f" Errore : File 'misure.csv' non trovato o non valido. Controlla il percorso.\n{e}")
    exit()

print("⏳ Inizializzazione Rete Neurale e caricamento cache immagini in corso...")
model = StarDist2D.from_pretrained('2D_versatile_he')

files_immagini = [f for f in os.listdir(cartella_immagini) if f.endswith('.tif')]
immagini_cache = {}

# Pre-caricamento in RAM per abbassare i tempi di I/O durante i trial
for nome_file in files_immagini:
    if nome_file in valori_reali_qupath:
        percorso = os.path.join(cartella_immagini, nome_file)
        img = io.imread(percorso)
        if len(img.shape) == 3:
            img_gray = color.rgb2gray(img)
        else:
            img_gray = img
            
        img_norm = normalize(img_gray, 1, 99.8, axis=(0,1))
        immagini_cache[nome_file] = {
            'norm': img_norm,
            'gray': img_gray,
            'target': valori_reali_qupath[nome_file]
        }

if not immagini_cache:
    print(" Errore: Nessuna immagine corrisponde ai nomi presenti nel file 'misure.csv'.")
    exit()

# =========================================================================
# 4. FUNZIONE OBIETTIVO OPTUNA (MOTORE DI RICERCA)
# =========================================================================
def obiettivo(trial):
    """Esplora le dimensioni nucleari ottimali per minimizzare lo scarto sul Ki-67."""
    
    area_minima = trial.suggest_int('area_minima_nucleo', 10, 80)
    area_massima = trial.suggest_int('area_massima_nucleo', 400, 1500)
    
    if area_minima >= area_massima:
        raise optuna.TrialPruned()

    errore_totale = 0.0
    immagini_analizzate = 0
    
    for nome_file, dati in immagini_cache.items():
        img_norm = dati['norm']
        img_gray = dati['gray']
        target_qupath = dati['target']

        # ---------------------------------------------------------------------
        # FASE A: INFERENZA STARDIST E FILTRAGGIO AREA
        # ---------------------------------------------------------------------
        labels, details = model.predict_instances(img_norm, prob_thresh=0.4, nms_thresh=0.3)
        if labels.max() == 0:
            continue
            
        # Rimuoviamo istantaneamente le etichette fuori range (vettorializzato in C)
        areas = np.bincount(labels.ravel())
        valid_labels = (areas >= area_minima) & (areas <= area_massima)
        valid_labels[0] = False 
        
        map_array = np.zeros(len(areas), dtype=labels.dtype)
        map_array[valid_labels] = np.arange(1, valid_labels.sum() + 1)
        masks = map_array[labels]

        # ---------------------------------------------------------------------
        # FASE B: ESTRAZIONE VETTORIALE DELLE INTENSITÀ
        # ---------------------------------------------------------------------
        num_nuclei = valid_labels.sum()
        if num_nuclei < 5: 
            continue
            
        sums = np.bincount(masks.ravel(), weights=img_gray.ravel())[1:] 
        counts = np.bincount(masks.ravel())[1:]
        intensities = sums / counts

        # ---------------------------------------------------------------------
        # FASE C: CLASSIFICAZIONE GMM E CALCOLO LOSS
        # ---------------------------------------------------------------------
        X = intensities.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, random_state=42)
        labels_gmm = gmm.fit_predict(X)
        
        medie_cluster = [X[labels_gmm == i].mean() for i in range(2)]
        id_cluster_positivo = np.argmax(medie_cluster)
        
        num_positivi = np.sum(labels_gmm == id_cluster_positivo)
        ki67_percentuale = (num_positivi / num_nuclei) * 100.0
        
        # Scarto assoluto tra predetto e clinico
        errore_immagine = abs(ki67_percentuale - target_qupath)
        errore_totale += errore_immagine
        immagini_analizzate += 1

    if immagini_analizzate == 0:
        return 9999.0 
        
    return errore_totale / immagini_analizzate

# =========================================================================
# 5. SISTEMA DI ARRESTO ANTICIPATO (PATIENCE)
# =========================================================================
class ArbitroStallo:
    """Ferma Optuna se non ci sono miglioramenti per 'n' iterazioni."""
    def __init__(self, pazienza_massima=15):
        self.pazienza_massima = pazienza_massima
        self.miglior_valore_finora = float('inf')
        self.tentativi_senza_miglioramento = 0

    def __call__(self, study, trial):
        valore_attuale = study.best_value
        if valore_attuale < self.miglior_valore_finora:
            self.miglior_valore_finora = valore_attuale
            self.tentativi_senza_miglioramento = 0
        else:
            self.tentativi_senza_miglioramento += 1

        if self.tentativi_senza_miglioramento >= self.pazienza_massima:
            print(f"\n STALLO RAGGIUNTO! Nessun miglioramento da {self.pazienza_massima} tentativi.")
            study.stop()

# =========================================================================
# 6. ESECUZIONE RICERCA ED ESPORTAZIONE
# =========================================================================
if __name__ == "__main__":
    
    # Sovrascriviamo il database vecchio usando lo stesso nome per ripartire da zero
    study = optuna.create_study(direction='minimize')
    arbitro_paziente = ArbitroStallo(pazienza_massima=15)
    
    print(f"\n RICERCA INIZIATA SU {len(immagini_cache)} IMMAGINI DI TEST...")
    
    # Esecuzione Ottimizzazione
    study.optimize(obiettivo, n_trials=100, callbacks=[arbitro_paziente])

    # Estrazione Risultati
    miglior_trial = study.best_trial
    parametri_ideali = miglior_trial.params

    print("\n" + "=" * 50)
    print(" OTTIMIZZAZIONE COMPLETATA ")
    print(f" Errore Medio Minimo (Scarto Ki-67): {miglior_trial.value:.2f}%")
    print(" Parametri ottenuti:")
    for chiave, valore in parametri_ideali.items():
        print(f"   -> {chiave}: {valore}")
    print("=" * 50)

    # Salvataggio su JSON
    percorso_json_out = os.path.join(cartella_base, "parametri_ottimali.json")
    with open(percorso_json_out, 'w') as file_json:
        json.dump(parametri_ideali, file_json, indent=4)
        
    print(f"\n I nuovi parametri sono stati salvati in '{percorso_json_out}'.")
    print("Ora puoi lanciare lo script principale per generare le patches finali!")
