# =========================================================================
# 1. IMPORTAZIONE LIBRERIE E MODULI
# =========================================================================
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Mette a tacere i log di TensorFlow
import time
import shutil
import json
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage import io, color, measure, morphology
from skimage.filters import threshold_otsu
from stardist.models import StarDist2D
from csbdeep.utils import normalize
import warnings
from skimage.segmentation import find_boundaries
from matplotlib.lines import Line2D             
warnings.filterwarnings("ignore")

# Stampa intestazione personalizzata
larghezza_schermo = 183
print("\n" * 2) 
print("=" * larghezza_schermo)
print("  PIPELINE KI-67: PATCHES FINALI ".center(larghezza_schermo, '='))
print("=" * larghezza_schermo)
print("\n")

# =========================================================================
# 2. CONFIGURAZIONI GENERALI E PATH
# =========================================================================
cartella_base = r"C:\Users\rocca\Desktop\corso progetto\PYTHON_TEST"
cartella_immagini = os.path.join(cartella_base, "Vetrini")
estensione_file = ".tif"
dimensione_tile = 2000
  
# =========================================================================
# 3. LETTURA PARAMETRI OPTUNA (JSON)
# =========================================================================
percorso_json = os.path.join(cartella_base, "parametri_ottimali.json")

if os.path.exists(percorso_json):
    print(" Trovato file JSON: Caricamento parametri in corso...")
    with open(percorso_json, 'r') as file_json:
        parametri_ottimali = json.load(file_json)
        
    prob_tresh = parametri_ottimali.get('prob_tresh', 0.50)
    nms_tresh = parametri_ottimali.get('nms_tresh', 0.40)
    sensibilita_dab = parametri_ottimali.get('sensibilita_dab', 0.80)
else: 
    print(" File JSON non trovato. Uso parametri di sicurezza.")
    prob_tresh = 0.50; nms_tresh = 0.40; sensibilita_dab = 0.80

# =========================================================================
# 4. PARAMETRI ESTETICI E DIRECTORY OUT
# =========================================================================
# --- PARAMETRI MASCHERA VISIVA (Per Estetica Heatmap) ---
raggio_chiusura_vis = 15
espansione_tessuto_vis = 4
area_minima_isola_vis = 500
dimensione_quadrato = 12   
forza_sfumatura = 3.5

# --- GESTIONE PULIZIA CARTELLE ---
cartella_heatmaps = os.path.join(cartella_base, "Heatmaps_Generate")
if os.path.exists(cartella_heatmaps): shutil.rmtree(cartella_heatmaps)
os.makedirs(cartella_heatmaps)

cartella_overlays = os.path.join(cartella_base, "Overlays_Cellule")
if os.path.exists(cartella_overlays): shutil.rmtree(cartella_overlays)
os.makedirs(cartella_overlays)

print("1. Inizializzazione AI (StarDist)...")
model = StarDist2D.from_pretrained('2D_versatile_he')

riepilogo_batch = []
files = [f for f in os.listdir(cartella_immagini) if f.endswith(estensione_file)]
print(f"Trovate {len(files)} immagini.\n")
tempo_inizio_totale = time.time()

# Il nostro punto di riferimento nello spazio 3D dei colori (Il Vetro Vuoto)
bianco_puro = np.array([255, 255, 255])

# =========================================================================
# 5. CICLO PRINCIPALE SUI VETRINI
# =========================================================================
for i, nome_file in enumerate(files, 1):
    percorso_completo = os.path.join(cartella_immagini, nome_file)
    print(f"[{i}/{len(files)}] Analizzando: {nome_file}...")
    
    try:
        img = io.imread(percorso_completo)
        altezza, larghezza, _ = img.shape
        
        # --- CALCOLO SOGLIE DINAMICHE AUTOMATICO ---
        img_thumb = img[::10, ::10, :]
        
        # Estraiamo parametri dell'immagine corrente per calcolare le soglie
        distanza_colore_thumb = np.linalg.norm(img_thumb - bianco_puro, axis=2)
        luce_thumb = np.mean(img_thumb, axis=2)
        std_thumb = np.std(img_thumb, axis=2)
        
        try:
            # 1. Distanza dal bianco (tessuto vs sfondo)
            soglia_distanza = threshold_otsu(distanza_colore_thumb) * 0.8
            
            # 2. Soglia Luminosità (sostituisce il 248)
            otsu_luce = threshold_otsu(luce_thumb)
            soglia_luce_max = otsu_luce + ((255 - otsu_luce) * 0.6)
            
            # 3. Soglia Grigio/Polvere (sostituisce il 2)
            soglia_std_min = threshold_otsu(std_thumb) * 0.3
        except:
            # valori di sicurezza: se l'immagine è tutta bianca o corrotta, usa i valori fissi
            soglia_distanza = 20
            soglia_luce_max = 248
            soglia_std_min = 2.0
            
        dati_cellule = []
        risultati_segmentazione = []
        
        for y in range(0, altezza, dimensione_tile):
            for x in range(0, larghezza, dimensione_tile):
                y_fine = min(y + dimensione_tile, altezza)
                x_fine = min(x + dimensione_tile, larghezza)
                tile_img = img[y:y_fine, x:x_fine, :]
                
                # --- 1. FILTRO ARTEFATTI ADATTIVO ---
                if tile_img.size == 0 or np.mean(tile_img) > soglia_luce_max: 
                    continue
                
                # Scarta la polvere in base alla vividezza reale di questo specifico vetrino
                if np.mean(np.std(tile_img, axis=2)) < soglia_std_min:
                    continue
                
                # --- 2. MASCHERA TESSUTO LOCALE ---
                distanza_colore_tile = np.linalg.norm(tile_img - bianco_puro, axis=2)
                maschera_tile = distanza_colore_tile > soglia_distanza 
                
                maschera_tile = morphology.remove_small_objects(maschera_tile, min_size=500)
                maschera_tile = morphology.binary_dilation(maschera_tile, morphology.disk(15))
                
                if not np.any(maschera_tile): 
                    continue
                
                # --- 3. INTELLIGENZA ARTIFICIALE ---
                stains = color.separate_stains(tile_img, color.hdx_from_rgb)
                dab_channel = stains[:, :, 1]
                
                tile_normalized = normalize(tile_img, 1, 99.8, axis=(0, 1))
                labels, _ = model.predict_instances(tile_normalized, prob_thresh=prob_tresh, nms_thresh=nms_tresh)
                
                if np.max(labels) == 0: continue
                
                indice_blocco = len(risultati_segmentazione)
                risultati_segmentazione.append({
                    'y': y, 'y_fine': y_fine, 'x': x, 'x_fine': x_fine,
                    'labels': labels
                })
                
                # --- 4. ESTRAZIONE DATI ---
                for cellula in measure.regionprops(labels, intensity_image=dab_channel):
                    cy, cx = int(cellula.centroid[0]), int(cellula.centroid[1])
                    if maschera_tile[cy, cx]:
                        dati_cellule.append({
                            'X': x + cx, 
                            'Y': y + cy, 
                            'Intensita_DAB': cellula.mean_intensity,
                            'ID_Cellula': cellula.label,
                            'Indice_Blocco': indice_blocco
                        })
        
        # --- ELABORAZIONE FINALE DEL VETRINO ---
        if len(dati_cellule) > 0:
            df = pd.DataFrame(dati_cellule) 
            
            # --- CLASSIFICAZIONE CELLULE: GMM ---
            if len(df) > 2:
                intensita = df['Intensita_DAB'].values.reshape(-1, 1)
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(intensita)
                etichette = gmm.predict(intensita)
                
                media_gruppo_0 = intensita[etichette == 0].mean()
                media_gruppo_1 = intensita[etichette == 1].mean()
                
                if media_gruppo_1 > media_gruppo_0:
                    mask_pos_gmm = (etichette == 1); mask_neg_gmm = (etichette == 0)
                else:
                    mask_pos_gmm = (etichette == 0); mask_neg_gmm = (etichette == 1)
                    
                if mask_pos_gmm.any() and mask_neg_gmm.any():
                    soglia_base_gmm = (intensita[mask_pos_gmm].min() + intensita[mask_neg_gmm].max()) / 2.0
                else:
                    soglia_base_gmm = intensita.mean()
            else:
                soglia_base_gmm = df['Intensita_DAB'].mean()
            
            soglia_sensibile = soglia_base_gmm * sensibilita_dab
            df['Positiva'] = (df['Intensita_DAB'] > soglia_sensibile).astype(int)
            
            tot = len(df)
            pos = df['Positiva'].sum()
            ki67_li = (pos / tot * 100) if tot > 0 else 0
            print(f"   -> Cellule Valide: {tot} | Positive: {pos} | Ki-67: {ki67_li:.2f}%")
           
            riepilogo_batch.append({
                "Nome_Vetrino": nome_file, "Cellule_Totali": tot, "Positive": pos,
                "Ki67_LI_Percentuale": round(ki67_li, 2)
            })
            
            # --- VISUALIZZAZIONE OVERLAYS CELLULARI E HEATMAPS ---
            overlay_contorni = np.zeros((altezza, larghezza, 4), dtype=np.uint8)

            for i, blocco in enumerate(risultati_segmentazione):
                labels = blocco['labels']
                df_blocco = df[df['Indice_Blocco'] == i]
                if df_blocco.empty: continue
                
                confini_base = find_boundaries(labels, mode='thick')
                confini_tile = morphology.binary_dilation(confini_base, morphology.disk(2))
                
                max_id = labels.max()
                mappa_r = np.zeros(max_id + 1, dtype=np.uint8)
                mappa_b = np.zeros(max_id + 1, dtype=np.uint8)
                
                id_pos = df_blocco[df_blocco['Positiva'] == 1]['ID_Cellula'].values
                id_neg = df_blocco[df_blocco['Positiva'] == 0]['ID_Cellula'].values
                
                mappa_r[id_pos] = 255 
                mappa_b[id_neg] = 255 
                
                colori_tile = np.zeros((labels.shape[0], labels.shape[1], 4), dtype=np.uint8)
                colori_tile[:, :, 0] = mappa_r[labels]
                colori_tile[:, :, 2] = mappa_b[labels]
                
                maschera_valide = np.isin(labels, df_blocco['ID_Cellula'].values)
                confini_validi = confini_tile & maschera_valide
                
                colori_tile[:, :, 3] = np.where(confini_validi, 255, 0).astype(np.uint8) 
                colori_tile[~confini_validi] = [0, 0, 0, 0]
                
                overlay_contorni[blocco['y']:blocco['y_fine'], blocco['x']:blocco['x_fine']] = colori_tile

            # DISEGNO OVERLAY
            fig_ov, ax_ov = plt.subplots(figsize=(12, 10))
            ax_ov.imshow(img)
            ax_ov.imshow(overlay_contorni)
            ax_ov.set_title(f"Evidenziazione Cellule - {nome_file} (Ki-67: {ki67_li:.2f}%)", fontsize=14)
            ax_ov.axis('off')
            legend_elements = [Line2D([0], [0], color='red', lw=2, label='Positive'),
                               Line2D([0], [0], color='blue', lw=2, label='Negative')]
            ax_ov.legend(handles=legend_elements, loc='upper right')
            nome_overlay = nome_file.replace(estensione_file, "_Evidenziazione_Cellule.png")
            plt.savefig(os.path.join(cartella_overlays, nome_overlay), dpi=300, bbox_inches='tight')
            plt.close(fig_ov)
            
           # --- GENERAZIONE HEATMAP FINALE ---
            nx_bins = int(larghezza / dimensione_quadrato)
            ny_bins = int(altezza / dimensione_quadrato)
            
            grid_count, x_edge, y_edge, _ = binned_statistic_2d(df['X'], df['Y'], None, statistic='count', bins=[nx_bins, ny_bins], range=[[0, larghezza], [0, altezza]])
            grid_pos, _, _, _ = binned_statistic_2d(df['X'], df['Y'], df['Positiva'], statistic='sum', bins=[nx_bins, ny_bins], range=[[0, larghezza], [0, altezza]])
            
            grid_count = grid_count.T; grid_pos = grid_pos.T
            
            # Maschera Visiva per le mappe
            mask_visiva = grid_count > 0
            mask_visiva = morphology.binary_closing(mask_visiva, morphology.disk(raggio_chiusura_vis))
            mask_visiva = morphology.remove_small_objects(mask_visiva, min_size=area_minima_isola_vis)
            if espansione_tessuto_vis > 0: 
                mask_visiva = morphology.binary_dilation(mask_visiva, morphology.disk(espansione_tessuto_vis))
            
            count_smooth = gaussian_filter(grid_count, sigma=forza_sfumatura)
            pos_smooth = gaussian_filter(grid_pos, sigma=forza_sfumatura)
            
            LI_grid = np.divide(pos_smooth, count_smooth, out=np.zeros_like(pos_smooth), where=count_smooth > 0.01)
            LI_grid[~mask_visiva] = np.nan

            X, Y = np.meshgrid((x_edge[:-1] + x_edge[1:]) / 2, (y_edge[:-1] + y_edge[1:]) / 2)

            fig, ax = plt.subplots(figsize=(12, 10))
            ax.set_facecolor('white')
            heatmap = ax.pcolormesh(X, Y, LI_grid, cmap='turbo', vmin=0, vmax=0.6, shading='auto')
            ax.set_aspect('equal'); ax.invert_yaxis() 
            ax.set_title(f"Ki-67 LI Map - {nome_file}", fontsize=16)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(heatmap, ax=ax, orientation='horizontal', pad=0.05, fraction=0.046)
            
            plt.savefig(os.path.join(cartella_heatmaps, nome_file.replace(estensione_file, "_Heatmap_LI.png")), dpi=300, bbox_inches='tight')
            plt.close(fig)          
 
            
        else:
            print("   -> Nessuna cellula rilevata. Vetrino vuoto.")
            
    except Exception as e:
        print(f"   -> ERRORE sul file {nome_file}: {e}")

# =========================================================================
# 6. ESPORTAZIONE DATI GLOBALI
# =========================================================================
if len(riepilogo_batch) > 0:
    df_riepilogo = pd.DataFrame(riepilogo_batch)
    
    totale_cellule_globali = df_riepilogo['Cellule_Totali'].sum()
    somma_indici_pesati = (df_riepilogo['Ki67_LI_Percentuale'] * df_riepilogo['Cellule_Totali']).sum()
    media_pesata_totale = somma_indici_pesati / totale_cellule_globali if totale_cellule_globali > 0 else 0
    
    riga_riepilogo = {
        "Nome_Vetrino": "MEDIA PESATA TOTALE", 
        "Cellule_Totali": totale_cellule_globali,
        "Positive": df_riepilogo['Positive'].sum(),
        "Ki67_LI_Percentuale": round(media_pesata_totale, 2)
    }
    df_riepilogo = pd.concat([df_riepilogo, pd.DataFrame([riga_riepilogo])], ignore_index=True)

    media_semplice = df_riepilogo['Ki67_LI_Percentuale'].mean()
    riga_media_semplice = {
        "Nome_Vetrino": "MEDIA ARITMETICA SEMPLICE", 
        "Cellule_Totali": "-", 
        "Positive": "-",
        "Ki67_LI_Percentuale": round(media_semplice, 2)
    }
    df_riepilogo = pd.concat([df_riepilogo, pd.DataFrame([riga_media_semplice])], ignore_index=True)
    
    percorso_csv = os.path.join(cartella_base, "Tabella_Risultati_Ki67.csv")
    df_riepilogo.to_csv(percorso_csv, index=False, sep=';', decimal=',')
    print(f"\n Report salvato con Media Pesata Totale: {media_pesata_totale:.2f}%")

tempo_fine = time.time()
tempo_impiegato = tempo_fine - tempo_inizio_totale
print(f"\n=== PIPELINE COMPLETATA IN {int(tempo_impiegato // 60)} MINUTI E {int(tempo_impiegato % 60)} SECONDI ===")
