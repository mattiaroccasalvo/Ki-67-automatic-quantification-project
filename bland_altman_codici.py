import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print(" Preparazione dati per il Bland-Altman Plot in corso...")

# --- 1. LETTURA E PULIZIA DEI DATI ---
# A. Dati Manuali (QuPath)
df_manuale = pd.read_csv(r"C:\Users\rocca\Desktop\corso progetto\QUPATH_TEST\Ki-67 segmenti project\misure.csv", sep=';', decimal='.')
df_manuale['slide_id'] = df_manuale['Image'].str.replace('.tif', '', regex=False)
df_manuale = df_manuale.drop_duplicates(subset=['slide_id'], keep='last')

# B. Dati del Tuo Codice Python
df_codice = pd.read_csv(r"C:\Users\rocca\Desktop\corso progetto\PYTHON_TEST\Tabella_Risultati_Ki67.csv", sep=';', decimal=',')
# (Opzionale ma consigliato: togli l'eventuale riga della media se presente)
df_codice = df_codice[df_codice['Nome_Vetrino'].astype(str).str.contains('MEDIA') == False]
# 2. CREA LA COLONNA 'slide_id' PRIMA di cercare i doppioni
df_codice['slide_id'] = df_codice['Nome_Vetrino'].str.replace('.tif', '', regex=False)
df_codice = df_codice.drop_duplicates(subset=['slide_id'], keep='last')

# --- 1.5 UNIONE DELLE TABELLE ---
# Uniamo le tabelle per i vetrini in comune
df_confronto = pd.merge(df_manuale[['slide_id', 'Positive %']], 
                        df_codice[['slide_id', 'Ki67_LI_Percentuale']], 
                        on='slide_id')

# --- 2. CALCOLI STATISTICI PER BLAND-ALTMAN ---

# 1. Creiamo due nuove colonne sicure, forzando i numeri decimali corretti
df_confronto['Manuale_Num'] = df_confronto['Positive %'].astype(str).str.replace(',', '.').astype(float)
df_confronto['Codice_Num'] = df_confronto['Ki67_LI_Percentuale'].astype(str).str.replace(',', '.').astype(float)

# 2. Calcoliamo la differenza provvisoria
df_confronto['Differenza_Provvisoria'] = df_confronto['Codice_Num'] - df_confronto['Manuale_Num']

df_confronto = df_confronto[df_confronto['Differenza_Provvisoria'].abs() <= 50]

print(f" Vetrini rimasti dopo aver escluso gli errori > 50%: {len(df_confronto)}")

# 4. Estraiamo i dati filtrati per il grafico
manuale = df_confronto['Manuale_Num']
mio_codice = df_confronto['Codice_Num']

# Calcoli standard del Bland-Altman sui dati puliti
media_valori = (manuale + mio_codice) / 2
differenza = mio_codice - manuale

# Calcolo del Bias (Errore medio) e dei Limiti di Confidenza (95%)
bias = differenza.mean()
dev_std = differenza.std()
limite_sup = bias + (1.96 * dev_std)
limite_inf = bias - (1.96 * dev_std)

print(f"✅ Bias ricalcolato senza valori estremi: {bias:.2f}%")

# --- 3. GENERAZIONE DEL GRAFICO ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Disegniamo i pallini per ogni vetrino
sns.scatterplot(x=media_valori, y=differenza, s=100, alpha=0.8, color="#9159b6", edgecolor='black')

# Linea dello ZERO (Accordo perfetto: nessuna differenza)
plt.axhline(0, color='black', linestyle='-', linewidth=1.5)

# Linea del BIAS (Errore medio del codice)
plt.axhline(bias, color='red', linestyle='--', linewidth=2, label=f'Bias Medio: {bias:.2f}%')

# Linee dei LIMITI DI CONFIDENZA (+/- 1.96 Deviazioni Standard)
plt.axhline(limite_sup, color='gray', linestyle=':', linewidth=2, label=f'Limite Sup. (+1.96 SD): {limite_sup:.2f}%')
plt.axhline(limite_inf, color='gray', linestyle=':', linewidth=2, label=f'Limite Inf. (-1.96 SD): {limite_inf:.2f}%')

# --- 4. DETTAGLI ESTETICI E SALVATAGGIO ---
plt.title("Bland-Altman Plot: Conteggio Manuale vs Algoritmo ", fontsize=15, fontweight='bold')
plt.xlabel("Media delle due misurazioni (Manuale + Codice) / 2  (%)", fontsize=12)
plt.ylabel("Differenza (Codice - Manuale)  (%)", fontsize=12)

# Spostiamo la legenda per non coprire i dati
plt.legend(loc='upper right', bbox_to_anchor=(1.30, 1.15), fontsize=10)

plt.tight_layout()

# Salvataggio in alta definizione
cartella_destinazione = r"C:\Users\rocca\Desktop\corso progetto\PYTHON_TEST"
percorso_immagine = os.path.join(cartella_destinazione, "Bland_Altman_Algoritmo_vs_Manuale.png")
plt.savefig(percorso_immagine, dpi=300)
print(f" Grafico generato e salvato in: {percorso_immagine}")
