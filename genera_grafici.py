import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Disabilitiamo eventuali warning grafici di Seaborn
warnings.filterwarnings("ignore")

# --- 1. CONFIGURAZIONE PERCORSI ---
# Otteniamo la cartella di lavoro corrente
cartella_base = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

percorso_auto = os.path.join(cartella_base, "Tabella_Risultati_Ki67.csv")
percorso_man = os.path.join(cartella_base, "misure.csv")

print("Lettura e sincronizzazione dei file in corso...")

# --- 2. CARICAMENTO E PULIZIA DEI DATI ---
# Controllo preliminare di esistenza dei file
if not os.path.exists(percorso_auto):
    print(f"❌ Errore: File automatizzato non trovato -> {percorso_auto}")
    exit(1)
if not os.path.exists(percorso_man):
    print(f"❌ Errore: File manuale non trovato -> {percorso_man}")
    exit(1)

# Lettura del file automatizzato
df_auto = pd.read_csv(percorso_auto, sep=';', decimal=',')
# Rimuoviamo le righe riassuntive testuali (es. MEDIA PESATA TOTALE)
df_auto = df_auto[~df_auto['Nome_Vetrino'].astype(str).str.contains('MEDIA', case=False)].copy()
df_auto = df_auto.rename(columns={'Nome_Vetrino': 'Image', 'Ki67_LI_Percentuale': 'Ki67_Auto'})
df_auto['Ki67_Auto'] = pd.to_numeric(df_auto['Ki67_Auto'], errors='coerce')

# Lettura del file manuale
df_man = pd.read_csv(percorso_man, sep=';', decimal=',')
# La colonna 'Positive %' in misure.csv adotta il punto come separatore decimale
df_man['Ki67_Manuale'] = pd.to_numeric(df_man['Positive %'], errors='coerce')

# Unione sincronizzata basata sulla colonna identificativa dell'immagine
df_merged = pd.merge(df_auto[['Image', 'Ki67_Auto']], df_man[['Image', 'Ki67_Manuale']], on='Image')
df_merged = df_merged.dropna(subset=['Ki67_Auto', 'Ki67_Manuale']).reset_index(drop=True)

print(f" Sincronizzazione completata: trovate {len(df_merged)} immagini in comune.")

# Esportazione della tabella comparativa unita per consultazione immediata
percorso_export = os.path.join(cartella_base, "Confronto_Ki67_Merged.csv")
df_merged.to_csv(percorso_export, index=False, sep=';', decimal=',')
print(" Tabella comparativa unita salvata in: Confronto_Ki67_Merged.csv")

# --- 3. IMPOSTAZIONE GRAFICA GENERALE ---
sns.set_theme(style="whitegrid")

# ==============================================================================
# GRAFICO 1: BOXPLOT AUTOMATIZZATO
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(y=df_merged['Ki67_Auto'], color="#2ecc71", width=0.3, fliersize=5, ax=ax)
# Aggiunta dei punti singoli semi-trasparenti per visualizzare la densità
sns.stripplot(y=df_merged['Ki67_Auto'], color="#27ae60", alpha=0.5, jitter=0.1, size=6, ax=ax)

ax.set_title("Boxplot Indice Ki-67\nAnalisi Automatizzata", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Ki-67 LI (%)", fontsize=12)
ax.set_ylim(0, 105)

percorso_box_auto = os.path.join(cartella_base, "Grafico_Boxplot_Automatizzato.png")
plt.tight_layout()
plt.savefig(percorso_box_auto, dpi=300, bbox_inches='tight')
plt.close(fig)
print("📊 Generato e salvato: Grafico_Boxplot_Automatizzato.png")

# ==============================================================================
# GRAFICO 2: BOXPLOT MANUALE
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 6))
sns.boxplot(y=df_merged['Ki67_Manuale'], color="#3498db", width=0.3, fliersize=5, ax=ax)
sns.stripplot(y=df_merged['Ki67_Manuale'], color="#2980b9", alpha=0.5, jitter=0.1, size=6, ax=ax)

ax.set_title("Boxplot Indice Ki-67\nConteggio Manuale", fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel("Ki-67 LI (%)", fontsize=12)
ax.set_ylim(0, 105)

percorso_box_man = os.path.join(cartella_base, "Grafico_Boxplot_Manuale.png")
plt.tight_layout()
plt.savefig(percorso_box_man, dpi=300, bbox_inches='tight')
plt.close(fig)
print(" Generato e salvato: Grafico_Boxplot_Manuale.png")

# ==============================================================================
# GRAFICO 3: ISTOGRAMMA / BARPLOT FASCE CLINICHE A CONFRONTO
# ==============================================================================
# Suddivisione fasce: 0-3%, 3-20%, 20-50%, >50%
limiti_fasce = [-1, 3, 20, 50, 105]
etichette_fasce = ['0-3%', '3-20%', '20-50%', '>50%']

df_merged['Fascia_Auto'] = pd.cut(df_merged['Ki67_Auto'], bins=limiti_fasce, labels=etichette_fasce, right=True)
df_merged['Fascia_Manuale'] = pd.cut(df_merged['Ki67_Manuale'], bins=limiti_fasce, labels=etichette_fasce, right=True)

# Conteggio per ciascun metodo
conteggio_auto = df_merged['Fascia_Auto'].value_counts(sort=False).rename('Automatizzato')
conteggio_manuale = df_merged['Fascia_Manuale'].value_counts(sort=False).rename('Manuale')

# Trasformazione del layout per abilitare le barre affiancate su Seaborn
df_fasce = pd.concat([conteggio_manuale, conteggio_auto], axis=1).reset_index()
df_fasce.columns = ['Fascia Clinica', 'Manuale', 'Automatizzato']
df_fasce_melted = df_fasce.melt(id_vars='Fascia Clinica', var_name='Metodo', value_name='Numero Vetrini')

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    data=df_fasce_melted, 
    x='Fascia Clinica', 
    y='Numero Vetrini', 
    hue='Metodo', 
    palette=['#3498db', '#2ecc71'],
    ax=ax
)

ax.set_title("Distribuzione Campioni per Fasce Cliniche Ki-67\nConfronto Manuale vs Automatizzato", fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel("Fasce Cliniche Ki-67", fontsize=12)
ax.set_ylabel("Numero di Vetrini", fontsize=12)

# Scrittura dinamica dei conteggi al di sopra delle rispettive barre
for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=11, fontweight='semibold')

percorso_fasce = os.path.join(cartella_base, "Istogramma_Fasce_Cliniche_Confronto.png")
plt.tight_layout()
plt.savefig(percorso_fasce, dpi=300, bbox_inches='tight')
plt.close(fig)
print(" Generato e salvato: Istogramma_Fasce_Cliniche_Confronto.png")

print("\n L'analisi comparativa è stata completata e i file grafici sono pronti all'uso!")
