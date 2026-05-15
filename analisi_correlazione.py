import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

print(" Avvio dell'Analisi di Correlazione Statistica (Spearman)...")

# --- 1. LETTURA E PULIZIA DATI ALGORITMO (IL TUO CODICE PYTHON) ---
# Puntiamo al file CSV generato dal tuo script!
percorso_algoritmo = r"C:\Users\rocca\Desktop\corso progetto\PYTHON_TEST\Tabella_Risultati_Ki67.csv"
df_alg = pd.read_csv(percorso_algoritmo, sep=';', decimal=',') # Usiamo decimal=',' perché l'hai salvato in formato europeo

# Scartiamo le righe di riepilogo finali
df_alg = df_alg[~df_alg['Nome_Vetrino'].isin(['MEDIA PESATA TOTALE', 'MEDIA ARITMETICA SEMPLICE'])].copy()

# Puliamo i nomi per l'unione
df_alg['slide_id'] = df_alg['Nome_Vetrino'].astype(str).str.replace('.tif', '', regex=False)
df_alg = df_alg.drop_duplicates(subset=['slide_id'], keep='last')

# Estraiamo i valori
df_alg['Ki67_Algoritmo'] = df_alg['Ki67_LI_Percentuale'].astype(float)


# --- 2. LETTURA E PULIZIA DATI MANUALI (QUPATH) ---
# Usiamo il file misure.csv originale che avevi nella cartella di test
percorso_manuale = r"C:\Users\rocca\Desktop\corso progetto\PYTHON_TEST\misure.csv"

# Gestione intelligente del separatore
try:
    df_man = pd.read_csv(percorso_manuale, sep=';')
    if 'Image' not in df_man.columns:
        df_man = pd.read_csv(percorso_manuale, sep=',')
except Exception:
    # Fallback se il file è in un'altra cartella
    df_man = pd.read_csv(r"C:\Users\rocca\Desktop\corso progetto\QUPATH_TEST\Ki-67 segmenti project\misure.csv", sep=';')

col_nome_man = 'Image' if 'Image' in df_man.columns else df_man.columns[0]
df_man['slide_id'] = df_man[col_nome_man].astype(str).str.replace('.tif', '', regex=False)
df_man = df_man.drop_duplicates(subset=['slide_id'], keep='last')

col_val_man = 'Positive %' if 'Positive %' in df_man.columns else df_man.columns[1]

# Convertiamo in numero, gestendo eventuali virgole di QuPath
if df_man[col_val_man].dtype == object:
    df_man['Ki67_Manuale'] = df_man[col_val_man].astype(str).str.replace(',', '.').astype(float)
else:
    df_man['Ki67_Manuale'] = df_man[col_val_man].astype(float)


# --- 3. MERGE (UNIONE) DEI DATI ---
df_unito = pd.merge(df_alg[['slide_id', 'Ki67_Algoritmo']], 
                    df_man[['slide_id', 'Ki67_Manuale']], 
                    on='slide_id', how='inner')

print(f" Vetrini combinati con successo: {len(df_unito)}\n")

# --- 4. CALCOLO STATISTICO E VALUTAZIONE (SPEARMAN) ---
rho_stat, p_value = spearmanr(df_unito['Ki67_Manuale'], df_unito['Ki67_Algoritmo'])

print(" RISULTATI STATISTICI:")
print(f"Correlazione di Spearman (rho) calcolata: {rho_stat:.4f}")

# Valutazione logica dell'indice rho
if rho_stat >= 0.90:
    print(" ➔ Esito: ECCELLENTE. L'algoritmo e il conteggio manuale sono quasi perfettamente allineati nei ranghi.")
elif rho_stat >= 0.75:
    print(" ➔ Esito: BUONO. C'è una forte correlazione monotonica, l'algoritmo si comporta in modo affidabile.")
elif rho_stat >= 0.50:
    print(" ➔ Esito: MODERATO. L'algoritmo segue il trend generale, ma ci sono discrepanze.")
else:
    print(" ➔ Esito: DEBOLE. I due metodi restituiscono ordinamenti molto diversi.")

print(f"\nP-value calcolato: {p_value:.4e}")

if p_value < 0.05:
    print(" ➔ Esito: SIGNIFICATIVO. Il risultato è statisticamente valido e non dovuto al caso.")
else:
    print(" ➔ Esito: NON SIGNIFICATIVO. I dati non sono sufficienti per trarre una conclusione statistica sicura.")

# --- 5. GENERAZIONE DEL GRAFICO (SCATTER PLOT) ---
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 8))

sns.regplot(data=df_unito, x='Ki67_Manuale', y='Ki67_Algoritmo', 
            scatter_kws={'alpha':0.6, 'color':"#9159b6"}, 
            line_kws={'color':'red', 'linewidth':2})

max_val = max(df_unito['Ki67_Manuale'].max(), df_unito['Ki67_Algoritmo'].max())
plt.plot([0, max_val], [0, max_val], linestyle='--', color='gray', label='Accordo Perfetto (x=y)')

# --- ARROTONDAMENTO E CONTROLLO SOGLIA P-VALUE ---
# Impostiamo un "pavimento" (floor) a 0.001. 
p_visualizzato = max(p_value, 0.01)

if p_value < 0.01:
    testo_stat = f"Spearman \u03c1 = {rho_stat:.2f}\np-value = {p_visualizzato:.2f}"
else:
    testo_stat = f"Spearman \u03c1 = {rho_stat:.2f}\np-value = {p_visualizzato:.2f}"

plt.text(0.05 * max_val, 0.9 * max_val, testo_stat, fontsize=12, 
         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

plt.title("Correlazione Ki-67 (Spearman): Manuale vs Algoritmo Python", fontsize=16, fontweight='bold')
plt.xlabel("Ki-67 LI (%) - Conteggio Manuale (QuPath)", fontsize=12, fontweight='bold')
plt.ylabel("Ki-67 LI (%) - Algoritmo Python (StarDist)", fontsize=12, fontweight='bold')
plt.xlim(0, max_val + 5)
plt.ylim(0, max_val + 5)
plt.legend(loc='lower right')

plt.tight_layout()

# --- 6. SALVATAGGIO ---
nome_immagine = "ScatterPlot_Spearman_Ki67.png"
plt.savefig(nome_immagine, dpi=300, bbox_inches='tight')
print(f"\n🎉 Grafico salvato con successo come: {nome_immagine}")
