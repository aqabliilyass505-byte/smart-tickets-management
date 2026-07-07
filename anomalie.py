import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════
# 1. CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("🔍 DÉTECTION D'ANOMALIES - TICKETS ATYPIQUES")
print("=" * 70)

df = pd.read_csv("demandes_nlp.csv")

print(f"\n📊 Dataset : {len(df)} tickets")
print(f"📋 Variables disponibles : {df.columns.tolist()}")

# ═══════════════════════════════════════════════════════════════════
# 2. PRÉPARATION DES FEATURES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("2️⃣ PRÉPARATION DES FEATURES")
print("─" * 70)

# Features numériques pour la détection
features = ["temps_traitement", "utilisateurs_impactes", "heure_creation"]

# Encodage des variables catégorielles (pour enrichir la détection)
df_encoded = pd.get_dummies(df[["type_demande", "service", "priorite"]], drop_first=True)
X = pd.concat([df[features], df_encoded], axis=1)

print(f"\n📊 Features utilisées :")
for f in features:
    print(f"   • {f}")
print(f"   • Variables catégorielles encodées : {df_encoded.shape[1]} colonnes")
print(f"   • Total features : {X.shape[1]}")

# Standardisation (important pour IsolationForest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ═══════════════════════════════════════════════════════════════════
# 3. ENTRAÎNEMENT DU MODÈLE
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("3️⃣ ENTRAÎNEMENT ISOLATION FOREST")
print("─" * 70)

# contamination=0.05 : on suppose 5% de tickets anormaux
iso = IsolationForest(
    contamination=0.05,      # 5% d'anomalies attendues
    random_state=42,
    n_estimators=100         # Nombre d'arbres
)

# Entraînement et prédiction
df["anomalie"] = iso.fit_predict(X_scaled)
df["anomalie_score"] = iso.decision_function(X_scaled)  # Score d'anomalie (plus négatif = plus anormal)

# -1 = anomalie, 1 = normal
df["est_anomalie"] = df["anomalie"].apply(lambda x: "ANOMALIE" if x == -1 else "Normal")

print(f"\n✅ Modèle entraîné")
print(f"   • {len(df[df['anomalie'] == -1])} tickets détectés comme ANOMALIES ({0.05*100:.0f}%)")
print(f"   • {len(df[df['anomalie'] == 1])} tickets considérés comme NORMAUX")

# ═══════════════════════════════════════════════════════════════════
# 4. ANALYSE DES ANOMALIES
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("4️⃣ ANALYSE DES ANOMALIES DÉTECTÉES")
print("─" * 70)

anomalies = df[df["anomalie"] == -1].copy()

print(f"\n📋 Top 10 des tickets les plus anormaux :")
top_anomalies = anomalies.nsmallest(10, "anomalie_score")[
    ["description", "type_demande", "service", "priorite", 
     "temps_traitement", "utilisateurs_impactes", "anomalie_score"]
]

for idx, row in top_anomalies.iterrows():
    print(f"\n   🔴 Ticket #{idx}")
    print(f"      Description : {row['description'][:60]}...")
    print(f"      Type : {row['type_demande']} | Service : {row['service']} | Priorité : {row['priorite']}")
    print(f"      ⏱️ Temps : {row['temps_traitement']:.1f}h | 👥 Utilisateurs : {row['utilisateurs_impactes']}")
    print(f"      📊 Score anomalie : {row['anomalie_score']:.4f} (plus négatif = plus anormal)")

# Comparaison anomalies vs normaux
print(f"\n📊 Comparaison Anomalies vs Normaux :")
print(f"\n{'Variable':<25} {'Normal':<15} {'Anomalie':<15} {'Écart'}")
print("-" * 60)

comp_vars = ["temps_traitement", "utilisateurs_impactes"]
for var in comp_vars:
    normal_mean = df[df["anomalie"] == 1][var].mean()
    anom_mean = df[df["anomalie"] == -1][var].mean()
    ecart = ((anom_mean - normal_mean) / normal_mean * 100) if normal_mean != 0 else 0
    print(f"{var:<25} {normal_mean:<15.2f} {anom_mean:<15.2f} {ecart:+.1f}%")

# Distribution par type
print(f"\n📋 Répartition des anomalies par type :")
anom_type = anomalies["type_demande"].value_counts()
normal_type = df[df["anomalie"] == 1]["type_demande"].value_counts()

for t in df["type_demande"].unique():
    n_anom = anom_type.get(t, 0)
    n_total = len(df[df["type_demande"] == t])
    pct = n_anom / n_total * 100 if n_total > 0 else 0
    print(f"   {t:<12s} : {n_anom:2d}/{n_total:3d} ({pct:5.1f}%) {'🔴' if pct > 10 else '🟡' if pct > 5 else '🟢'}")

# ═══════════════════════════════════════════════════════════════════
# 5. INTERPRÉTATION MÉTIER
# ═══════════════════════════════════════════════════════════════════
print("\n" + "═" * 70)
print("5️⃣ INTERPRÉTATION MÉTIER ET RECOMMANDATIONS")
print("═" * 70)

print(f"\n💬 DIAGNOSTIC :")

# Analyse des causes probables
causes = []

# Vérifier si les anomalies sont des tickets "bloqués"
temps_seuil = df["temps_traitement"].quantile(0.95)
anom_longs = anomalies[anomalies["temps_traitement"] > temps_seuil]
if len(anom_longs) > len(anomalies) * 0.5:
    causes.append(f"🕐 TICKETS BLOQUÉS : {len(anom_longs)}/{len(anomalies)} anomalies ont un temps > {temps_seuil:.0f}h")

# Vérifier si certaines priorités sont sur-représentées
prio_anom = anomalies["priorite"].value_counts(normalize=True)
prio_norm = df[df["anomalie"] == 1]["priorite"].value_counts(normalize=True)

for p in ["haute", "moyenne", "faible"]:
    if p in prio_anom.index and p in prio_norm.index:
        if prio_anom[p] > prio_norm[p] * 1.5:
            causes.append(f"🔴 PRIORITÉ {p.upper()} : sur-représentée dans les anomalies")

# Vérifier les services
serv_anom = anomalies["service"].value_counts(normalize=True)
serv_norm = df[df["anomalie"] == 1]["service"].value_counts(normalize=True)

for s in df["service"].unique():
    if s in serv_anom.index and s in serv_norm.index:
        if serv_anom[s] > serv_norm[s] * 1.5:
            causes.append(f"🏢 SERVICE {s.upper()} : plus d'anomalies que la normale")

if not causes:
    causes.append("✅ Répartition des anomalies homogène")

for c in causes:
    print(f"   {c}")

print(f"\n📋 RECOMMANDATIONS :")
print(f"   1. 🔍 Vérifier manuellement les {len(anomalies)} tickets marqués comme anormaux")
print(f"   2. 📞 Contacter les utilisateurs si ticket bloqué depuis > {temps_seuil:.0f}h")
print(f"   3. 📊 Analyser les processus des services sur-représentés")
print(f"   4. 🔄 Mettre en place une alerte automatique quotidienne")

# ═══════════════════════════════════════════════════════════════════
# 6. VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("6️⃣ VISUALISATIONS")
print("─" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Scatter plot : Temps vs Utilisateurs (coloré par anomalie)
ax1 = axes[0, 0]
normaux = df[df["anomalie"] == 1]
anoms = df[df["anomalie"] == -1]

ax1.scatter(normaux["utilisateurs_impactes"], normaux["temps_traitement"], 
           c="blue", alpha=0.3, s=30, label=f"Normal ({len(normaux)})")
ax1.scatter(anoms["utilisateurs_impactes"], anoms["temps_traitement"], 
           c="red", alpha=0.8, s=80, marker="X", label=f"Anomalie ({len(anoms)})")
ax1.set_xlabel("Utilisateurs impactés")
ax1.set_ylabel("Temps de traitement (heures)")
ax1.set_title("Détection d'anomalies : Temps vs Impact")
ax1.legend()
ax1.axhline(y=df["temps_traitement"].quantile(0.95), color="orange", 
           linestyle="--", alpha=0.5, label="Seuil 95%")

# 2. Distribution des scores d'anomalie
ax2 = axes[0, 1]
ax2.hist(df[df["anomalie"] == 1]["anomalie_score"], bins=30, alpha=0.5, 
         color="blue", label="Normal")
ax2.hist(df[df["anomalie"] == -1]["anomalie_score"], bins=30, alpha=0.8, 
         color="red", label="Anomalie")
ax2.axvline(x=0, color="black", linestyle="--")
ax2.set_xlabel("Score d'anomalie")
ax2.set_ylabel("Nombre de tickets")
ax2.set_title("Distribution des scores d'anomalie")
ax2.legend()

# 3. Boxplot par type (normaux vs anomalies)
ax3 = axes[1, 0]
df_plot = df.copy()
df_plot["Type"] = df_plot["est_anomalie"]
sns.boxplot(data=df_plot, x="type_demande", y="temps_traitement", hue="Type", ax=ax3)
ax3.set_title("Temps de traitement : Normaux vs Anomalies")
ax3.set_ylabel("Temps (heures)")
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

# 4. Heatmap des anomalies par Service × Priorité
ax4 = axes[1, 1]
heatmap_anom = df[df["anomalie"] == -1].pivot_table(
    values="temps_traitement", 
    index="service", 
    columns="priorite", 
    aggfunc="count",
    fill_value=0
)
sns.heatmap(heatmap_anom, annot=True, fmt="d", cmap="Reds", ax=ax4)
ax4.set_title("Nombre d'anomalies : Service × Priorité")

plt.tight_layout()
plt.savefig("anomalies_detection.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Graphiques sauvegardés dans 'anomalies_detection.png'")

# Sauvegarder les résultats
df.to_csv("demandes_anomalies.csv", index=False)
print(f"\n💾 Résultats sauvegardés dans 'demandes_anomalies.csv'")

print("\n" + "=" * 70)
print("🏁 FIN DE LA DÉTECTION D'ANOMALIES")
print("=" * 70)
