import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, ttest_ind, f_oneway, chi2_contingency

df = pd.read_csv("demandes_complet.csv")
n = len(df)

print("=" * 60)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 60)


moyenne = df["temps_traitement"].mean()
ecart_type = df["temps_traitement"].std()
median = df["temps_traitement"].median()

t_crit = stats.t.ppf(0.975, n-1)
me = t_crit * (ecart_type / np.sqrt(n))
print(f"\n Temps moyen : {moyenne:.2f}h [{moyenne-me:.2f}, {moyenne+me:.2f}]")
print(f" Écart-type : {ecart_type:.2f}h")
print(f" Médiane : {median:.2f}h")


print("\n Demandes par type :")
print(df["type_demande"].value_counts())

print("\n Demandes par service :")
print(df["service"].value_counts())

print("\n Demandes par priorité :")
print(df["priorite"].value_counts())


print("\n" + "=" * 60)
print(" MOYENNES PAR GROUPE (avec IC 95%)")
print("=" * 60)

for var in ["type_demande", "service", "priorite"]:
    print(f"\n Par {var} :")
    for cat in df[var].unique():
        data = df[df[var] == cat]["temps_traitement"]
        m = data.mean()
        s = data.std()
        n_cat = len(data)
        t = stats.t.ppf(0.975, n_cat-1)
        me_cat = t * (s / np.sqrt(n_cat))
        print(f"   {cat}: {m:.2f}h [{m-me_cat:.2f}, {m+me_cat:.2f}] (n={n_cat})")

print("\n" + "=" * 60)
print(" TESTS D'HYPOTHÈSES")
print("=" * 60)

stat, p = ttest_1samp(df["temps_traitement"], 24)
print(f"\n H0: μ = 24h")
print(f"   t = {stat:.4f}, p = {p:.4f}")
print(f"   {' Rejet H0' if p < 0.05 else ' Garde H0'}")


it = df[df["service"] == "IT"]["temps_traitement"]
fin = df[df["service"] == "Finance"]["temps_traitement"]
stat, p = ttest_ind(it, fin)
print(f"\n IT vs Finance")
print(f"   IT: {it.mean():.2f}h vs Finance: {fin.mean():.2f}h")
print(f"   p = {p:.4f} → {' Différence significative' if p < 0.05 else ' Pas de différence'}")


groups = [df[df["service"] == s]["temps_traitement"] for s in df["service"].unique()]
stat, p = f_oneway(*groups)
print(f"\n ANOVA (services)")
print(f"   F = {stat:.4f}, p = {p:.4f}")
print(f"   {' Au moins un service diffère' if p < 0.05 else ' Pas de différence'}")


contingence = pd.crosstab(df["type_demande"], df["priorite"])
stat, p, dof, expected = chi2_contingency(contingence)
print(f"\n Chi-2 (type vs priorité)")
print(f"   χ² = {stat:.4f}, p = {p:.4f}")
print(f"   {' Dépendance significative' if p < 0.05 else 'Indépendance'}")


plt.figure(figsize=(16, 12))


plt.subplot(3, 3, 1)
df["temps_traitement"].hist(bins=25, color="skyblue", edgecolor="black")
plt.axvline(moyenne, color="red", linestyle="--", label=f"Moyenne={moyenne:.1f}h")
plt.title("Distribution des temps")
plt.legend()

# Boxplot par type
plt.subplot(3, 3, 2)
sns.boxplot(data=df, x="type_demande", y="temps_traitement")
plt.title("Temps par type")
plt.xticks(rotation=45)

# Boxplot par service
plt.subplot(3, 3, 3)
sns.boxplot(data=df, x="service", y="temps_traitement")
plt.title("Temps par service")

# Boxplot par priorité
plt.subplot(3, 3, 4)
sns.boxplot(data=df, x="priorite", y="temps_traitement", order=["faible", "moyenne", "haute"])
plt.title("Temps par priorité")

# Moyennes avec IC
plt.subplot(3, 3, 5)
services = df["service"].unique()
moyennes = []
err_bas = []
err_haut = []
for s in services:
    data = df[df["service"] == s]["temps_traitement"]
    m = data.mean()
    me = stats.t.ppf(0.975, len(data)-1) * (data.std()/np.sqrt(len(data)))
    moyennes.append(m)
    err_bas.append(m - me)
    err_haut.append(m + me)

plt.bar(services, moyennes, color="coral", alpha=0.7)
plt.errorbar(services, moyennes, 
             yerr=[np.array(moyennes)-np.array(err_bas), 
                   np.array(err_haut)-np.array(moyennes)],
             fmt="none", color="black", capsize=5)
plt.title("Moyennes par service (IC 95%)")
plt.xticks(rotation=45)

#  Heatmap type vs priorité
plt.subplot(3, 3, 6)
heatmap_data = df.pivot_table(values="temps_traitement", index="type_demande", 
                               columns="priorite", aggfunc="mean")
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd")
plt.title("Temps moyen : Type × Priorité")

#  Scatter utilisateurs vs temps
plt.subplot(3, 3, 7)
plt.scatter(df["utilisateurs_impactes"], df["temps_traitement"], alpha=0.5, c="green")
plt.xlabel("Utilisateurs impactés")
plt.ylabel("Temps (h)")
plt.title("Impact vs Temps")

#  QQ-plot
plt.subplot(3, 3, 8)
stats.probplot(df["temps_traitement"], dist="norm", plot=plt)
plt.title("QQ-Plot (Normalité)")

#  Corrélation
plt.subplot(3, 3, 9)
corr_vars = ["temps_traitement", "utilisateurs_impactes", "heure_creation"]
corr_matrix = df[corr_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0)
plt.title("Matrice de corrélation")

plt.tight_layout()
plt.savefig("analyse_complete.png", dpi=150)
plt.show()
print("\n📊 Graphiques sauvegardés")
