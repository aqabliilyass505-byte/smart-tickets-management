import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 1000
poids = np.array([0.5 if i < 10 else (0.3 if i < 50 else 0.2) for i in range(1, 201)])
poids = poids / poids.sum()

#Variables de base (indépendantes)
df = pd.DataFrame({
    "type_demande": np.random.choice(
        ["login", "materiel", "logiciel", "reseau"], 
        n, 
        p=[0.30, 0.25, 0.25, 0.20]  # login plus fréquent
    ),
    
    "service": np.random.choice(
        ["RH", "Finance", "Technique", "IT"], 
        n,
        p=[0.20, 0.25, 0.30, 0.25]
    ),
    
    # Nombre d'utilisateurs impactés (1 à 200)
     "utilisateurs_impactes": np.random.choice(
        range(1, 201), 
        n,
        p=poids
    ),
    
    # Heure de création (8h-18h en semaine)
    "heure_creation": np.random.choice(range(8, 19), n),
    "jour_semaine": np.random.choice(
        ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"], 
        n
    ),
})

#Calcul de la priorité selon les règles métier
def calcul_priorite(row):
    score = 0
    
    # Règle 1 : Type de demande
    if row["type_demande"] == "reseau":
        score += 30
    elif row["type_demande"] == "materiel":
        score += 15
    elif row["type_demande"] == "logiciel":
        score += 10
    elif row["type_demande"] == "login":
        score += 5
    
    # Règle 2 : Nombre d'utilisateurs impactés
    if row["utilisateurs_impactes"] > 50:
        score += 25
    elif row["utilisateurs_impactes"] > 5:
        score += 10
    else:
        score += 0
    
    # Règle 3 : Service (IT et Technique = plus critiques)
    if row["service"] in ["IT", "Technique"]:
        score += 5
    
    # Conversion score → priorité
    if score >= 35:
        return "haute"
    elif score >= 20:
        return "moyenne"
    else:
        return "faible"

df["priorite"] = df.apply(calcul_priorite, axis=1)

#Calcul du temps de traitement (avec vraies relations)
def calcul_temps(row):
    temps = 8  # Temps de base (heures)
    
    # Effet type de demande
    effet_type = {
        "login": 2,
        "logiciel": 8,
        "materiel": 15,
        "reseau": 20
    }
    temps += effet_type[row["type_demande"]]
    
    # Effet priorité (haute = traitement plus rapide car urgent)
    effet_priorite = {
        "faible": 8,
        "moyenne": 4,
        "haute": -2  # On met les moyens, donc plus rapide
    }
    temps += effet_priorite[row["priorite"]]
    
    # Effet service (IT plus efficace, Finance plus lent)
    effet_service = {
        "IT": -3,
        "Technique": -1,
        "RH": 2,
        "Finance": 5
    }
    temps += effet_service[row["service"]]
    
    # Effet heure (créé en fin de journée = traité le lendemain)
    if row["heure_creation"] >= 16:
        temps += 4
    
    # Effet vendredi (traite le lundi)
    if row["jour_semaine"] == "Vendredi":
        temps += 6
    
    # Bruit aléatoire (±2 heures)
    temps += np.random.normal(0, 2)
    
    # Minimum 1 heure
    return max(1, round(temps, 2))

df["temps_traitement"] = df.apply(calcul_temps, axis=1)

#Ajout de la date de création
start_date = datetime(2024, 1, 1)
jours_map = {"Lundi": 0, "Mardi": 1, "Mercredi": 2, "Jeudi": 3, "Vendredi": 4}

df["date_creation"] = [
    start_date + timedelta(days=7*(i//5) + jours_map[j]) 
    for i, j in enumerate(df["jour_semaine"])
]
df["date_creation"] = df["date_creation"] + pd.to_timedelta(df["heure_creation"], unit="h")

#Création du dataset pour le ML (sans priorité !)
# On sauvegarde aussi une version "cachée" pour la classification
df.to_csv("demandes_complet.csv", index=False)

# Version pour la prédiction du temps (on cache la priorité pour tester le modèle)
df_ml = df.drop(columns=["priorite"])
df_ml.to_csv("demandes2.csv", index=False)



print("VÉRIFICATION DES RÈGLES MÉTIER")


print("\n Distribution des priorités par type :")
print(pd.crosstab(df["type_demande"], df["priorite"], normalize="index").round(2))

print("\n Temps moyen par type :")
print(df.groupby("type_demande")["temps_traitement"].mean().round(2))

print("\n Temps moyen par priorité :")
print(df.groupby("priorite")["temps_traitement"].mean().round(2))

print("\n Temps moyen par service :")
print(df.groupby("service")["temps_traitement"].mean().round(2))

print("\n Aperçu :")
print(df[["type_demande", "service", "utilisateurs_impactes", 
          "priorite", "temps_traitement", "jour_semaine"]].head(10))
