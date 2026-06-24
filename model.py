import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              classification_report, confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES = ["type_demande", "service", "utilisateurs_impactes",
            "heure_creation", "jour_semaine"]

# ─── MODELE 1: Regression lineaire ───
print("=" * 60)
print("MODELE 1 : PREDICTION DU TEMPS DE TRAITEMENT")
print("=" * 60)

df = pd.read_csv("demandes2.csv")  # Sans priorite !

missing_cols = [c for c in FEATURES + ["temps_traitement"] if c not in df.columns]
if missing_cols:
    raise ValueError(f"Colonnes manquantes dans demandes.csv : {missing_cols}")

X = pd.get_dummies(df[FEATURES], drop_first=True)
y = df["temps_traitement"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRegression Lineaire :")
print(f"   R2 = {lr.score(X_test, y_test):.4f}")
print(f"   RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}h")
print(f"   MAE = {mean_absolute_error(y_test, y_pred_lr):.2f}h")

print("\nRandom Forest :")
print(f"   R2 = {rf.score(X_test, y_test):.4f}")
print(f"   RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}h")
print(f"   MAE = {mean_absolute_error(y_test, y_pred_rf):.2f}h")

importance = pd.DataFrame({
    "Variable": X.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nVariables les plus importantes :")
print(importance.head(8))

# ─── MODELE 2 : CLASSIFICATION (retrouver la priorite) ───
print("\n" + "=" * 60)
print("MODELE 2 : CLASSIFICATION DE LA PRIORITE")
print("=" * 60)

df_class = pd.read_csv("demandes_complet.csv")

missing_cols_c = [c for c in FEATURES + ["priorite"] if c not in df_class.columns]
if missing_cols_c:
    raise ValueError(f"Colonnes manquantes dans demandes_complet.csv : {missing_cols_c}")

X_class = pd.get_dummies(df_class[FEATURES], drop_first=True)
y_class = df_class["priorite"]

# On verifie les classes reellement presentes plutot que de les coder en dur
class_labels = sorted(y_class.unique())

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

print("\nRapport de classification :")
print(classification_report(yc_test, yc_pred, labels=class_labels))

# Matrice de confusion
plt.figure(figsize=(8, 6))
cm = confusion_matrix(yc_test, yc_pred, labels=class_labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title("Matrice de confusion - Prediction de la priorite")
plt.ylabel("Reel")
plt.xlabel("Predit")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()  # evite un blocage en mode script

imp_class = pd.DataFrame({
    "Variable": X_class.columns,
    "Importance": clf.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nVariables determinantes pour la priorite :")
print(imp_class.head(8))
