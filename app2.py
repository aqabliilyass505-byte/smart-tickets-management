import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import sqlite3

st.set_page_config(page_title="DSI Intelligence", layout="wide")

# ─── Chargement des données ───
@st.cache_data  # Cache pour ne pas recharger à chaque interaction
def load_data():
    try:
        # Essayer SQLite d'abord
        conn = sqlite3.connect("dsi_database.db")
        df = pd.read_sql("SELECT * FROM tickets", conn)
        conn.close()
    except:
        # Sinon CSV
        df = pd.read_csv("demandes_complet.csv")
    return df

df = load_data()

st.title("🏢 DSI Intelligence - Système de Gestion des Tickets")
st.markdown("---")

# ─── KPIs ───
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📋 Tickets totaux", len(df))
col2.metric("⏱️ Temps moyen", f"{df['temps_traitement'].mean():.1f}h")
col3.metric("🔥 Haute priorité", len(df[df['priorite']=='haute']))
col4.metric("⚡ Service + rapide", 
            df.groupby("service")["temps_traitement"].mean().idxmin())
col5.metric("🐌 Type + long", 
            df.groupby("type_demande")["temps_traitement"].mean().idxmax())

st.markdown("---")

# ─── Filtres interactifs ───
st.sidebar.header("🔍 Filtres")
type_filter = st.sidebar.multiselect(
    "Type de demande", 
    df["type_demande"].unique(), 
    default=df["type_demande"].unique()
)
service_filter = st.sidebar.multiselect(
    "Service", 
    df["service"].unique(), 
    default=df["service"].unique()
)
prio_filter = st.sidebar.multiselect(
    "Priorité", 
    df["priorite"].unique(), 
    default=df["priorite"].unique()
)

# Application des filtres
filtered = df[
    (df["type_demande"].isin(type_filter)) &
    (df["service"].isin(service_filter)) &
    (df["priorite"].isin(prio_filter))
]

st.sidebar.markdown(f"**{len(filtered)} tickets filtrés**")

# ─── Onglets ───
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🤖 Prédiction", "📈 Analyse", "📋 Données"])

# ═══════════════════════════════════════
# TAB 1 : DASHBOARD
# ═══════════════════════════════════════
with tab1:
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📊 Temps par type")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=filtered, x="type_demande", y="temps_traitement", ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with col_right:
        st.subheader("🔥 Répartition des priorités")
        fig, ax = plt.subplots(figsize=(8, 5))
        filtered["priorite"].value_counts().plot(
            kind="pie", ax=ax, autopct="%1.1f%%", 
            colors=["green", "orange", "red"]
        )
        st.pyplot(fig)
    
    # Scatter plot
    st.subheader("📈 Impact utilisateurs vs Temps")
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_map = {"faible": "green", "moyenne": "orange", "haute": "red"}
    for prio in filtered["priorite"].unique():
        data = filtered[filtered["priorite"] == prio]
        ax.scatter(data["utilisateurs_impactes"], data["temps_traitement"], 
                   c=colors_map.get(prio, "blue"), label=prio, alpha=0.6)
    plt.xlabel("Utilisateurs impactés")
    plt.ylabel("Temps (heures)")
    plt.legend(title="Priorité")
    st.pyplot(fig)
    
    # Moyennes par service
    st.subheader("🏢 Performance par service")
    fig, ax = plt.subplots(figsize=(10, 4))
    service_stats = filtered.groupby("service")["temps_traitement"].mean().sort_values()
    service_stats.plot(kind="barh", color="skyblue", ax=ax)
    plt.xlabel("Temps moyen (heures)")
    st.pyplot(fig)

# ═══════════════════════════════════════
# TAB 2 : PRÉDICTION
# ═══════════════════════════════════════
with tab2:
    st.subheader("🎯 Simulateur de prédiction")
    
    col_sim1, col_sim2, col_sim3, col_sim4 = st.columns(4)
    type_sim = col_sim1.selectbox("Type", df["type_demande"].unique())
    service_sim = col_sim2.selectbox("Service", df["service"].unique())
    users_sim = col_sim3.slider("Utilisateurs", 1, 200, 10)
    heure_sim = col_sim4.selectbox("Heure", range(8, 19))
    
    # Entraînement du modèle (simple)
    X = pd.get_dummies(df[["type_demande", "service", "utilisateurs_impactes", "heure_creation"]], drop_first=True)
    y = df["temps_traitement"]
    
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Créer la ligne de prédiction
    input_data = pd.DataFrame({
        "type_demande": [type_sim],
        "service": [service_sim],
        "utilisateurs_impactes": [users_sim],
        "heure_creation": [heure_sim]
    })
    input_encoded = pd.get_dummies(input_data)
    
    # Aligner les colonnes
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X.columns]
    
    prediction = model.predict(input_encoded)[0]
    
    st.metric("⏱️ Temps estimé", f"{prediction:.1f} heures")
    
    # Comparaison avec la moyenne historique
    moyenne_historique = df[
        (df["type_demande"] == type_sim) & 
        (df["service"] == service_sim)
    ]["temps_traitement"].mean()
    
    if not np.isnan(moyenne_historique):
        delta = prediction - moyenne_historique
        st.metric("📊 Moyenne historique", f"{moyenne_historique:.1f}h", f"{delta:+.1f}h")
    
    # Prédiction priorité
    st.subheader("🔮 Prédiction de la priorité")
    
    # Règle métier simple
    if type_sim == "reseau" and users_sim > 50:
        prio_pred = "haute"
    elif type_sim == "login" and users_sim < 5:
        prio_pred = "faible"
    elif users_sim > 50:
        prio_pred = "haute"
    elif users_sim > 5:
        prio_pred = "moyenne"
    else:
        prio_pred = "faible"
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.metric("Priorité prédite", prio_pred.upper())
    with col_p2:
        st.metric("Confiance", "85%" if prio_pred == "haute" else "70%")
    with col_p3:
        st.metric("Action recommandée", "Urgent" if prio_pred == "haute" else "Standard")

# ═══════════════════════════════════════
# TAB 3 : ANALYSE AVANCÉE
# ═══════════════════════════════════════
with tab3:
    st.subheader("📊 Heatmap Type × Priorité")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    heatmap_data = filtered.pivot_table(
        values="temps_traitement", 
        index="type_demande", 
        columns="priorite", 
        aggfunc="mean"
    )
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
    st.pyplot(fig)
    
    st.subheader("📅 Distribution temporelle")
    if "date_creation" in df.columns:
        df["date_creation"] = pd.to_datetime(df["date_creation"])
        fig, ax = plt.subplots(figsize=(12, 4))
        df.groupby(df["date_creation"].dt.date).size().plot(ax=ax)
        plt.title("Nombre de tickets par jour")
        st.pyplot(fig)

# ═══════════════════════════════════════
# TAB 4 : DONNÉES BRUTES
# ═══════════════════════════════════════
with tab4:
    st.subheader(f"📋 {len(filtered)} tickets filtrés")
    
    # Colonnes à afficher
    cols = ["type_demande", "service", "priorite", "utilisateurs_impactes", 
            "temps_traitement", "jour_semaine"]
    if "date_creation" in filtered.columns:
        cols.insert(0, "date_creation")
    
    st.dataframe(filtered[cols], use_container_width=True)
    
    # Export CSV
    csv = filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Télécharger CSV",
        csv,
        "tickets_filtres.csv",
        "text/csv"
    )

st.markdown("---")
st.caption("Projet Data Science INSEA - Système intelligent de gestion DSI")
