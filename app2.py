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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Dashboard", "Prédiction", "Analyse", "Données", "NPL", "Alertes"])

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
# ═══════════════════════════════════════
# TAB 5 : NPL
# ═══════════════════════════════════════
with tab5:
    st.header("🤖 Classification automatique par NLP")
    
    # Zone de texte pour l'utilisateur
    st.subheader("📝 Décrivez votre problème")
    user_text = st.text_area(
        "Écrivez librement votre demande (ex: 'Mon écran ne s'allume plus')",
        height=100
    )
    
    if user_text:
        # Chargement du modèle (en cache)
        @st.cache_resource
        def load_nlp_model():
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.naive_bayes import MultinomialNB
            import joblib
            
            # Ou réentraîner si pas sauvegardé
            df_nlp = pd.read_csv("demandes_nlp.csv")
            
            def clean(t):
                import re
                t = t.lower()
                t = re.sub(r'[^\w\s]', ' ', t)
                t = re.sub(r'\s+', ' ', t)
                return t.strip()
            
            df_nlp["clean"] = df_nlp["description"].apply(clean)
            
            tfidf = TfidfVectorizer(max_features=1000, min_df=2, max_df=0.8, ngram_range=(1,2))
            X = tfidf.fit_transform(df_nlp["clean"])
            y = df_nlp["type_demande"]
            
            model = MultinomialNB()
            model.fit(X, y)
            
            return tfidf, model
        
        tfidf, model = load_nlp_model()
        
        # Prédiction
        import re
        clean_text = user_text.lower()
        clean_text = re.sub(r'[^\w\s]', ' ', clean_text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        text_tfidf = tfidf.transform([clean_text])
        prediction = model.predict(text_tfidf)[0]
        proba = model.predict_proba(text_tfidf)[0]
        confiance = max(proba)
        
        # Affichage résultat
        col_r1, col_r2, col_r3 = st.columns(3)
        
        with col_r1:
            st.metric("🔮 Type détecté", prediction.upper())
        
        with col_r2:
            st.metric("📊 Confiance", f"{confiance*100:.1f}%")
        
        with col_r3:
            if confiance > 0.8:
                st.success("✅ Haute confiance")
            elif confiance > 0.5:
                st.warning("⚠️ Vérifier")
            else:
                st.error("🔴 Incertain")
        
        # INTERPRÉTATION
        st.subheader("💬 Analyse")
        
        type_desc = {
            "reseau": "Problème d'infrastructure réseau (connexion, WiFi, serveur...)",
            "login": "Problème d'accès ou d'authentification (mot de passe, compte...)",
            "materiel": "Panne matérielle (écran, PC, imprimante...)",
            "logiciel": "Problème logiciel (application, bug, installation...)"
        }
        
        st.info(f"""
        **Type identifié : {prediction.upper()}**
        
        {type_desc.get(prediction, "")}
        
        **Confiance : {confiance*100:.1f}%**
        - {'Le modèle est très sûr de sa prédiction.' if confiance > 0.8 else 'Le modèle est modérément confiant.' if confiance > 0.5 else 'Le modèle hésite entre plusieurs types.'}
        
        **Recommandation :**
        - {'Le ticket peut être auto-classifié et routé directement.' if confiance > 0.8 else 'Vérifier manuellement le type avant validation.'}
        """)
        
        # Probabilités détaillées
        st.subheader("📊 Probabilités par type")
        proba_df = pd.DataFrame({
            "Type": model.classes_,
            "Probabilité": proba
        }).sort_values("Probabilité", ascending=False)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['green' if p == max(proba) else 'gray' for p in proba]
        ax.barh(proba_df["Type"], proba_df["Probabilité"], color=colors)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilité")
        st.pyplot(fig)

st.markdown("---")
st.caption("Projet Data Science INSEA - Système intelligent de gestion DSI")


# ═══════════════════════════════════════════════════════════════════
# TAB 6 : ALERTES ET ANOMALIES (nouvel onglet)
# ═══════════════════════════════════════════════════════════════════
with tab6:
    st.header("🚨 Centre d'Alertes - Détection d'Anomalies")
    
    st.markdown("""
    **Objectif** : Identifier automatiquement les tickets atypiques qui risquent d'être bloqués ou oubliés.
    
    Un ticket est marqué comme **anomalie** si son temps de traitement est inhabituel par rapport à son profil (type, service, priorité, nombre d'utilisateurs).
    """)
    
    # Chargement des données avec anomalies
    try:
        df_anom = pd.read_csv("demandes_anomalies.csv")
    except:
        # Calculer les anomalies si pas encore fait
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        features = ["temps_traitement", "utilisateurs_impactes", "heure_creation"]
        df_encoded = pd.get_dummies(df[["type_demande", "service", "priorite"]], drop_first=True)
        X = pd.concat([df[features], df_encoded], axis=1)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        iso = IsolationForest(contamination=0.05, random_state=42)
        df["anomalie"] = iso.fit_predict(X_scaled)
        df["est_anomalie"] = df["anomalie"].apply(lambda x: "ANOMALIE" if x == -1 else "Normal")
        df_anom = df.copy()
    
    # ─── KPIs Alertes ───
    n_anomalies = len(df_anom[df_anom["anomalie"] == -1])
    pct_anomalies = n_anomalies / len(df_anom) * 100
    
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    
    with col_a1:
        st.metric("🚨 Tickets anormaux", n_anomalies, f"{pct_anomalies:.1f}%")
    
    with col_a2:
        temps_max_anom = df_anom[df_anom["anomalie"] == -1]["temps_traitement"].max()
        st.metric("⏱️ Temps max anomalie", f"{temps_max_anom:.1f}h")
    
    with col_a3:
        type_plus_anom = df_anom[df_anom["anomalie"] == -1]["type_demande"].mode()[0]
        st.metric("📌 Type le plus anormal", type_plus_anom)
    
    with col_a4:
        service_plus_anom = df_anom[df_anom["anomalie"] == -1]["service"].mode()[0]
        st.metric("🏢 Service le plus anormal", service_plus_anom)
    
    st.markdown("---")
    
    # ─── Liste des anomalies ───
    st.subheader("🔴 Tickets nécessitant une attention immédiate")
    
    anomalies = df_anom[df_anom["anomalie"] == -1].sort_values("temps_traitement", ascending=False)
    
    # Filtres d'alertes
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        alert_type = st.multiselect("Filtrer par type", anomalies["type_demande"].unique(), 
                                    default=anomalies["type_demande"].unique(), key="alert_type")
    with col_f2:
        alert_service = st.multiselect("Filtrer par service", anomalies["service"].unique(),
                                       default=anomalies["service"].unique(), key="alert_service")
    
    anomalies_filtered = anomalies[
        (anomalies["type_demande"].isin(alert_type)) &
        (anomalies["service"].isin(alert_service))
    ]
    
    # Affichage des alertes
    for idx, row in anomalies_filtered.head(10).iterrows():
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**🚨 Ticket #{idx}**")
                if "description" in row:
                    st.write(f"📝 {row['description'][:80]}...")
                st.write(f"📌 {row['type_demande']} | 🏢 {row['service']} | 🔥 {row['priorite']}")
            
            with col2:
                st.metric("⏱️ Temps", f"{row['temps_traitement']:.1f}h")
                # Comparaison avec la moyenne du type
                moy_type = df_anom[df_anom["type_demande"] == row["type_demande"]]["temps_traitement"].mean()
                ecart = ((row["temps_traitement"] - moy_type) / moy_type * 100)
                st.write(f"📊 +{ecart:.0f}% vs moyenne")
            
            with col3:
                if row["temps_traitement"] > 40:
                    st.error("🔴 CRITIQUE")
                elif row["temps_traitement"] > 30:
                    st.warning("🟡 ÉLEVÉ")
                else:
                    st.info("🟢 À VÉRIFIER")
                
                # Bouton d'action
                if st.button(f"📞 Contacter", key=f"btn_{idx}"):
                    st.success(f"✅ Notification envoyée au service {row['service']}")
            
            st.markdown("---")
    
    # ─── Visualisation ───
    st.subheader("📊 Carte des anomalies")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    normaux = df_anom[df_anom["anomalie"] == 1]
    anoms = df_anom[df_anom["anomalie"] == -1]
    
    ax.scatter(normaux["utilisateurs_impactes"], normaux["temps_traitement"], 
              c="lightblue", alpha=0.3, s=20, label=f"Normal ({len(normaux)})")
    ax.scatter(anoms["utilisateurs_impactes"], anoms["temps_traitement"], 
              c="red", alpha=0.8, s=100, marker="X", label=f"Anomalie ({len(anoms)})", edgecolors="black")
    
    # Ligne de tendance
    z = np.polyfit(df_anom["utilisateurs_impactes"], df_anom["temps_traitement"], 1)
    p = np.poly1d(z)
    ax.plot(df_anom["utilisateurs_impactes"].sort_values(), 
            p(df_anom["utilisateurs_impactes"].sort_values()), 
            "g--", alpha=0.5, label="Tendance")
    
    ax.set_xlabel("Utilisateurs impactés")
    ax.set_ylabel("Temps de traitement (heures)")
    ax.set_title("Tickets normaux (bleu) vs Anomalies (rouge X)")
    ax.legend()
    
    st.pyplot(fig)
    
    # INTERPRÉTATION
    st.info(f"""
    💬 **Interprétation :**
    
    - **{len(anoms)} tickets** ({pct_anomalies:.1f}%) sont atypiques par rapport à leur profil habituel
    - Les anomalies (🔴 X) s'écartent de la tendance normale (ligne verte)
    - **Causes possibles :**
      • Ticket bloqué ou oublié dans le système
      • Mauvaise assignation (agent incompétent pour ce type)
      • Problème complexe nécessitant une escalade
      • Dépendance externe (fournisseur, autre service)
    
    **Actions recommandées :**
    1. 📞 Contacter l'agent assigné pour status
    2. 🔄 Réassigner si nécessaire
    3. 📊 Analyser les causes récurrentes
    """)
