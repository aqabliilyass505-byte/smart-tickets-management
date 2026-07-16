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
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Dashboard", "Prédiction", "Analyse", "Données", "NPL", "Alertes","Planificateur"])

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

# ═══════════════════════════════════════════════════════════════════
# TAB 7 : PLANIFICATEUR (nouvel onglet)
# ═══════════════════════════════════════════════════════════════════
with tab7:
    st.header("📅 Planificateur Intelligent de Tickets")
    
    st.markdown("""
    **Objectif** : Assigner automatiquement chaque ticket à l'agent le plus compétent, 
    en optimisant la charge de travail et en minimisant le temps total de traitement.
    
    **Algorithme** : Score d'adéquation = Expertise + Efficacité + Priorité + Capacité
    """)
    
    # ─── Données des agents (modifiable) ───
    st.subheader("👥 Configuration de l'équipe")
    
    agents_config = {
        "Alice": {"service": "IT", "expertise": ["reseau", "login", "logiciel"], "efficacite": 0.85, "heures": 8},
        "Bob": {"service": "Technique", "expertise": ["materiel", "reseau"], "efficacite": 0.75, "heures": 8},
        "Charlie": {"service": "IT", "expertise": ["logiciel", "login"], "efficacite": 0.90, "heures": 8},
        "Diana": {"service": "RH", "expertise": ["login", "logiciel"], "efficacite": 0.70, "heures": 7},
        "Eve": {"service": "Finance", "expertise": ["logiciel", "materiel"], "efficacite": 0.80, "heures": 8},
        "Frank": {"service": "Technique", "expertise": ["reseau", "materiel", "login"], "efficacite": 0.78, "heures": 8},
    }
    
    # Affichage agents
    cols_agents = st.columns(len(agents_config))
    for i, (nom, info) in enumerate(agents_config.items()):
        with cols_agents[i]:
            st.metric(
                label=nom,
                value=f"{info['efficacite']*100:.0f}%",
                delta=f"{info['service']}"
            )
            st.caption(f"Expertise: {', '.join(info['expertise'])}")
    
    # ─── Tickets à planifier ───
    st.markdown("---")
    st.subheader("📋 Tickets à assigner")
    
    # Simulation : tickets ouverts du dataset
    if "statut" not in df.columns:
        df["statut"] = np.random.choice(["ouvert", "en_cours", "resolu"], len(df), p=[0.3, 0.2, 0.5])
    
    tickets_ouverts = df[df["statut"] == "ouvert"].copy()
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        type_plan = st.multiselect("Filtrer par type", tickets_ouverts["type_demande"].unique(),
                                   default=tickets_ouverts["type_demande"].unique(), key="plan_type")
    with col_f2:
        prio_plan = st.multiselect("Filtrer par priorité", tickets_ouverts["priorite"].unique(),
                                   default=["haute", "moyenne"], key="plan_prio")
    
    tickets_filtres = tickets_ouverts[
        (tickets_ouverts["type_demande"].isin(type_plan)) &
        (tickets_ouverts["priorite"].isin(prio_plan))
    ]
    
    st.write(f"**{len(tickets_filtres)} tickets à planifier**")
    st.dataframe(tickets_filtres[["type_demande", "service", "priorite", "temps_traitement", "description"]].head(10),
                 use_container_width=True)
    
    # ─── Lancer la planification ───
    if st.button("🚀 Lancer la planification optimale", key="btn_plan"):
        with st.spinner("📅 Optimisation en cours..."):
            
            # Algorithme d'assignation
            def score_agent(agent_name, agent_info, ticket):
                score = 0
                # Expertise
                if ticket["type_demande"] in agent_info["expertise"]:
                    score += 100
                else:
                    score += 20
                # Efficacité
                score += agent_info["efficacite"] * 100
                # Priorité
                prio_bonus = {"haute": 50, "moyenne": 20, "faible": 0}
                score += prio_bonus.get(ticket["priorite"], 0)
                # Même service
                if ticket["service"] == agent_info["service"]:
                    score += 25
                return score
            
            def temps_estime(agent_info, ticket):
                base = ticket["temps_traitement"]
                if ticket["type_demande"] in agent_info["expertise"]:
                    return base * (1 - agent_info["efficacite"] * 0.3)
                return base * 1.2
            
            # Planification
            planning_result = []
            charge = {a: 0 for a in agents_config}
            
            # Trier par priorité
            tickets_sorted = tickets_filtres.sort_values(
                "priorite",
                key=lambda x: x.map({"haute": 0, "moyenne": 1, "faible": 2})
            )
            
            for idx, ticket in tickets_sorted.iterrows():
                best_agent = None
                best_score = -1
                
                for nom, info in agents_config.items():
                    t_est = temps_estime(info, ticket)
                    if charge[nom] + t_est > info["heures"]:
                        continue
                    
                    sc = score_agent(nom, info, ticket)
                    if sc > best_score:
                        best_score = sc
                        best_agent = nom
                
                if best_agent:
                    t_reel = temps_estime(agents_config[best_agent], ticket)
                    charge[best_agent] += t_reel
                    
                    planning_result.append({
                        "ticket_id": idx,
                        "description": str(ticket.get("description", f"Ticket {idx}"))[:40],
                        "type": ticket["type_demande"],
                        "priorite": ticket["priorite"],
                        "temps_base": ticket["temps_traitement"],
                        "agent": best_agent,
                        "temps_estime": t_reel,
                        "gain": (ticket["temps_traitement"] - t_reel) / ticket["temps_traitement"] * 100,
                        "charge_agent": charge[best_agent]
                    })
            
            df_plan = pd.DataFrame(planning_result)
            
            # ─── RÉSULTATS ───
            st.markdown("---")
            st.subheader("✅ Planification optimale générée")
            
            # KPIs
            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            with col_p1:
                st.metric("📋 Assignés", f"{len(df_plan)}/{len(tickets_filtres)}")
            with col_p2:
                gain_moy = df_plan["gain"].mean() if len(df_plan) > 0 else 0
                st.metric("📈 Gain efficacité", f"{gain_moy:.1f}%")
            with col_p3:
                temps_total = df_plan["temps_estime"].sum() if len(df_plan) > 0 else 0
                st.metric("⏱️ Temps total", f"{temps_total:.1f}h")
            with col_p4:
                non_assignes = len(tickets_filtres) - len(df_plan)
                st.metric("❌ Non assignés", non_assignes)
            
            # Tableau de planification
            st.subheader("📅 Tableau d'assignation")
            st.dataframe(df_plan.sort_values("priorite", key=lambda x: x.map({"haute": 0, "moyenne": 1, "faible": 2})),
                        use_container_width=True)
            
            # Visualisation charge par agent
            st.subheader("👥 Charge de travail par agent")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            agents_list = list(agents_config.keys())
            charges = [charge[a] for a in agents_list]
            capacites = [agents_config[a]["heures"] for a in agents_list]
            
            x = np.arange(len(agents_list))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, charges, width, label='Charge planifiée', color='skyblue')
            bars2 = ax.bar(x + width/2, capacites, width, label='Capacité max', color='lightgray', alpha=0.5)
            
            # Colorer en rouge si surcharge
            for i, (c, cap) in enumerate(zip(charges, capacites)):
                if c > cap * 0.9:
                    bars1[i].set_color('red')
                elif c < cap * 0.5:
                    bars1[i].set_color('green')
            
            ax.set_ylabel('Heures')
            ax.set_title('Charge de travail vs Capacité par agent')
            ax.set_xticks(x)
            ax.set_xticklabels(agents_list, rotation=45)
            ax.legend()
            
            st.pyplot(fig)
            
            # INTERPRÉTATION
            surcharges = [a for a in agents_list if charge[a] > agents_config[a]["heures"] * 0.9]
            sous_utilises = [a for a in agents_list if charge[a] < agents_config[a]["heures"] * 0.5]
            
            st.info(f"""
            💬 **Analyse de la planification :**
            
            - **{len(df_plan)} tickets assignés** sur {len(tickets_filtres)} demandés
            - **Gain moyen d'efficacité : {gain_moy:.1f}%** grâce à l'assignation optimale
            
            **Agents surchargés (>90%) :**
            {', '.join(surcharges) if surcharges else 'Aucun ✅'}
            
            **Agents sous-utilisés (<50%) :**
            {', '.join(sous_utilises) if sous_utilises else 'Aucun ✅'}
            
            **Recommandations :**
            {'🔴 Réduire la charge de ' + ', '.join(surcharges) if surcharges else '✅ Charge bien répartie'}
            {'🟡 Augmenter les tickets de ' + ', '.join(sous_utilises) if sous_utilises else ''}
            """)
            
            # Gantt-like : Planning horaire
            st.subheader("⏰ Planning journalier suggéré")
            
            # Créer un planning horaire simple
            heure_debut = 9  # 9h
            planning_horaire = {a: [] for a in agents_config}
            
            for _, row in df_plan.iterrows():
                agent = row["agent"]
                duree = row["temps_estime"]
                # Trouver le premier créneau disponible
                debut = heure_debut
                while any(abs(debut - h[0]) < h[1] for h in planning_horaire[agent]):
                    debut += 1
                planning_horaire[agent].append((debut, duree, row["type"], row["priorite"]))
            
            # Affichage
            fig_gantt, ax_gantt = plt.subplots(figsize=(12, 6))
            
            colors_prio = {"haute": "red", "moyenne": "orange", "faible": "green"}
            colors_type = {"reseau": "#FF6B6B", "login": "#4ECDC4", "materiel": "#45B7D1", "logiciel": "#96CEB4"}
            
            for i, agent in enumerate(agents_list):
                for debut, duree, type_t, prio in planning_horaire[agent]:
                    ax_gantt.barh(i, duree, left=debut, height=0.4, 
                                 color=colors_type.get(type_t, "gray"), 
                                 edgecolor=colors_prio.get(prio, "black"),
                                 linewidth=2 if prio == "haute" else 1)
            
            ax_gantt.set_yticks(range(len(agents_list)))
            ax_gantt.set_yticklabels(agents_list)
            ax_gantt.set_xlabel("Heure de la journée")
            ax_gantt.set_title("Planning journalier optimisé (couleur = type, bordure = priorité)")
            ax_gantt.set_xlim(9, 18)
            
            # Légende
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=colors_type[t], label=t) for t in colors_type]
            ax_gantt.legend(handles=legend_elements, loc='upper right', title="Type")
            
            st.pyplot(fig_gantt)
            
            st.info("""
            💬 **Lecture du planning :**
            - Chaque barre = un ticket assigné
            - **Couleur** = type de demande
            - **Bordure rouge** = haute priorité (à traiter en priorité)
            - **Bordure noire** = priorité standard
            
            **Optimisation :** Les tickets urgents sont placés en début de journée sur les agents les plus efficaces.
            """)
            
            # Export
            csv_plan = df_plan.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Télécharger le planning (CSV)", csv_plan, "planification.csv", "text/csv")# ═══════════════════════════════════════════════════════════════════
# TAB 8 : PLANIFICATEUR INTELLIGENT (NLP-style)
# ═══════════════════════════════════════════════════════════════════
with tab7:
    st.header("📅 Planificateur Intelligent - Assignation par Description")
    
    st.markdown("""
    **Comment ça marche :**
    1. 📝 Vous décrivez votre problème en langage naturel
    2. 🤖 L'IA détecte automatiquement le **type** de demande
    3. 👤 L'IA assigne le **meilleur agent** selon son expertise et sa disponibilité
    4. ⏱️ L'IA estime le **temps de résolution**
    """)
    
    # ─── Configuration des agents ───
    st.subheader("👥 Équipe disponible")
    
    agents_config = {
        "Alice": {"service": "IT", "expertise": ["reseau", "login", "logiciel"], "efficacite": 0.85, "heures": 8},
        "Bob": {"service": "Technique", "expertise": ["materiel", "reseau"], "efficacite": 0.75, "heures": 8},
        "Charlie": {"service": "IT", "expertise": ["logiciel", "login"], "efficacite": 0.90, "heures": 8},
        "Diana": {"service": "RH", "expertise": ["login", "logiciel"], "efficacite": 0.70, "heures": 7},
        "Eve": {"service": "Finance", "expertise": ["logiciel", "materiel"], "efficacite": 0.80, "heures": 8},
        "Frank": {"service": "Technique", "expertise": ["reseau", "materiel", "login"], "efficacite": 0.78, "heures": 8},
    }
    
    # Affichage agents
    cols_agents = st.columns(len(agents_config))
    for i, (nom, info) in enumerate(agents_config.items()):
        with cols_agents[i]:
            st.metric(label=nom, value=f"{info['efficacite']*100:.0f}%", delta=f"{info['service']}")
            st.caption(f"Expertise: {', '.join(info['expertise'])}")
    
    st.markdown("---")
    
    # ─── ÉTAPE 1 : Description du problème ───
    st.subheader("📝 Étape 1 : Décrivez votre problème")
    
    user_description = st.text_area(
        "Décrivez votre problème en langage naturel",
        height=120,
        placeholder="Exemples : 'Mon écran ne s'allume plus' ou 'Le WiFi est down dans tout le bâtiment'..."
    )
    
    # Informations complémentaires
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        service_user = st.selectbox("Votre service", ["IT", "RH", "Finance", "Technique"], key="plan_service")
    with col_info2:
        users_impact = st.slider("Utilisateurs impactés", 1, 200, 5, key="plan_users")
    with col_info3:
        urgence_user = st.selectbox("Niveau d'urgence perçu", ["Faible", "Moyenne", "Haute"], key="plan_urgence")
    
    # ─── Chargement NLP si disponible ───
    nlp_available = False
    try:
        if tfidf is not None and clf_type is not None:
            nlp_available = True
    except:
        pass
    
    if st.button("🚀 Trouver le meilleur agent", key="btn_plan_nlp", type="primary"):
        if not user_description.strip():
            st.error("❌ Veuillez décrire votre problème")
            st.stop()
        
        with st.spinner("🤖 Analyse en cours..."):
            
            # ═══════════════════════════════════════════════════════════
            # ÉTAPE 2 : Détection du type (NLP ou mots-clés)
            # ═══════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("🔍 Étape 2 : Analyse du problème")
            
            # Méthode 1 : NLP si disponible
            if nlp_available:
                clean_desc = user_description.lower()
                clean_desc = re.sub(r'[^\w\s]', ' ', clean_desc)
                clean_desc = re.sub(r'\d+', ' ', clean_desc)
                clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
                
                desc_tfidf = tfidf.transform([clean_desc])
                type_detecte = clf_type.predict(desc_tfidf)[0]
                proba_type = clf_type.predict_proba(desc_tfidf)[0]
                confiance_type = max(proba_type)
                
                col_type1, col_type2 = st.columns(2)
                with col_type1:
                    st.metric("🔮 Type détecté", type_detecte.upper())
                with col_type2:
                    st.metric("📊 Confiance NLP", f"{confiance_type*100:.1f}%")
                
                if confiance_type > 0.8:
                    st.success("✅ Classification fiable par NLP")
                else:
                    st.warning("⚠️ Faible confiance, vérification manuelle recommandée")
            
            # Méthode 2 : Mots-clés (fallback)
            else:
                # Détection par mots-clés simples
                desc_lower = user_description.lower()
                
                if any(mot in desc_lower for mot in ["wifi", "réseau", "vpn", "connexion", "internet", "switch", "serveur"]):
                    type_detecte = "reseau"
                elif any(mot in desc_lower for mot in ["écran", "pc", "ordinateur", "imprimante", "souris", "clavier", "matériel"]):
                    type_detecte = "materiel"
                elif any(mot in desc_lower for mot in ["excel", "logiciel", "application", "bug", "plantage", "installation"]):
                    type_detecte = "logiciel"
                elif any(mot in desc_lower for mot in ["mot de passe", "compte", "login", "session", "accès"]):
                    type_detecte = "login"
                else:
                    type_detecte = "logiciel"  # Défaut
                
                st.info(f"🔍 Type détecté par mots-clés : **{type_detecte.upper()}**")
                st.caption("NLP non disponible - Utilisation de la détection par mots-clés")
            
            # ═══════════════════════════════════════════════════════════
            # ÉTAPE 3 : Recherche du meilleur agent
            # ═══════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("👤 Étape 3 : Agent recommandé")
            
            def score_agent(agent_name, agent_info, type_ticket, service_ticket, urgence):
                """Calcule le score d'adéquation"""
                score = 0
                
                # 1. Expertise (0 ou 100 points)
                if type_ticket in agent_info["expertise"]:
                    score += 100
                    expertise_match = True
                else:
                    score += 20
                    expertise_match = False
                
                # 2. Efficacité (0-90 points)
                score += agent_info["efficacite"] * 100
                
                # 3. Urgence (bonus priorité)
                urgence_bonus = {"Haute": 50, "Moyenne": 20, "Faible": 0}
                score += urgence_bonus.get(urgence, 0)
                
                # 4. Même service
                meme_service = (service_ticket == agent_info["service"])
                if meme_service:
                    score += 25
                
                return score, expertise_match, meme_service
            
            # Calculer le score pour chaque agent
            scores = []
            for nom, info in agents_config.items():
                sc, exp_match, same_serv = score_agent(nom, info, type_detecte, service_user, urgence_user)
                scores.append({
                    "agent": nom,
                    "score": sc,
                    "expertise_match": exp_match,
                    "meme_service": same_serv,
                    "efficacite": info["efficacite"],
                    "expertise_list": info["expertise"],
                    "service": info["service"]
                })
            
            # Trier par score
            scores_df = pd.DataFrame(scores).sort_values("score", ascending=False)
            
            # Agent gagnant
            best_agent = scores_df.iloc[0]["agent"]
            best_info = agents_config[best_agent]
            
            # ═══════════════════════════════════════════════════════════
            # AFFICHAGE RÉSULTAT
            # ═══════════════════════════════════════════════════════════
            
            # Carte principale de l'agent
            st.markdown("---")
            
            col_agent, col_details = st.columns([1, 2])
            
            with col_agent:
                # Avatar/couleur selon l'agent
                colors = {"Alice": "#FF6B6B", "Bob": "#4ECDC4", "Charlie": "#45B7D1", 
                         "Diana": "#96CEB4", "Eve": "#FFEAA7", "Frank": "#DDA0DD"}
                
                st.markdown(f"""
                <div style="background-color: {colors.get(best_agent, '#333')}; 
                            padding: 20px; border-radius: 10px; text-align: center;">
                    <h1 style="color: white; margin: 0;">👤</h1>
                    <h2 style="color: white; margin: 5px 0;">{best_agent}</h2>
                    <p style="color: white; font-size: 18px; margin: 0;">{best_info['service']}</p>
                    <p style="color: white; font-size: 24px; margin: 10px 0;">{best_info['efficacite']*100:.0f}%</p>
                    <p style="color: white; font-size: 12px; margin: 0;">Efficacité</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_details:
                st.subheader("📋 Pourquoi cet agent ?")
                
                # Raison principale
                if scores_df.iloc[0]["expertise_match"]:
                    st.success(f"✅ **Expert en {type_detecte.upper()}** - C'est son domaine de prédilection")
                else:
                    st.warning(f"⚠️ **Pas d'expert {type_detecte.upper()} disponible** - {best_agent} est le mieux adapté malgré tout")
                
                if scores_df.iloc[0]["meme_service"]:
                    st.info(f"🏢 **Même service** ({service_user}) - Connaissance du contexte")
                
                # Compétences
                st.write("**Expertise :** " + ", ".join([f"`{e}`" for e in best_info['expertise']]))
                
                # Score
                st.progress(scores_df.iloc[0]["score"] / 300, text=f"Score d'adéquation : {scores_df.iloc[0]['score']:.0f}/300")
            
            # ═══════════════════════════════════════════════════════════
            # ÉTAPE 4 : Estimation du temps
            # ═══════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("⏱️ Étape 4 : Estimation du temps de résolution")
            
            # Temps de base selon le type
            temps_base_type = {
                "reseau": 20,
                "materiel": 15,
                "logiciel": 10,
                "login": 5
            }
            temps_base = temps_base_type.get(type_detecte, 10)
            
            # Ajustements
            if users_impact > 50:
                temps_base *= 1.5
            elif users_impact > 10:
                temps_base *= 1.2
            
            # Efficacité de l'agent
            if type_detecte in best_info["expertise"]:
                temps_estime = temps_base * (1 - best_info["efficacite"] * 0.3)
            else:
                temps_estime = temps_base * 1.2
            
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.metric("⏱️ Temps estimé", f"{temps_estime:.1f} heures")
            with col_t2:
                st.metric("📊 vs Temps standard", f"{temps_base:.1f}h", f"{((temps_estime-temps_base)/temps_base)*100:+.0f}%")
            with col_t3:
                if temps_estime <= 24:
                    st.success("✅ Délai acceptable")
                else:
                    st.error("🔴 Délai long")
            
            # ═══════════════════════════════════════════════════════════
            # ÉTAPE 5 : Classement des agents
            # ═══════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("📊 Classement de tous les agents")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            
            noms = scores_df["agent"].tolist()
            scores_vals = scores_df["score"].tolist()
            colors_bar = ['green' if i == 0 else 'orange' if i == 1 else 'lightgray' 
                         for i in range(len(noms))]
            
            bars = ax.barh(noms[::-1], scores_vals[::-1], color=colors_bar[::-1], edgecolor='black')
            ax.set_xlabel("Score d'adéquation")
            ax.set_title("Classement des agents pour ce ticket")
            
            # Ajouter les valeurs
            for i, (bar, score) in enumerate(zip(bars, scores_vals[::-1])):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2, 
                       f"{score:.0f}", va='center', fontsize=10)
            
            st.pyplot(fig)
            
            # Tableau détaillé
            st.dataframe(scores_df[["agent", "score", "expertise_match", "meme_service", "efficacite"]], 
                        use_container_width=True)
            
            # ═══════════════════════════════════════════════════════════
            # SYNTHÈSE FINALE
            # ═══════════════════════════════════════════════════════════
            st.markdown("---")
            st.subheader("📋 Synthèse et recommandation")
            
            # Déterminer la priorité
            if urgence_user == "Haute" or (type_detecte == "reseau" and users_impact > 50):
                prio_finale = "HAUTE"
                color_prio = "red"
                action = "🔴 Assigner IMMÉDIATEMENT - Ticket critique"
            elif urgence_user == "Moyenne" or users_impact > 10:
                prio_finale = "MOYENNE"
                color_prio = "orange"
                action = "⚠️ Assigner aujourd'hui - Ticket prioritaire"
            else:
                prio_finale = "FAIBLE"
                color_prio = "green"
                action = "✅ Assigner dans les 48h - Ticket standard"
            
            st.success(f"""
            ### 🎯 Ticket à créer
            
            | Élément | Valeur |
            |---------|--------|
            | **Description** | {user_description[:60]}... |
            | **Type** | {type_detecte.upper()} |
            | **Service demandeur** | {service_user} |
            | **Utilisateurs impactés** | {users_impact} |
            | **Agent assigné** | **{best_agent}** |
            | **Temps estimé** | **{temps_estime:.1f} heures** |
            | **Priorité** | **{prio_finale}** |
            
            ### 📋 Action recommandée
            {action}
            
            ### 👤 Contact
            **{best_agent}** ({best_info['service']}) - Expertise : {', '.join(best_info['expertise'])}
            """)
            
            # Bouton d'assignation
            if st.button("✅ Confirmer l'assignation", key="confirm_assign"):
                st.balloons()
                st.success(f"✅ Ticket assigné à **{best_agent}** avec succès !")
                st.info(f"📧 Notification envoyée à {best_agent} et au service {best_info['service']}")
