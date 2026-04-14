import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime, date

# Configuration de la page
st.set_page_config(page_title="IA Détective - Fraude Assurance", layout="wide")

# Chemins des modèles (relatifs au script)
BASE_DIR = os.path.dirname(__file__)
xgb_model_path = os.path.join(BASE_DIR, 'xgb_best_model.joblib')
cat_model_path = os.path.join(BASE_DIR, 'cat_best_model.joblib')

# Chargement des modèles
@st.cache_resource(show_spinner=True)
def load_models():
    if not os.path.exists(xgb_model_path) or not os.path.exists(cat_model_path):
        st.error(f"Modèles non trouvés. Assurez-vous qu'ils sont dans : {BASE_DIR}")
        st.stop()
    xgb_model = joblib.load(xgb_model_path)
    cat_model = joblib.load(cat_model_path)
    return xgb_model, cat_model

xgb_model, cat_model = load_models()

# --- BARRE LATÉRALE ---
st.sidebar.header("🛡️ Paramètres de Détection")
st.sidebar.markdown("""
Ajustez la sensibilité du système. 
Un seuil plus bas augmente le **Rappel** (capture plus de fraudes).
""")
threshold = st.sidebar.slider("Seuil de Décision (Sensibilité)", 0.05, 0.95, 0.25, step=0.05)
st.sidebar.info(f"Seuil actuel : {threshold * 100:.0f}%")

st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ Intelligence Automatisée")
st.sidebar.write("Les 'Signaux Experts' sont désormais calculés automatiquement par l'IA.")

# --- CORPS DE L'APP ---
st.title("🕵️ IA Détective : Analyse de Fraude Automobile")
st.markdown("Ce système utilise des modèles prédictifs et une analyse experte automatisée.")

# Formulaire utilisateur
st.header("📋 Dossier de Sinistre")
col1, col2 = st.columns(2)

input_dict_raw = {}

with col1:
    st.subheader("💰 Informations Financières")
    input_dict_raw['total_claim_amount'] = st.number_input("Montant total réclamé ($)", min_value=0.0, value=75000.0)
    input_dict_raw['policy_deductable'] = st.number_input("Franchise ($)", min_value=0, value=1000)
    input_dict_raw['policy_annual_premium'] = st.number_input("Prime annuelle ($)", min_value=0.0, value=1200.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🗓️ Dates Clés")
    policy_date = st.date_input("Date de début de police", value=date(2015, 1, 1))
    incident_date = st.date_input("Date de l'accident", value=date(2015, 1, 15))
    incident_hour = st.slider("Heure de l'accident", 0, 23, 2, help="Nuit = 23h à 5h")

with col2:
    st.subheader("💥 Circonstances de l'Incident")
    input_dict_raw['incident_severity'] = st.selectbox("Gravité", ['MAJOR DAMAGE', 'MINOR DAMAGE', 'TOTAL LOSS', 'TRIVIAL DAMAGE'])
    input_dict_raw['incident_type'] = st.selectbox("Type d'accident", ['SINGLE VEHICLE COLLISION', 'MULTI-VEHICLE COLLISION', 'PARKED CAR', 'VEHICLE THEFT'])
    input_dict_raw['police_report_available'] = st.selectbox("Rapport de police", ['YES', 'NO'])
    input_dict_raw['witnesses'] = st.number_input("Nombre de témoins", min_value=0, value=0)
    input_dict_raw['bodily_injuries'] = st.number_input("Blessures corporelles", min_value=0, value=0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("👤 Profil Client")
    input_dict_raw['age'] = st.number_input("Âge de l'assuré", min_value=18, max_value=100, value=22)
    input_dict_raw['months_as_customer'] = st.number_input("Ancienneté (mois)", min_value=0, value=1)
    input_dict_raw['insured_hobbies'] = st.selectbox("Hobby de l'assuré", [
        'CHESS', 'CROSS-FIT', 'YACHTING', 'BOARD-GAMES', 'POLO', 'READING', 'BASE-JUMPING', 'HIKING', 'PAINTBALL',
        'SKYDIVING', 'VIDEO-GAMES', 'SLEEPING', 'EXERCISE', 'BINGE-WATCHING', 'DANCING', 'GOLF', 'KAYAKING', 'CAMPING'
    ])

# --- LOGIQUE D'AUTOMATISATION ---
# 1. Calcul des features de base
claim_delay_days = (incident_date - policy_date).days
incident_month = incident_date.month
policy_year = policy_date.year

# 2. Calcul des 6 Relations Expertes
# Rel 1 : Early Incident (< 30 jours)
rel1 = 1 if claim_delay_days < 30 else 0

# Rel 2 : Gravité faible + montant élevé (> 70k)
rel2 = 1 if (input_dict_raw['incident_severity'] in ['MINOR DAMAGE', 'TRIVIAL DAMAGE'] and input_dict_raw['total_claim_amount'] > 70000) else 0

# Rel 3 : Proximité franchise (Franchise < Montant < Franchise + 1000)
rel3 = 1 if (input_dict_raw['policy_deductable'] < input_dict_raw['total_claim_amount'] < input_dict_raw['policy_deductable'] + 1000) else 0

# Rel 4 : Pas de rapport de police + montant élevé (> 70k)
rel4 = 1 if (input_dict_raw['police_report_available'] == 'NO' and input_dict_raw['total_claim_amount'] > 70000) else 0

# Rel 5 : Petite prime (< 1100) + énorme réclamation (> 70k)
rel5 = 1 if (input_dict_raw['policy_annual_premium'] < 1100 and input_dict_raw['total_claim_amount'] > 70000) else 0

# Rel 6 : Nuit (23h-5h) + Single Vehicle
is_night = (incident_hour >= 23 or incident_hour <= 5)
rel6 = 1 if (is_night and input_dict_raw['incident_type'] == 'SINGLE VEHICLE COLLISION') else 0

# Préparation du dictionnaire final pour le modèle
final_input = {
    'total_claim_amount': input_dict_raw['total_claim_amount'],
    'policy_deductable': input_dict_raw['policy_deductable'],
    'policy_annual_premium': input_dict_raw['policy_annual_premium'],
    'incident_severity': input_dict_raw['incident_severity'],
    'police_report_available': input_dict_raw['police_report_available'],
    'incident_type': input_dict_raw['incident_type'],
    'witnesses': input_dict_raw['witnesses'],
    'bodily_injuries': input_dict_raw['bodily_injuries'],
    'claim_delay_days': claim_delay_days,
    'incident_month': incident_month,
    'policy_year': policy_year,
    'rel1_early_incident': rel1,
    'rel2_severity_amount_mismatch': rel2,
    'rel3_just_above_deductible': rel3,
    'rel4_no_police_high_amount': rel4,
    'rel5_low_premium_high_claim': rel5,
    'rel6_night_single_vehicle': rel6,
    'age': input_dict_raw['age'],
    'months_as_customer': input_dict_raw['months_as_customer'],
    'insured_hobbies': input_dict_raw['insured_hobbies']
}

st.markdown("---")
# Affichage des signaux détectés
st.subheader("🔎 Analyse Automatique des Signaux")
detected_signals = []
if rel1: detected_signals.append("⚠️ ACCIDENT TRÈS PRÉCOCE (<30j)")
if rel2: detected_signals.append("⚠️ INCOHÉRENCE GRAVITÉ/MONTANT")
if rel3: detected_signals.append("⚠️ PROXIMITÉ FRANCHISE")
if rel4: detected_signals.append("⚠️ GROS MONTANT SANS POLICE")
if rel5: detected_signals.append("⚠️ PRIME FAIBLE / MONTANT ÉLEVÉ")
if rel6: detected_signals.append("🌙 ACCIDENT DE NUIT (VÉHICULE SEUL)")

if detected_signals:
    for signal in detected_signals:
        st.warning(signal)
else:
    st.info("Aucun signal de fraude flagrant détecté automatiquement.")

if st.button("🚀 LANCER L'ANALYSE IA FINALE", use_container_width=True):
    X_input = pd.DataFrame([final_input])
    # Prédictions
    proba_xgb = xgb_model.predict_proba(X_input)[0, 1]
    proba_cat = cat_model.predict_proba(X_input)[0, 1]

    st.markdown("---")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.subheader("🔥 Risque XGBoost")
        st.progress(float(proba_xgb))
        st.write(f"Probabilité : **{float(proba_xgb):.1%}**")
        if float(proba_xgb) >= threshold:
            st.error("🚨 ALERTE : HAUT RISQUE DE FRAUDE")
        else:
            st.success("✅ DOSSIER LÉGITIME")

    with res_col2:
        st.subheader("🧊 Risque CatBoost")
        st.progress(float(proba_cat))
        st.write(f"Probabilité : **{float(proba_cat):.1%}**")
        if float(proba_cat) >= threshold:
            st.error("🚨 ALERTE : HAUT RISQUE DE FRAUDE")
        else:
            st.success("✅ DOSSIER LÉGITIME")
