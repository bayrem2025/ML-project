# 🕵️ Détection de Fraude — Assurance Automobile

Système de prédiction de fraude basé sur le Machine Learning, avec une interface Streamlit pour l'analyse en temps réel.

---

## 📌 Contexte

La fraude à l'assurance automobile représente un coût majeur pour les compagnies. Ce projet construit un modèle capable de détecter automatiquement les dossiers de sinistres frauduleux à partir des données disponibles au moment de la déclaration.

- **Type** : Supervised Learning — Classification Binaire — Non Linéaire
- **Dataset** : 1000 dossiers de sinistres automobiles (Kaggle)
- **Cible** : `fraud_reported` → Y (fraude) / N (légitime)
- **Déséquilibre** : 75.3% légitimes vs 24.7% fraudes

---

## 📁 Structure du projet

```
├── data/
│   └── insurance_claims.csv          # Dataset brut
├── process/
│   ├── etape1b_analyse_visuelle.ipynb  # EDA & visualisations
│   ├── etape2_data_cleaning.ipynb      # Nettoyage des données
│   ├── etape3_modelisation.ipynb       # Modélisation ML
│   ├── insurance_claims_cleaned.csv    # Dataset nettoyé
│   ├── xgb_best_model.joblib           # Modèle XGBoost sauvegardé
│   ├── cat_best_model.joblib           # Modèle CatBoost sauvegardé
│   └── app.py                          # Application Streamlit
├── draft/
│   ├── brouillon_projet.ipynb          # Notebook brouillon complet
│   └── brouillons.md                   # Notes & explications
├── requirements.txt
└── README.md
```

---

## 🔄 Pipeline du projet

```
Données brutes
    ↓
Étape 1 — EDA (analyse visuelle, distributions, corrélations)
    ↓
Étape 2 — Data Cleaning (suppression colonnes, imputation, standardisation)
    ↓
Étape 3 — Feature Engineering (6 signaux experts)
    ↓
Pipeline ML (ColumnTransformer + Classifier)
    ↓
Tuning (RandomizedSearchCV, scoring=roc_auc)
    ↓
Optimisation du seuil (F1 max → 0.54)
    ↓
Application Streamlit
```

---

## 🧠 Modèles testés

| Modèle | F1 (fraude) | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.676 | 0.815 |
| Decision Tree | 0.654 | 0.722 |
| Random Forest | 0.543 | 0.840 |
| XGBoost (tuné) | **0.733** | **0.848** |
| CatBoost (tuné) | 0.722 | **0.851** |

**Modèles retenus : XGBoost + CatBoost**

---

## ⚙️ Feature Engineering — 6 Signaux Experts

| Variable | Logique métier |
|---|---|
| `rel1_early_incident` | Accident < 30 jours après souscription |
| `rel2_severity_amount_mismatch` | Gravité faible + montant élevé (> Q75) |
| `rel3_just_above_deductible` | Montant juste au-dessus de la franchise |
| `rel4_no_police_high_amount` | Pas de rapport police + montant élevé |
| `rel5_low_premium_high_claim` | Prime faible (< Q25) + réclamation élevée |
| `rel6_night_single_vehicle` | Accident de nuit (23h-5h) + véhicule seul |

---

## 🚀 Lancer l'application Streamlit

```bash
cd process
streamlit run app.py
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 📊 Résultats finaux (XGBoost tuné)

- **Accuracy** : 84.5%
- **Recall fraude** : 85.9% ← priorité : capturer un maximum de fraudes
- **Precision fraude** : 63.9%
- **F1 fraude** : 0.733
- **ROC-AUC** : 0.848

---

## 👤 Auteur

**BENKHOUD Mohamed Bay