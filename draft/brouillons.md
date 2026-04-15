# Brouillons — Projet Détection de Fraude Assurance Automobile

---

## CONTEXTE & PROBLÉMATIQUE

- La fraude à l'assurance automobile coûte des milliards chaque année
- Des assurés déclarent de faux sinistres ou exagèrent les dommages
- Objectif : construire un modèle ML pour détecter automatiquement les fraudes
- Type de problème : Supervised Learning → Classification binaire → Non linéaire

---

## LE DATASET

- Source : Kaggle — insurance_claims.csv
- 1000 dossiers de sinistres automobiles
- 40 variables au départ
- Variable cible : fraud_reported → Y (fraude) ou N (légitime)
- Déséquilibre : 753 légitimes (75.3%) vs 247 fraudes (24.7%)

---

## ÉTAPE 1 — ANALYSE VISUELLE (EDA)

Objectif : explorer les données pour comprendre les patterns de fraude

Ce qu'on a découvert :
- Déséquilibre des classes : 75.3% légitimes vs 24.7% fraudes
- Le montant réclamé est un signal fort → fraudes ont des montants plus élevés
- Les hobbies corrèlent avec la fraude (chess, cross-fit, yachting)
- Valeurs manquantes déguisées en "?" dans collision_type, property_damage, police_report_available
- Colonne parasite _c39 entièrement vide

Visualisations réalisées :
- Camembert répartition fraude/légitime
- Boxplot montant réclamé par statut
- Matrice de corrélation
- Taux de fraude par hobby

---

## ÉTAPE 2 — DATA CLEANING

Objectif : nettoyer les données pour les rendre exploitables

Actions réalisées :

1. Suppression de 4 colonnes inutiles
   - policy_number → identifiant unique, aucune valeur prédictive
   - incident_location → adresse trop spécifique
   - insured_zip → code postal trop granulaire
   - _c39 → colonne vide parasite
   - Résultat : 40 colonnes → 36 colonnes

2. Remplacement des "?" par NaN
   - df.replace('?', np.nan)

3. Traitement des valeurs manquantes
   - authorities_contacted : 91 NaN (9.1%) → imputé par mode

4. Vérification des doublons → aucun détecté

5. Conversion des dates
   - policy_bind_date et incident_date → datetime

6. Standardisation des catégorielles
   - 18 colonnes → .str.strip().str.upper()
   - 'Minor Damage' → 'MINOR DAMAGE'

Résultat final : 1000 lignes × 36 colonnes → insurance_claims_cleaned.csv

---

## ÉTAPE 3 — MODÉLISATION

### Feature Engineering (6 signaux experts)

| Variable | Logique |
|---|---|
| rel1_early_incident | Accident < 30 jours après souscription |
| rel2_severity_amount_mismatch | Gravité faible + montant > Q75 |
| rel3_just_above_deductible | Montant entre franchise et franchise+1000$ |
| rel4_no_police_high_amount | Pas de rapport police + montant élevé |
| rel5_low_premium_high_claim | Prime < Q25 + montant > Q75 |
| rel6_night_single_vehicle | Nuit (23h-5h) + véhicule seul |

### Préparation

- 20 features sélectionnées
- Split : 60% train (600) / 40% test (400), stratifié
- Pipeline preprocessing : imputation médiane (num) + imputation mode + OneHotEncoding (cat)

### Modèles testés

| Modèle | F1 fraude | ROC-AUC |
|---|---|---|
| Logistic Regression | 0.676 | baseline |
| Decision Tree | ~0.68 | — |
| Random Forest | ~0.70 | — |
| XGBoost (tuné) | 0.733 | 0.848 |
| CatBoost (tuné) | 0.722 | 0.851 |

### Pourquoi XGBoost et CatBoost ?

- Algorithmes de gradient boosting → arbres séquentiels, chaque arbre corrige les erreurs du précédent
- Gèrent nativement le déséquilibre des classes
- XGBoost : scale_pos_weight = ratio non-fraude/fraude ≈ 3
- CatBoost : auto_class_weights='Balanced' + adapté aux variables catégorielles
- Meilleurs résultats sur toutes les métriques

### Tuning (RandomizedSearchCV)

- scoring='roc_auc', cv=3, n_iter=10
- Meilleurs params XGBoost : n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8
- Meilleurs params CatBoost : iterations=100, depth=4, learning_rate=0.05, l2_leaf_reg=3

### Optimisation du seuil

- Seuil par défaut : 0.5
- Seuil optimal (F1 max) : 0.54
- En détection de fraude, on préfère plus de faux positifs que rater une vraie fraude

### Résultats finaux XGBoost

- Accuracy : 84.5%
- Precision fraude : 63.9%
- Recall fraude : 85.9%
- F1 fraude : 0.733
- ROC-AUC : 0.848

---

## LES PIPELINES

### Pourquoi un Pipeline ?
- Enchaîne preprocessing + modèle dans un seul objet
- Évite le data leakage (tout appris sur le train uniquement)
- Modèle sauvegardé contient tout → prêt à l'emploi dans Streamlit

### Structure

Pipeline numérique :
  SimpleImputer(median) → remplace NaN par la médiane

Pipeline catégoriel :
  SimpleImputer(most_frequent) → remplace NaN par la valeur la plus fréquente
  OneHotEncoder → transforme chaque catégorie en colonnes 0/1

Pipeline final :
  ColumnTransformer (num + cat) → XGBClassifier / CatBoostClassifier

### OneHotEncoding exemple
incident_severity = "MINOR DAMAGE"
→ col_MINOR_DAMAGE=1, col_MAJOR_DAMAGE=0, col_TOTAL_LOSS=0, col_TRIVIAL=0

### Data Leakage
- Si on calcule la médiane sur tout le dataset avant le split → le modèle voit des infos du test
- Le Pipeline calcule la médiane uniquement sur le train et l'applique sur le test

---

## MÉTRIQUES UTILISÉES

- classification_report → precision, recall, F1-score, support
- confusion_matrix → vrais positifs, faux positifs, vrais négatifs, faux négatifs
- roc_auc_score → métrique principale pour le tuning

Pourquoi ROC-AUC et pas accuracy ?
- Avec 75% de légitimes, un modèle qui prédit toujours "pas de fraude" a 75% d'accuracy
- ROC-AUC mesure la capacité à distinguer les deux classes indépendamment du seuil
- Recall fraude prioritaire : mieux vaut investiguer un dossier légitime que rater une fraude

---

## APPLICATION STREAMLIT

- Formulaire de saisie d'un dossier de sinistre
- Calcul automatique des 6 signaux experts
- Affichage des alertes détectées
- Prédiction XGBoost + CatBoost avec probabilités
- Seuil de décision ajustable (slider 0.05 à 0.95)
- Modèles chargés via joblib depuis les fichiers .joblib

---

## PROBLÈMES RENCONTRÉS

1. Déséquilibre des classes → résolu par scale_pos_weight et auto_class_weights
2. Incohérence majuscules/minuscules entre notebook et app.py
   - Notebook entraîne sur 'Minor Damage' mais app envoie 'MINOR DAMAGE'
3. Seuils des signaux experts hardcodés dans app.py (70000$, 1100$)
   - Dans le notebook ils sont calculés dynamiquement avec quantile()
4. Performances limitées sur la classe fraude (F1 = 0.733)
   - Dataset petit (1000 cas)

---

## RÉPONSES AUX QUESTIONS DU PROF

Q: C'est quoi un pipeline ?
R: Une chaîne d'étapes automatiques qui s'exécutent dans l'ordre. On enchaîne preprocessing et modèle dans un seul objet pour éviter le data leakage et simplifier le déploiement.

Q: Pourquoi XGBoost et CatBoost ?
R: Ce sont des algorithmes de gradient boosting qui donnent les meilleurs résultats sur données tabulaires déséquilibrées. Ils gèrent nativement le déséquilibre et CatBoost est particulièrement adapté aux variables catégorielles.

Q: C'est de la classification ou régression ?
R: Classification binaire — on prédit une classe (fraude/pas fraude), pas une valeur continue.

Q: C'est du supervised learning ?
R: Oui, chaque dossier a une étiquette connue (fraud_reported = Y/N). Le modèle apprend à partir de ces exemples labellisés.

Q: Pourquoi ROC-AUC comme métrique ?
R: Parce que le dataset est déséquilibré. L'accuracy serait trompeuse — un modèle qui prédit toujours "pas de fraude" aurait 75% d'accuracy. ROC-AUC mesure la vraie capacité discriminante du modèle.

Q: C'est quoi le data leakage ?
R: Quand le modèle apprend des infos du test pendant l'entraînement. Le Pipeline l'évite en calculant les transformations uniquement sur le train.
