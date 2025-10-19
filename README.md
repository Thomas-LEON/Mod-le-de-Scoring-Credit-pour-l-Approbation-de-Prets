# Projet : Modèle de Scoring Crédit pour l'Approbation de Prêts

**Auteur :** Thomas LEON
**Contact :** https://www.linkedin.com/in/thomas-leon-893316262/
**Repo GitHub :** https://github.com/Thomas-LEON/Mod-le-de-Scoring-Credit-pour-l-Approbation-de-Prets

---

## 1. Problématique Métier

Dans le secteur bancaire, l'octroi de prêts comporte un risque financier : le risque de défaut de paiement du client. Une décision incorrecte peut entraîner des pertes significatives.

**L'objectif de ce projet est de construire un modèle de machine learning (scoring) capable de prédire la probabilité qu'un demandeur de prêt fasse défaut.**

Ce score de crédit vise à :
* **Fiabiliser** et **automatiser** la décision d'octroi de prêt.
* **Minimiser** le risque financier pour l'institution prêteuse.
* **Identifier** les clients les plus risqués et, inversement, les clients fiables.

Il s'agit d'un problème de **classification binaire** : le client va-t-il rembourser (0) ou faire défaut (1) ?

## 2. Le Dataset

Les données utilisées proviennent de Credit scoring for borrowers in bank sur kaggle.

* **Source :** https://www.kaggle.com/datasets/kapturovalexander/bank-credit-scoring
* **Volume :** [Nombre] lignes et 17 colonnes.
* **Description :** Dataset may have originated from an outside source Bank Marketing. UC Irvine
(https://archive.ics.uci.edu/dataset/222/bank+marketing)

### Déséquilibre des classes
Ce dataset présente un fort déséquilibre, typique des problèmes de fraude ou de risque :
* **Clients fiables (Cible 0) :** 98 %
* **Clients en défaut (Cible 1) :** 2 %

La gestion de ce déséquilibre est un enjeu majeur du projet (par exemple, via des techniques de ré-échantillonnage comme SMOTE ou en ajustant les poids des classes).

## 3. Méthodologie du Projet

Le projet suit un pipeline Data Science classique, de l'exploration à la modélisation.

### Étape 1 : Nettoyage et Exploration (EDA)
* **Nettoyage :** Traitement des valeurs manquantes (imputation par la médiane, le mode, ou création de catégories "Inconnu"). Correction des types de données et gestion des valeurs aberrantes (outliers).
* **Analyse Exploratoire (EDA) :** Analyse de la distribution de la variable cible `TARGET` et des variables principales (ex: `AMT_CREDIT`, `DAYS_BIRTH`). Visualisation des corrélations pour identifier les premières pistes.

### Étape 2 : Feature Engineering
La création de nouvelles variables (features) est cruciale pour améliorer la performance du modèle :
* **Ratios financiers :** Création de variables clés comme `CREDIT_INCOME_PERCENT` (montant du crédit / revenus) ou `ANNUITY_INCOME_PERCENT` (annuités / revenus).
* **Variables catégorielles :** Encodage des variables textuelles (ex: `NAME_CONTRACT_TYPE`) en utilisant le One-Hot Encoding ou le Label Encoding.
* **Normalisation :** Standardisation des variables numériques (avec `StandardScaler`) pour les modèles sensibles aux échelles de valeur (comme la Régression Logistique).

### Étape 3 : Modélisation et Évaluation
Plusieurs modèles ont été testés et comparés :

1.  **Régression Logistique (Baseline) :** Un modèle simple, rapide et interprétable, servant de référence.
2.  **Random Forest :** Un modèle d'ensemble (bagging) robuste et performant.
3.  **LightGBM / XGBoost :** Modèles de Gradient Boosting, souvent les plus performants sur ce type de données tabulaires.

#### Métrique d'évaluation
En raison du déséquilibre des classes, l'**accuracy** n'est pas une bonne métrique. Nous avons privilégié l'**AUC-ROC (Area Under the Curve)**, qui mesure la capacité du modèle à discriminer les deux classes, ainsi que le **Rappel (Recall)** sur la classe "défaut" (pour s'assurer de bien détecter les clients à risque).

## 4. Résultats et Interprétabilité

Le modèle [Nom du meilleur modèle, ex: LightGBM] a obtenu les meilleures performances :

| Modèle | AUC (Jeu de test) | Rappel (Classe 1) | Précision (Classe 1) |
| :--- | :---: | :---: | :---: |
| Régression Logistique | [Score, ex: 0.72] | [Score] | [Score] |
| Random Forest | [Score, ex: 0.76] | [Score] | [Score] |
| **LightGBM** | **[Score, ex: 0.79]** | **[Score]** | **[Score]** |

### Interprétabilité (Feature Importance)
L'analyse de l'importance des variables (via SHAP ou la feature importance native du modèle) montre que les principaux prédicteurs du risque de défaut sont :

1.  `[Variable 1, ex: EXT_SOURCE_2]`
2.  `[Variable 2, ex: CREDIT_INCOME_PERCENT]`
3.  `[Variable 3, ex: DAYS_BIRTH]`
4.  `[Variable 4, ex: ...]`

*(Bonus : Insérez ici un graphique de Feature Importance si vous le pouvez)*

## 5. Déploiement (Exemple d'utilisation)

Pour démontrer l'utilisation concrète du modèle, une API simple a été développée avec **FastAPI**. Le modèle entraîné a été sauvegardé (`credit_scoring.pkl`).

### Installation
Pour lancer le projet et l'API :

```bash
# 1. Cloner le dépôt
git clone [URL_DE_VOTRE_REPO]
cd [NOM_DU_REPO]

# 2. Créer un environnement virtuel et l'activer
python -m venv venv
source venv/bin/activate  # (ou .\venv\Scripts\activate pour Windows)

# 3. Installer les dépendances
pip install -r requirements.txt
# (Pensez à générer ce fichier avec 'pip freeze > requirements.txt')

# 4. Lancer l'API
cd api
uvicorn main:app --reload
