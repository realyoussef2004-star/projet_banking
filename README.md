# Classification des intentions bancaires — BANKING77

Projet organisé en trois notebooks pour comparer TF-IDF + régression logistique et DistilBERT fine-tuné sur 77 catégories d’intentions bancaires.

Sur le même jeu de validation, DistilBERT atteint **91,55 % d’accuracy** et **91,61 % de macro-F1**, contre **84,79 %** et **84,08 %** pour la baseline TF-IDF.

## Installation et lancement

Utiliser un environnement Python dédié, puis :

```bash
python -m pip install -r requirements.txt
python -m jupyter lab
```

Lancer Jupyter depuis ce dossier. Ajouter les fichiers originaux `train.csv` et `test.csv` dans `banking_data/`. Ils doivent contenir les colonnes `text` et `category`. Les données ne sont pas incluses dans cette livraison.

Exécuter de haut en bas, dans cet ordre :

1. `01_preparation_donnees.ipynb` : chargement, nettoyage, exploration, découpage stratifié et sauvegarde.
2. `02_modele_tfidf.ipynb` : entraînement, évaluation, analyse des erreurs et sauvegarde de la baseline.
3. `03_distilbert_comparaison.ipynb` : fine-tuning, comparaison et sauvegarde de DistilBERT ; une section est prévue pour le test final facultatif.

Les échanges entre notebooks passent par les fichiers sauvegardés. Les étapes de préparation, de baseline et de fine-tuning peuvent ainsi être lancées dans des noyaux distincts, dans cet ordre. Si les données ou le découpage changent, relancer les trois notebooks pour éviter de comparer des résultats périmés.

**État du test final :** la dernière cellule du notebook 03 doit être corrigée avant utilisation : `est_df` au lieu de `test_df`, caractère `=` superflu devant `baseline_test_predictions`, et absence de condition `if RUN_FINAL_TEST:`. Le drapeau est actuellement défini à `True`, mais ne contrôle pas l’exécution. Aucun résultat de test final n’est enregistré dans les notebooks fournis.

## Données et protocole expérimental

Les sorties du notebook de préparation donnent les effectifs suivants :

| Étape | Nombre de messages |
| --- | ---: |
| Fichier d’entraînement initial | 10 003 |
| Après nettoyage | 9 992 |
| Entraînement | 7 993 |
| Validation | 1 999 |
| Test séparé | 3 080 |

Le nettoyage retire **11 lignes au total**, dont **7 textes également présents dans le test**. Aucun texte associé à plusieurs catégories n’a été détecté. Les textes transmis aux modèles restent ceux d’origine ; une version normalisée sert à repérer les doublons.

Le jeu nettoyé est séparé en **80 % d’entraînement et 20 % de validation**, avec stratification par catégorie et graine aléatoire **42**. Les deux modèles sont comparés sur les mêmes 1 999 messages de validation.

## Résultats sur la validation

Les scores ci-dessous sont ceux enregistrés dans les notebooks fournis et confirmés par la capture de comparaison.

| Modèle | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| TF-IDF + régression logistique | 84,79 % | 84,08 % |
| DistilBERT fine-tuné | **91,55 %** | **91,61 %** |

L’**accuracy** mesure la proportion de prédictions correctes. Le **macro-F1** est la moyenne des scores F1 des 77 catégories, avec le même poids pour chaque catégorie.

DistilBERT améliore l’accuracy de **6,76 points de pourcentage** et le macro-F1 de **7,53 points** par rapport à la baseline. Il obtient donc les meilleurs scores sur cette validation. Cette comparaison ne constitue pas une évaluation finale indépendante : la validation sert également à sélectionner le meilleur checkpoint de DistilBERT.

### Configuration et temps d’entraînement

| Élément | TF-IDF + régression logistique | DistilBERT fine-tuné |
| --- | --- | --- |
| Représentation | TF-IDF, unigrammes et bigrammes | `distilbert/distilbert-base-uncased` |
| Paramètres principaux | `C=1.0`, `solver="lbfgs"`, `max_iter=1000` | Longueur maximale 128, taux d’apprentissage `2e-5`, 10 époques, weight decay `0.01` |
| Lots | Sans objet | Taille 8, accumulation sur 2 étapes |
| Sélection | Modèle entraîné sur le train | Meilleur checkpoint selon le macro-F1 de validation |
| Durée observée | Environ 10 secondes sur CPU | Environ 12,5 minutes sur GPU |

Le GPU détecté lors de l’exécution de DistilBERT est une **NVIDIA GeForce RTX 4060 Laptop GPU**, avec CUDA disponible. Ces durées dépendent du matériel et de l’environnement ; elles ne constituent pas une mesure comparative à matériel identique.

### Analyse des erreurs

La baseline commet **304 erreurs sur 1 999 messages** de validation. Les exemples d’erreurs de DistilBERT montrent notamment des confusions entre intentions proches : paiement par carte et retrait refusés, frais de rechargement par carte et par virement, ou taux de change général et taux appliqué aux retraits.

Les prédictions de validation sont sauvegardées pour permettre une analyse détaillée par catégorie.

## Fichiers générés à l’exécution

- `data/processed/` : données d’entraînement, de validation et de test, ainsi que `categories.json`.
- `models/` : `baseline.joblib`, checkpoints DistilBERT et modèle final dans `distilbert_final/`.
- `results/` : résumé des données, métriques, prédictions et graphiques.

La comparaison sur la validation est enregistrée dans :

- `results/comparison_validation.csv` : tableau des scores.
- `results/comparison_validation.png` : graphique comparatif.
- `results/baseline_metrics.json` et `results/distilbert_metrics.json` : métriques et durées.
- `results/baseline_validation_predictions.csv` et `results/distilbert_validation_predictions.csv` : prédictions détaillées.

Après correction et exécution de la cellule de test final, la comparaison sur le test sera enregistrée dans `results/comparison_test.csv`.

## Ce qui est conservé et ajouté

Le nettoyage, le split 80/20 avec graine 42, la factorisation des catégories, les modèles, les hyperparamètres et les métriques du code d’origine sont conservés. Les imports sont répartis entre notebooks. L’appel `train_data.head` devient `train_data.head()` et les chemins de sortie sont regroupés.

Ajouts : explications en français, contrôles d’entrée, graphiques d’exploration et de comparaison, échanges entre notebooks, sauvegarde des modèles, examen des erreurs et section de test final facultatif.

## Ressources et validation

Les notebooks fournis contiennent les sorties d’une exécution de la préparation, de la baseline et du fine-tuning de DistilBERT avec les effectifs et scores présentés ci-dessus. Ces résultats remplacent la mention initiale selon laquelle seul un contrôle sur données synthétiques avait été réalisé. Ils n’ont pas été recalculés lors de la mise à jour de ce README.

Le premier téléchargement de DistilBERT nécessite Internet. Un GPU est conseillé pour les 10 époques ; le code peut aussi utiliser le CPU. La variante `uncased` choisie dans le notebook initial est conservée.

Les dépendances ne constituent pas un environnement verrouillé : la compatibilité de l’environnement complet reste à vérifier lors de l’installation. Le README initial indique que Transformers est borné à la série 4 ; le fichier `requirements.txt` n’étant pas joint aux éléments examinés, cette contrainte et sa compatibilité avec les interfaces utilisées restent à vérifier.

L’évaluation finale sur les 3 080 messages de test reste à effectuer après correction de la cellule concernée et fixation des réglages. Les scores de validation ne doivent pas être présentés comme des scores de test.
