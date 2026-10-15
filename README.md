# Classification des intentions bancaires — BANKING77

Projet organisé à partir du notebook fourni. Comparaison de TF-IDF + régression logistique et de DistilBERT fine-tuné, sur 77 catégories.

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
3. `03_distilbert_comparaison.ipynb` : fine-tuning, comparaison, sauvegarde et test final facultatif.

Chaque notebook peut être ouvert dans un noyau neuf : les échanges passent par les fichiers sauvegardés. Si les données ou le découpage changent, relancer les trois notebooks dans l'ordre pour éviter de comparer des résultats périmés.

## Fichiers générés à l'exécution

- `data/processed/` : train, validation, test et catégories.
- `models/` : baseline, checkpoints et modèle DistilBERT final.
- `results/` : métriques, prédictions et graphiques.

## Ce qui est conservé et ajouté

Le nettoyage, le split 80/20 avec graine 42, la factorisation des catégories, les modèles, les hyperparamètres et les métriques du code d'origine sont conservés. Les imports sont répartis entre notebooks. L'appel `train_data.head` devient `train_data.head()` et les chemins de sortie sont regroupés.

Ajouts : explications en français, contrôles d'entrée, graphiques d'exploration et de comparaison, échanges entre notebooks, sauvegarde des modèles, examen des erreurs et test final désactivé par défaut.

## Ressources et validation

Le premier téléchargement de DistilBERT nécessite Internet. Un GPU est conseillé pour les 10 époques ; le code peut aussi utiliser le CPU. La variante `uncased` choisie dans le notebook initial est conservée.

La structure des notebooks et la syntaxe des cellules ont été vérifiées. Le parcours préparation → baseline a été vérifié avec un petit jeu synthétique de 77 catégories, uniquement pour contrôler les échanges entre fichiers. Ces données et scores synthétiques ne font pas partie du projet livré. Le véritable entraînement, le téléchargement et le parcours DistilBERT restent à exécuter avec les CSV originaux et les dépendances installées. Aucun résultat expérimental n'est annoncé.

Les dépendances ne constituent pas un environnement verrouillé : la compatibilité de l'environnement complet reste à vérifier lors de l'installation. Les versions Transformers sont bornées à la série 4 pour conserver les interfaces du notebook.
