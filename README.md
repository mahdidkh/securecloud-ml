# SecureCloud AI — Module ML (détection d'anomalies)

## Contenu
| Fichier | Rôle |
| --- | --- |
| `generate_dataset.py` | Génère `dataset.csv` (synthétique, graine 42) — 7 variables par fenêtre de 5 min |
| `train.ipynb` | Pipeline complet : inventaire, dédoublonnage, baseline, comparaison, seuil, test, contamination, export |
| `ml_utils.py` | Baseline Mahalanobis, choix du seuil, métriques |
| `inference.py` | Interface d'inférence pour le backend (`AnomalyDetector.predict`) |
| `model.joblib`, `metrics.json` | Modèle exporté et métriques |
| `results_table.csv`, `contamination_results.csv`, `figures/` | Résultats du notebook |
| `RAPPORT.md` | Rapport de validation (baseline + protocole d'évaluation) |

## Exécution
```bash
pip install -r requirements.txt
python generate_dataset.py          # optionnel : le notebook le fait si dataset.csv est absent
jupyter nbconvert --to notebook --execute --inplace train.ipynb
```

## Protocole
Entraînement sur normal uniquement → réglages et seuil sur validation → test utilisé une seule fois.
Doublons supprimés dans et entre ensembles ; le pool `contam` ne sert qu'à l'expérience de contamination.

## Changement de contrat
La variable `mean_event_gap_s` (secondes moyennes entre deux événements consécutifs de la fenêtre)
est **nouvelle** : le backend doit la calculer à partir des horodatages.
