# Baseline et protocole d'évaluation validés — Module ML SecureCloud AI

*Jeu de données synthétique (graine 42). Résultats = test, utilisé une seule fois.*

## Protocole
- **Données** : fenêtres de 5 min, 7 variables ; train 4 960 normaux ; validation 1 367 (≈13 % d'attaques) ; test 1 357 (≈13 % d'attaques). 76 doublons supprimés dans et entre ensembles, vérification automatique d'absence de recouvrement.
- **Entraînement** sur événements normaux uniquement ; étiquettes jamais utilisées pour entraîner.
- **Validation** : choix des hyperparamètres (PR-AUC) et du seuil (F1 maximal). **Test** : une seule évaluation finale.
- **Métriques** : précision, rappel, F1, PR-AUC, ROC-AUC, matrice de confusion, alertes pour 10 000 événements, rappel par famille, temps d'exécution.

## Résultats (test)
| Détecteur | P | R | F1 | PR-AUC | Alertes/10 000 |
| --- | --- | --- | --- | --- | --- |
| **Baseline Mahalanobis** | 0,805 | 0,800 | 0,802 | 0,894 | 1 319 |
| **Isolation Forest (retenu)** | 0,890 | 0,894 | 0,892 | 0,946 | 1 334 |
| One-Class SVM | 0,891 | 0,867 | 0,879 | 0,941 | 1 290 |
| LOF (novelty) | 0,757 | 0,639 | 0,693 | 0,800 | 1 120 |
| IF + règle métier (max_severity ≥ 5,2, choisi sur validation) | 0,857 | 0,900 | 0,878 | — | 1 393 |

Isolation Forest dépasse la baseline de +9 points de F1 et +5 points de PR-AUC ; garde-fou F1 ≥ 0,80 et rappel ≥ 0,85 : **atteint**. Point faible : famille `sg_tampering` (rappel 0,68).

## Contamination de l'entraînement (Isolation Forest, F1 test, 5 tirages)
0 % : 0,883 — 1 % : 0,883 ± 0,006 — 5 % : 0,847 ± 0,004 (baseline : 0,802 / 0,808 / 0,774). Dégradation modérée à 5 %.

## Corrections par rapport à la version initiale
1. Baseline statistique ajoutée ; deux détecteurs de comparaison (LOF, One-Class SVM).
2. Doublons : 4 941 → 76 supprimés ; plus aucune ligne commune entre ensembles (auparavant 835/1 150 lignes de test identiques à l'entraînement).
3. Jeu de données rendu non trivial (look-alikes bénins, attaques qui chevauchent le normal) : la baseline ne fait plus 100 %.
4. Règle `max_severity ≥ 7` circulaire supprimée du protocole principal ; la règle métier est désormais un seuil choisi sur la validation, présenté séparément.
5. Ajout de PR-AUC, courbes PR, alertes/10 000, rappel par famille, temps, expérience de contamination, interface d'inférence.

## Limites
Données synthétiques ; nouvelle variable `mean_event_gap_s` à calculer côté backend ; prévalence d'attaques (13 %) bien supérieure à la réalité, donc précision réelle plus faible.
