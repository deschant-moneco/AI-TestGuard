# 4. Rapport de performance et de sécurité opérationnelle

*Activité 8. Pour reproduire : `python scripts/performance_report.py`. Tests : `tests/test_performance.py`, `tests/test_security.py`. Chiffres bruts : `reports/performance.json`.*

## 4.1 Protocole de mesure

| Élément | Valeur |
|---|---|
| Matériel | Apple M3 Pro (arm64), 12 cœurs logiques, 19,3 Go de RAM, macOS 26.5 |
| Logiciel | Python 3.12.8, scikit-learn 1.9.1, numpy 2.5.5, pandas 3.1.1 |
| Modèle | Régression logistique retenue, rechargée depuis le disque (contrôle SHA-256) |
| Horloge | `time.perf_counter` (monotone, haute résolution) |
| Préchauffage | 50 appels, exclus des mesures |
| Latence d'une prédiction | 2 000 appels successifs sur des lignes différentes du test (DataFrame d'une ligne) |
| Latence d'une requête API | 1 000 requêtes `POST /predict` (1 transaction) sur un serveur HTTP local (`http.server`, même machine) |
| Débit | Lots de 1, 100, 1 000, 10 000 et 56 746 lignes ; 5 répétitions, médiane retenue |
| Mémoire | RSS du processus (`psutil`) avant et après chargement du modèle et pendant le scoring ; pic d'allocation Python (`tracemalloc`) |
| Exécution | Machine non dédiée (session utilisateur active) : les valeurs maximales sont bruitées, on s'appuie sur P95 et P99 |

**Distinction des latences mesurées :**

1. **Modèle brut** : `pipeline.predict_proba` seul.
2. **Prédiction applicative** : validation du contrat (`validate_data`) puis modèle. C'est le coût d'un appel de bibliothèque.
3. **Requête API** : en plus, connexion TCP, analyse HTTP, désérialisation JSON, validation stricte des champs et types, sérialisation de la réponse. Cette mesure n'inclut ni le réseau réel, ni TLS, ni l'authentification.

## 4.2 Résultats

| Mesure | P50 | P95 | P99 | Max | Seuil d'acceptation | Verdict |
|---|---|---|---|---|---|---|
| Modèle brut (1 transaction) | 0,97 ms | 1,02 ms | 1,08 ms | 4,8 ms | – | – |
| Prédiction avec validation (1 transaction) | 1,45 ms | **1,55 ms** | 1,75 ms | 5,0 ms | P95 ≤ 10 ms | ✅ |
| Requête API HTTP locale (1 transaction) | 2,85 ms | **3,21 ms** | 3,47 ms | 4,5 ms | P95 ≤ 50 ms | ✅ |

Environ 0,5 ms (un tiers du temps applicatif) part dans la validation du contrat, et environ 1,4 ms supplémentaire dans la couche HTTP/JSON.

| Taille du lot | Durée médiane | Débit |
|---|---|---|
| 1 | 3,3 ms | ≈ 300 / s |
| 100 | 2,2 ms | ≈ 45 600 / s |
| 1 000 | 2,4 ms | ≈ 422 000 / s |
| 10 000 | 4,8 ms | ≈ 2,1 M / s |
| 56 746 (jeu de test) | 24,8 ms | **≈ 2,3 M / s** (seuil ≥ 20 000 / s ✅) |

| Ressource | Mesure | Seuil | Verdict |
|---|---|---|---|
| Taille du modèle sauvegardé | **4,2 Ko** (+ métadonnées JSON) | ≤ 5 Mo | ✅ |
| Mémoire au chargement du modèle | Variation de RSS < 1 Mo (non mesurable) | – | ✅ |
| Pic d'allocation pour scorer 56 746 lignes | 55,9 Mo (`tracemalloc`) | ≤ 200 Mo | ✅ |
| RSS du processus pendant la mesure | ≈ 665 Mo, dû au jeu de 284 807 lignes chargé pour l'expérience, pas au modèle | – | – |

### Justification des seuils

* **10 ms au P95 pour une prédiction, 50 ms pour l'API.** L'autorisation d'un paiement par carte doit répondre en quelques centaines de millisecondes de bout en bout, et le scoring de fraude n'en est qu'une étape. On lui réserve au plus environ 10 % du budget.
* **20 000 prédictions par seconde en batch.** Rescorer une journée de 10 millions de transactions doit tenir en moins de 10 minutes.
* **5 Mo et 200 Mo.** Déploiement dans un conteneur léger, avec plusieurs réplicas par nœud.

### Anomalies et remarques

* Le lot d'une seule ligne en mode débit (3,3 ms) est plus lent que la boucle de latence (1,45 ms). Avec 5 répétitions, la médiane est sensible au réchauffement des caches. Pour une seule transaction, la mesure de référence est celle sur 2 000 appels.
* En CI (`test_performance.py`), les bornes sont **5 fois plus larges** que les seuils d'acceptation, car les runners GitHub sont partagés et bruités. Le test sert de garde-fou contre une régression grossière, la mesure de référence reste le script.
* Aucun test de charge concurrent n'a été réalisé (Locust ou JMeter) : il n'y a pas d'API déployée. Ce test sera **nécessaire avant le pilote** (voir le rapport final).

## 4.3 Sécurité opérationnelle

| Contrôle | Mise en œuvre | Test(s) | Résultat |
|---|---|---|---|
| Champs manquants | `parse_payload` compare les clés reçues aux 30 variables attendues | `test_missing_field_is_rejected` | HTTP 400 « champs manquants ['V17'] » ✅ |
| Champs supplémentaires | Liste blanche stricte. Un champ `card_number` est refusé, pas ignoré | `test_extra_field_is_rejected`, `test_malformed_payloads_are_rejected` (clé `admin`) | HTTP 400 ✅ |
| Types incorrects | Seuls `int` et `float` sont acceptés. `bool` est refusé explicitement (sous-classe de `int` en Python), de même que les chaînes, `null`, les listes et les objets | `test_incorrect_types_are_rejected` (5 cas) | HTTP 400 ✅ |
| NaN, Infinity, montant négatif | Refusés en mode strict. JSON `NaN` et `Infinity` sont détectés après l'analyse | `test_nan_and_infinity_rejected_in_api`, `test_negative_amount_rejected_in_api` | HTTP 400 ✅ |
| Taille des entrées | Corps ≤ 16 Ko (HTTP 413), au plus 100 transactions par requête | `test_oversized_payload_and_too_many_transactions` | ✅ |
| JSON malformé | Message générique, pas de trace de pile renvoyée | `test_malformed_payloads_are_rejected` (7 cas) | ✅ |
| Fuite dans les journaux | Seul un identifiant pseudonyme (SHA-256 tronqué) et la décision sont journalisés, jamais le montant ni V1–V28. Les journaux d'accès de `http.server` sont désactivés | `test_logs_do_not_leak_sensitive_values` | Ni le montant (9 876,54) ni V1 n'apparaissent ✅ |
| Secrets dans le dépôt | Recherche de clés AWS, jetons GitHub et Kaggle, clés privées, `password=` / `token=` | `test_no_secrets_in_repository` | 0 occurrence ✅ |
| Fichiers sensibles | `.env`, `kaggle.json`, `*.pem`, le jeu complet et `models/` sont dans `.gitignore` | `test_sensitive_files_are_gitignored` | ✅ |
| Fichier de modèle | Extension `.joblib` imposée, taille ≤ 50 Mo, **empreinte SHA-256 vérifiée avant `joblib.load`** (pickle peut exécuter du code) | `test_load_model_rejects_tampered_file`, `…bad_extension…` | Fichier altéré refusé ✅ |
| Fichiers de données | Extension `.csv` imposée, erreurs d'analyse transformées en `DataValidationError` | `test_load_dataset_errors_are_explicit` | ✅ |
| CI | `permissions: contents: read` dans le workflow, aucun secret utilisé | – | ✅ |

**Limites.**

* Le prototype n'a ni authentification, ni TLS, ni limitation de débit : ces fonctions relèvent d'une passerelle d'API et sont **obligatoires** avant toute exposition.
* L'empreinte SHA-256 détecte une altération accidentelle ou naïve. Elle ne protège pas d'un attaquant qui peut réécrire à la fois le modèle et ses métadonnées : il faudrait une signature (par exemple Sigstore) et un stockage en lecture seule.
* La recherche de secrets repose sur des expressions régulières. En complément, la protection *push protection* / *secret scanning* de GitHub est recommandée.
