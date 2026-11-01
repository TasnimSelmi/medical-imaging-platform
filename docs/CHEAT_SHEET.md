# Cheat Sheet — Soutenance Plateforme d'Imagerie Médicale

## Chiffres clés

| Élément | Valeur |
|---------|--------|
| Filtres de prétraitement | 9 (gaussien, médian, moyenneur, bilatéral, NLM, CLAHE, hist. eq., Z-score, N4) |
| Opérations morphologiques | 5 (érosion, dilatation, ouverture, fermeture, gradient) |
| Détecteurs de contours | 3 (Sobel, Laplace, Canny) |
| Méthodes de segmentation classique | 7 (Otsu, adaptatif, watershed, Chan-Vese, K-means, Mean Shift, region growing) |
| Architecture U-Net | 5 blocs encodeur, filtres 16→256, Dropout 0.1→0.3, ~1.9M paramètres |
| Perte d'entraînement | 0.5 × BCE + 0.5 × Dice loss |
| MC Dropout passes | N = 30 (compromis précision/temps) |
| Métriques d'évaluation | 5 (Dice, IoU, Pearson, MSE, NMI) |
| Stack backend | FastAPI + TensorFlow/Keras + OpenCV + scikit-image |
| Stack frontend | React 18 + TypeScript + Vite + TailwindCSS + Zustand + Recharts |
| Tests | 25 (8 filtres + 12 métriques + 5 incertitude) |

---

## Top 15 réponses d'entretien

**1. Pourquoi Keras ?** Le TP4 l'impose. MC Dropout = un seul `training=True`. Pas de raison technique de migrer.

**2. Dice modeste ?** L'objectif est pédagogique : comprendre chaque brique. nnU-Net serait une boîte noire. Dataset de 100 images trop petit pour un modèle complexe.

**3. MC Dropout vs ensemble ?** MC Dropout = 1 modèle, N masques aléatoires. Ensemble = N modèles séparés. MC est N× moins cher mais sous-estime l'incertitude.

**4. Pourquoi 30 passes ?** Le Dice et la confiance saturent vers N=20-30 (gain marginal en $1/\sqrt{N}$). N=100 → 30s/image, prohibitif en interactif.

**5. Incertitude non corrélée à l'erreur ?** Modèle mal calibré. Causes : Dropout trop faible, dataset trop homogène. Solutions : augmenter le taux, deep ensemble, temperature scaling.

**6. Overfitting sur 100 images ?** 3 remèdes : Dropout (régularisation), augmentation (diversité artificielle), EarlyStopping (arrêt avant surapprentissage).

**7. Votre watershed vs naïf ?** Marqueurs construits par distance transform + seuillage + composantes connexes. Un marqueur par objet au lieu de centaines de minima.

**8. CLAHE vs égalisation standard ?** Standard = globale, amplifie les reflets. CLAHE = locale + clip limit, rehausse le contraste sans exploser les zones brillantes.

**9. N4 seulement sur IRM ?** Le biais de champ est un artefact physique des bobines RF. L'endoscopie (caméra optique) n'a pas ce biais.

**10. Autre dataset (thorax) ?** Fenêtrage Hounsfield au lieu de CLAHE, possible 3D, pas d'elastic transform (thorax rigide), dataset plus grand → modèle plus complexe.

**11. Image d'un autre hôpital ?** Domain shift probable. Solutions : Z-score/CLAHE pour normaliser, fine-tuning sur quelques images locales, domain adaptation.

**12. Dice loss non-convexe ?** Le quotient crée des minima locaux. La BCE (convexe) stabilise le gradient → on combine les deux.

**13. Faux positif vs faux négatif ?** FP = biopsie inutile. FN = cancer manqué. FN plus grave → on veut un rappel élevé + MC Dropout signale les zones à vérifier.

**14. Combien de paramètres ?** ~1.9M. C'est 13× moins que ResNet-50, adapté à 100 images avec Dropout + augmentation.

**15. Segmentation sémantique vs instance ?** Notre U-Net = sémantique binaire (fond vs objet). Instance distinguerait tumeur 1 vs tumeur 2 → nécessiterait Mask R-CNN.

---

## Pitch 1 minute

> « C'est une plateforme web d'analyse d'images médicales qui couvre toute la chaîne : du prétraitement — 9 filtres incluant CLAHE pour l'endoscopie et N4 pour l'IRM — à la segmentation par 7 méthodes classiques et un U-Net entraîné avec Dice+BCE. L'extension recherche utilise le Monte Carlo Dropout pour quantifier l'incertitude du modèle pixel par pixel : en gardant le Dropout actif à l'inférence sur 30 passes, on obtient une carte d'incertitude et un score de confiance. Le notebook montre que l'incertitude est corrélée à l'erreur, et que rejeter les 20% de cas les moins confiants améliore significativement le Dice — un scénario de triage clinique. La principale limite est la taille du dataset, mais le projet démontre que savoir *quand* un modèle se trompe est aussi important que la prédiction elle-même. »

---

## Checklist jour J

- [ ] Lancer `docker-compose up` et vérifier que l'app tourne
- [ ] Préparer 2-3 images de test (une endoscopie, une IRM, une « difficile »)
- [ ] Vérifier que le modèle .hdf5 est chargé (pas d'erreur 503)
- [ ] Tester la démo complète : upload → CLAHE → Otsu → U-Net → MC Dropout → évaluation
- [ ] Ouvrir le notebook `uncertainty_study.ipynb` avec les résultats pré-calculés
- [ ] Connaître les 3 formules clés : Dice, entropie de Bernoulli, confiance MC Dropout
- [ ] Savoir dessiner le U-Net (5 blocs, skip connections, bottleneck)
- [ ] Préparer la réponse honnête sur les limites (10 images de val, pas de test set, pas de validation clinique)
- [ ] Relire le glossaire (30+ termes)
- [ ] Respirer. Tu connais ton projet.
