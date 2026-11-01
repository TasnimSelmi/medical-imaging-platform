# Rapport de projet : Plateforme web d'analyse d'images médicales

## Table des matières

1. Introduction et contexte médical
2. État de l'art
3. Architecture de la plateforme
4. Modules de prétraitement
5. Segmentation conventionnelle
6. Segmentation par U-Net
7. Quantification d'incertitude
8. Interface utilisateur et choix de design
9. Limites et perspectives
10. Conclusion

---

## 1. Introduction et contexte médical

L'imagerie médicale constitue un pilier fondamental du diagnostic clinique moderne. L'endoscopie des cordes vocales et l'IRM cérébrale sont deux modalités essentielles qui nécessitent des outils d'analyse automatisés pour assister les cliniciens dans leur pratique quotidienne.

Ce projet s'inscrit dans le cadre du module d'Intelligence Artificielle de SUP'COM (option SIA, 2026-2027, encadré par Mme Wafa Rekik). Il vise à concevoir et développer une plateforme web interactive permettant d'importer des images médicales et d'appliquer une chaîne complète de traitement, depuis le prétraitement jusqu'à la segmentation par apprentissage profond, en passant par les méthodes conventionnelles.

**Objectifs principaux :**
- Intégrer les méthodes étudiées lors des 5 TPs du module
- Proposer une interface de visualisation comparative
- Étendre le projet par une brique de recherche : quantification d'incertitude par Monte Carlo Dropout

## 2. État de l'art

### 2.1 Prétraitement d'images médicales

Le prétraitement est une étape cruciale pour améliorer la qualité des images avant analyse. Les techniques principales incluent :

- **CLAHE** (Contrast Limited Adaptive Histogram Equalization) : particulièrement adaptée aux images d'endoscopie où les intensités restent sombres et les reflets spéculaires perturbent l'analyse.
- **Filtrage bilatéral** : préserve les contours tout en réduisant le bruit, crucial pour l'endoscopie.
- **Correction N4 bias field** : corrige les inhomogénéités d'intensité propres à l'IRM.

### 2.2 Segmentation conventionnelle

- **Seuillage d'Otsu** : méthode automatique basée sur la maximisation de la variance inter-classes.
- **Watershed (Ligne de partage des eaux)** : segmentation par marqueurs, adaptée aux objets contigus.
- **Chan-Vese** : contours actifs sans bord, robuste aux images avec faible gradient.
- **K-means et Mean Shift** : méthodes de clustering appliquées à la segmentation.

### 2.3 U-Net pour la segmentation médicale

L'architecture U-Net (Ronneberger et al., 2015) est devenue le standard pour la segmentation d'images médicales. Sa structure en encodeur-décodeur avec skip connections permet de capturer à la fois le contexte global et les détails fins.

Notre implémentation suit l'architecture du TP4 : encodeur 5 blocs (16→32→64→128→256), Dropout progressif (0.1→0.3), décodeur symétrique avec Conv2DTranspose.

### 2.4 Monte Carlo Dropout

Gal et Ghahramani (2016) ont montré que le Dropout, habituellement désactivé à l'inférence, peut être maintenu actif pour approximer l'inférence bayésienne. En effectuant N passes stochastiques, on obtient une distribution de prédictions dont la variance capture l'incertitude épistémique du modèle.

## 3. Architecture de la plateforme

### 3.1 Stack technique

- **Backend** : Python 3.11, FastAPI, Uvicorn
- **Deep learning** : TensorFlow/Keras
- **Vision classique** : OpenCV, scikit-image, scikit-learn, SimpleITK
- **Frontend** : React 18, TypeScript, Vite, TailwindCSS
- **State management** : Zustand
- **Containerisation** : Docker + docker-compose

### 3.2 Architecture REST

L'API expose des endpoints organisés par catégorie fonctionnelle :
- `/api/upload` : chargement d'images (PNG, JPG, DICOM, NIfTI)
- `/api/preprocess/*` : filtres de prétraitement
- `/api/morphology/*` : opérations morphologiques
- `/api/edges/*` : détection de contours
- `/api/segment/classical/*` : segmentation conventionnelle
- `/api/segment/deep/*` : segmentation par U-Net et MC Dropout
- `/api/registration/*` : recalage d'images
- `/api/evaluate` et `/api/compare` : évaluation quantitative

### 3.3 Gestion des images

Les images sont stockées temporairement sur le serveur avec un identifiant UUID unique. Un mécanisme de nettoyage automatique supprime les images après 1 heure d'inactivité (TTL).

## 4. Modules de prétraitement (lien TP2)

### 4.1 Filtres spatiaux

| Filtre | Implémentation | Paramètre principal |
|--------|---------------|-------------------|
| Moyenneur | `skimage.filters.rank.mean` | Taille du noyau |
| Médian | `cv2.medianBlur` | Taille du noyau |
| Gaussien | `cv2.GaussianBlur` | Sigma |
| Bilatéral | `cv2.bilateralFilter` | d, sigma_color, sigma_space |
| Non-local means | `cv2.fastNlMeansDenoising` | h, fenêtre template/recherche |

### 4.2 Amélioration du contraste

- **CLAHE** : `cv2.createCLAHE` avec paramètres clip_limit et tile_grid_size
- **Égalisation d'histogramme** : `cv2.equalizeHist`
- **Normalisation Z-score** : centrage-réduction avec rescaling [0, 255]
- **Correction N4** : `SimpleITK.N4BiasFieldCorrectionImageFilter` pour l'IRM

### 4.3 Opérations morphologiques

Érosion, dilatation, ouverture, fermeture et gradient morphologique (différence dilatation - érosion). Éléments structurants elliptiques.

### 4.4 Détection de contours

Sobel, Laplacien et Canny avec paramètres ajustables.

## 5. Segmentation conventionnelle (lien TP2 + TP3)

### 5.1 Méthodes implémentées

| Méthode | Référence TP | Paramètres |
|---------|-------------|------------|
| Otsu | TP2 | Automatique |
| Seuillage adaptatif | TP2 | block_size, C |
| Watershed | TP2 | Marqueurs automatiques |
| Chan-Vese | TP3 | mu, lambda1, lambda2, max_iter |
| K-means | TP3 | k (nombre de clusters) |
| Mean Shift | TP3 | spatial_radius, color_radius |
| Croissance de régions | — | seed_x, seed_y, threshold |

### 5.2 Résultats quantitatifs

Les résultats comparatifs seront présentés via l'interface de comparaison multi-méthodes, avec métriques Dice et IoU lorsqu'un masque ground truth est fourni.

### 5.3 Mesure du temps de calcul

Chaque segmentation retourne sa durée d'exécution en millisecondes, conformément au TP3.

## 6. Segmentation par U-Net (lien TP4)

### 6.1 Architecture

Architecture fidèle au TP4 :
- Encodeur : Conv2D(3,3) ReLU he_normal, Dropout progressif (0.1→0.3), MaxPooling(2,2)
- Bottleneck : 256 filtres, Dropout 0.3
- Décodeur : Conv2DTranspose, skip connections, Dropout symétrique

### 6.2 Améliorations par rapport au TP4

| Aspect | TP4 | Version améliorée |
|--------|-----|-------------------|
| Perte | BCE seule | 0.5 × BCE + 0.5 × Dice loss |
| Augmentation | Aucune | Albumentations (flip, rotation, brightness, elastic) |
| Arrêt | Fixe (10/50 epochs) | EarlyStopping patience=15, ModelCheckpoint |
| Dice (endoscopie) | 0.47 (50 epochs) | Objectif > 0.75 |

### 6.3 Datasets

- **Endoscopie** : dataset du TP4 (100 paires image/masque 256×256 niveaux de gris)
- **IRM cérébrale** : Kaggle LGG MRI Segmentation

## 7. Quantification d'incertitude (extension recherche)

### 7.1 Méthodologie

Extension au-delà du cahier des charges : Monte Carlo Dropout pour la quantification d'incertitude épistémique.

**Principe** : En maintenant les couches Dropout actives à l'inférence (`training=True`) et en effectuant N passes stochastiques, on obtient une distribution de prédictions dont on extrait :
- La **prédiction moyenne** (masque de segmentation)
- La **carte d'écart-type** pixel par pixel (incertitude épistémique)
- La **carte d'entropie prédictive** : H = -(p log p + (1-p) log(1-p))
- Un **score de confiance global** : 1 - E[H] / log(2)

### 7.2 Résultats

Voir `notebooks/uncertainty_study.ipynb` pour les expériences détaillées :
- Sensibilité au nombre de passes N
- Corrélation incertitude ↔ erreur de segmentation
- Rejet sélectif des prédictions peu fiables
- Galerie de cas illustratifs

### 7.3 Discussion

L'incertitude permet d'identifier les cas où le modèle est peu fiable, ce qui est particulièrement important dans un contexte clinique. Les zones de haute incertitude correspondent souvent aux petites régions non détectées (problème identifié en TP4) et aux reflets spéculaires en endoscopie.

## 8. Interface utilisateur et choix de design

### 8.1 Principes de design

Interface sobre et professionnelle, proche d'un outil clinique :
- Palette restreinte : fond #fafafa, surfaces blanches, accent #1e40af
- Typographie Inter (UI) + JetBrains Mono (valeurs numériques)
- Grille 8px, transitions 150ms
- Pas d'ombres, pas de glassmorphism, pas de gradients

### 8.2 Pages

- **Home** : sélection de modalité (endoscopie / IRM)
- **Workspace** : vue 3 colonnes (sidebar paramètres, viewer central avec onglets, historique pipeline)
- **Compare** : grille comparative multi-méthodes avec graphique Recharts
- **Uncertainty** : visualisation dédiée MC Dropout avec carte σ et score de confiance

### 8.3 Fonctionnalités UX

- Zoom et pan sur les images (react-zoom-pan-pinch)
- Overlay avec slider d'opacité
- Raccourcis clavier (o: overlay, m: masque, r: original, u: incertitude)
- Historique du pipeline avec retour à n'importe quelle étape
- Toasts discrets pour les erreurs (sonner)

## 9. Limites et perspectives

### 9.1 Limites

- **Taille du dataset** : seulement 100 images d'endoscopie et pas de test set indépendant
- **Performance CPU** : MC Dropout avec N=30 passes peut être lent sur CPU
- **Généralisation** : modèle entraîné sur une seule modalité endoscopique
- **Incertitude** : seule l'incertitude épistémique est capturée (pas l'aléatorique)

### 9.2 Perspectives

- Combiner MC Dropout avec la perte Dice+BCE pour améliorer à la fois la performance et la calibration
- Ajouter des modèles pré-entraînés (DeepLabV3, SegFormer) pour comparaison
- Intégrer le support DICOM complet avec métadonnées patient
- Ajouter un module de recalage non-rigide
- Exploiter l'incertitude pour l'apprentissage actif

## 10. Conclusion

Ce projet a permis de consolider les apprentissages des 5 TPs du module en une plateforme intégrée, tout en l'enrichissant d'une brique de recherche crédible sur la quantification d'incertitude. L'approche MC Dropout démontre qu'il est possible, sans ré-entraînement, d'exploiter les couches Dropout existantes pour obtenir des estimations d'incertitude utiles cliniquement.

---

## Références

- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.
- Zuiderveld, K. (1994). Contrast Limited Adaptive Histogram Equalization.
- Tustison, N. J., et al. (2010). N4ITK: Improved N3 Bias Correction.
