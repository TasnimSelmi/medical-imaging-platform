# Plateforme d'analyse d'images médicales

Application web pour l'analyse et le traitement d'images médicales par approches conventionnelles et apprentissage profond.

**Mini-projet de fin de module** — SUP'COM, Option SIA, encadré par Mme Wafa Rekik, 2026-2027.

## Ce que fait l'application

L'utilisateur importe une image médicale dans le navigateur. Il lui applique ensuite une chaîne de traitements, étape par étape, et compare les segmentations obtenues par les différentes approches.

| Exigence du sujet | Implémentation |
|---|---|
| Import d'images médicales | PNG, JPG, BMP, TIFF, DICOM (`.dcm`), NIfTI (`.nii`, `.nii.gz`, coupe centrale) |
| Prétraitement : filtrage, contraste, bruit | Filtres gaussien, médian, moyenneur, bilatéral, Non-Local Means ; CLAHE, égalisation d'histogramme, normalisation Z-score, correction de biais N4 (IRM) |
| Segmentation par seuillage | Otsu, seuillage adaptatif gaussien |
| Segmentation par contours actifs | Chan-Vese (contour actif par régions), contour actif géodésique morphologique (contour attiré par les bords) |
| Ligne de partage des eaux | Watershed avec marqueurs issus de la transformée de distance |
| Autres méthodes conventionnelles | K-means, Mean Shift, croissance de régions |
| Deep learning pré-entraîné | U-Net dont l'encodeur est un **MobileNetV2 pré-entraîné sur ImageNet**, affiné sur Kvasir-SEG (polypes en endoscopie) |
| Visualisation comparative | Superposition masque/image, curseur avant/après, zoom, histogramme ; page **Comparaison** qui exécute toutes les méthodes sur la même image et affiche les masques, le Dice, l'IoU et la durée de chacune |

L'application propose aussi des opérations morphologiques (érosion, dilatation, ouverture, fermeture, gradient), des détecteurs de contours (Sobel, Laplacien, Canny), des métriques (Dice, IoU, Pearson, EQM, NMI) et une estimation de l'incertitude par Monte Carlo Dropout (carte d'écart-type, carte d'entropie, score de confiance). Le recalage par flot optique (Farnebäck) est disponible uniquement via l'API.

## Architecture

```
┌──────────────────────┐   /api (proxy)   ┌─────────────────────────────┐
│ Frontend             │ ───────────────► │ Backend                     │
│ React 18, TypeScript │                  │ FastAPI, Python 3.11        │
│ Vite, Tailwind       │                  │ OpenCV, scikit-image,       │
│ http://localhost:5173│                  │ SimpleITK, TensorFlow/Keras │
└──────────────────────┘                  │ http://localhost:8000       │
                                          └─────────────────────────────┘
```

Le backend charge au démarrage les modèles présents dans `backend/app/models_registry/`. Les images importées sont stockées dans `/tmp/mediplat`, chemin modifiable avec la variable `MEDIPLAT_UPLOAD_DIR`.

## Installation après clonage

### 1. Prérequis

- Python 3.11
- Node.js 20 et npm
- Optionnel, pour entraîner rapidement : un GPU NVIDIA avec un pilote récent

### 2. Backend : environnement Python

```bash
cd medical-imaging-platform/backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pour entraîner sur GPU, installez en plus les bibliothèques CUDA de TensorFlow :

```bash
pip install "tensorflow[and-cuda]==2.16.2"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"   # doit lister un GPU
```

### 3. Dataset Kvasir-SEG

Kvasir-SEG contient 1000 images d'endoscopie avec leurs masques de polypes. Le plus simple est de le télécharger depuis son miroir Hugging Face :

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('berkaytrhn/kvasir-seg', repo_type='dataset', local_dir='$HOME/datasets/kvasir-hf')"
```

Le dataset se trouve alors dans `~/datasets/kvasir-hf/Kvasir-SEG/`, avec un dossier `images/` et un dossier `masks/`. Il est aussi disponible sur le site officiel : https://datasets.simula.no/kvasir-seg/

### 4. Modèle de deep learning

Si `backend/app/models_registry/unet_endoscopy_pretrained.keras` est déjà présent, passez à l'étape 5. Sinon, entraînez-le (depuis `backend/`, venv activé) :

```bash
python -m training.train_endoscopy_pretrained --data_dir ~/datasets/kvasir-hf/Kvasir-SEG
```

- **Phase 1** (10 epochs) : l'encodeur MobileNetV2 reste gelé avec ses poids ImageNet, seul le décodeur apprend (learning rate 1e-3).
- **Phase 2** (jusqu'à 60 epochs) : tout le réseau est affiné (learning rate 1e-4).
- Les poids ImageNet sont téléchargés automatiquement par Keras.
- Le meilleur modèle, choisi sur le Dice de validation, est écrit dans `app/models_registry/unet_endoscopy_pretrained.keras`.
- Durée : quelques minutes sur GPU, nettement plus sur CPU.

Optionnel : un U-Net de référence entraîné from scratch, avec l'architecture du TP4, sert de point de comparaison. Le backend ne l'utilise que si le modèle pré-entraîné est absent.

```bash
python -m training.train_endoscopy --data_dir ~/datasets/kvasir-hf/Kvasir-SEG --epochs 100
# → app/models_registry/unet_endoscopy.h5
```

Les deux scripts utilisent :
- la perte 0.5 × BCE + 0.5 × Dice ;
- l'augmentation albumentations (flips, rotation ±15°, luminosité/contraste, transformation élastique) ;
- la même séparation entraînement/validation 80/20 (`random_state=42`).

### 5. Lancer le backend

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --port 8000
```

La documentation interactive de l'API (Swagger) est sur http://localhost:8000/docs. Si aucun modèle n'est trouvé, un avertissement s'affiche au démarrage et les routes de deep learning répondent `503`. Les autres fonctionnalités restent utilisables.

### 6. Lancer le frontend

Dans un second terminal :

```bash
cd frontend
npm install
npm run dev
```

L'application est sur http://localhost:5173. Vite redirige les appels `/api` vers le backend sur le port 8000.

## Utilisation

1. Sur la page d'accueil, choisir **Endoscopie**.
2. Importer une image, par exemple une image de `~/datasets/kvasir-hf/Kvasir-SEG/images/`.
3. Appliquer des prétraitements, une segmentation classique ou le **U-Net** depuis le panneau latéral. Chaque étape s'ajoute au pipeline.
4. Optionnel : **Charger masque GT** avec le masque correspondant de `masks/`, puis **Calculer métriques**.
5. Cliquer sur **Comparer les méthodes**. La page Comparaison exécute toutes les méthodes classiques et le U-Net sur l'image d'origine, puis affiche les masques côte à côte, un graphique du Dice par méthode et un tableau Dice/IoU/durée.

Raccourcis dans l'espace de travail : `o` superposition, `m` masque seul, `r` image d'origine, `u` incertitude, `1`–`9` retour à l'étape N du pipeline.

> Pour une évaluation honnête, utilisez une image de **validation** : les autres ont servi à l'entraînement. Pour afficher la liste de ces images (depuis `backend/`, venv activé) :
> ```bash
> python -c "from pathlib import Path; from sklearn.model_selection import train_test_split; p=sorted(Path('$HOME/datasets/kvasir-hf/Kvasir-SEG/images').glob('*.jpg')); print('\n'.join(x.name for x in train_test_split(p, test_size=0.2, random_state=42)[1]))"
> ```

## Tests

### 1. Tests unitaires du backend

```bash
cd backend
source .venv/bin/activate
pytest tests/ -v
```

25 tests au total :
- `test_metrics.py` : Dice, IoU, Pearson, EQM, NMI ;
- `test_filters.py` : formes de sortie et plages de valeurs des filtres ;
- `test_uncertainty.py` : MC Dropout sur un petit modèle factice (formes des sorties, écart-type positif, confiance entre 0 et 1, masque binaire).

Ces tests n'ont besoin ni du dataset ni du modèle entraîné.

### 2. Vérification du frontend

```bash
cd frontend
npm run build    # vérifie les types TypeScript (tsc) puis construit le bundle de production
```

### 3. Test de bout en bout de l'API

Prérequis : le backend est lancé (étape 5) et le modèle est entraîné. Depuis un autre terminal :

```bash
D=~/datasets/kvasir-hf/Kvasir-SEG
NAME=cju5ekty5ckzf07550c9u3ckk.jpg     # image de validation
A=http://localhost:8000/api

IMG=$(curl -s -F "file=@$D/images/$NAME" $A/upload | python3 -c "import json,sys;print(json.load(sys.stdin)['image_id'])")
GT=$(curl -s -F "file=@$D/masks/$NAME"  $A/upload | python3 -c "import json,sys;print(json.load(sys.stdin)['image_id'])")

curl -s -X POST $A/compare -H 'content-type: application/json' \
  -d "{\"image_id\":\"$IMG\",\"ground_truth_id\":\"$GT\",\"methods\":[\"otsu\",\"adaptive_threshold\",\"watershed\",\"chan_vese\",\"active_contour\",\"kmeans\",\"meanshift\",\"region_growing\",\"unet\"]}" \
  | python3 -m json.tool
```

La réponse contient, pour chaque méthode, l'identifiant du masque produit, sa durée et son Dice/IoU par rapport au masque de vérité terrain. Si `unet` est absent de la réponse, le modèle n'est pas chargé : relisez les avertissements au démarrage du backend.

### 4. Test manuel dans l'interface

Suivez la section [Utilisation](#utilisation) avec la même image de validation et son masque. Sur la page Comparaison, le U-Net doit apparaître (cadre et barre de couleur différente) avec un Dice nettement supérieur à celui des méthodes classiques.

## Résultats

Dice de validation sur Kvasir-SEG (200 images de validation) :

| Modèle | Epochs | Dice (val) |
|---|---|---|
| U-Net TP4 d'origine | 50 | ~0.47 |
| U-Net from scratch (BCE+Dice, augmentation) | 100 | 0.812 |
| **U-Net encodeur MobileNetV2 pré-entraîné ImageNet** | 60 | **0.894** |

Comparaison sur trois images de validation (Dice par rapport à la vérité terrain) :

| Image | Otsu | Watershed | Chan-Vese | Contour actif géodésique | K-means | U-Net pré-entraîné |
|---|---|---|---|---|---|---|
| `cju5ekty5ckzf07550c9u3ckk` | 0.385 | 0.483 | 0.393 | 0.557 | 0.385 | **0.980** |
| `cju7da88w2eod0755wejzynvt` | 0.114 | 0.133 | 0.122 | 0.144 | 0.114 | **0.930** |
| `cju7dda8w2br20818zhsuz8s7` | 0.782 | 0.803 | 0.783 | 0.925 | 0.780 | **0.983** |

Les méthodes conventionnelles ne fonctionnent que lorsque le polype se distingue nettement du fond par son intensité. Le U-Net reste robuste dans tous les cas.

## Lancement avec Docker (optionnel)

```bash
docker-compose up --build
```

Le frontend est servi par nginx sur http://localhost:5173 et le backend sur le port 8000. L'image du backend copie le contenu de `backend/`, donc le modèle doit être entraîné **avant** le build pour être inclus. L'image utilise TensorFlow sur CPU, ce qui suffit pour l'inférence.

## API

| Méthode | Route | Rôle |
|---|---|---|
| POST | `/api/upload` | Importer une image, renvoie `image_id` |
| GET | `/api/image/{id}`, `/api/image/{id}/thumbnail`, `/api/download/{id}` | Récupérer ou télécharger une image ou un résultat |
| POST | `/api/preprocess/{method}` | Prétraitement (`gaussian`, `median`, `mean`, `bilateral`, `nlm`, `clahe`, `histogram_eq`, `zscore`, `n4_bias`) |
| GET | `/api/preprocess/histogram/{id}` | Histogramme |
| POST | `/api/morphology/{method}` | Morphologie (`erosion`, `dilation`, `opening`, `closing`, `gradient`) |
| POST | `/api/edges/{method}` | Contours (`sobel`, `laplace`, `canny`) |
| POST | `/api/segment/classical/{method}` | `otsu`, `adaptive_threshold`, `watershed`, `chan_vese`, `active_contour`, `kmeans`, `meanshift`, `region_growing` |
| POST | `/api/segment/deep/unet` | Segmentation U-Net |
| POST | `/api/segment/deep/unet_mc` | U-Net + incertitude MC Dropout |
| POST | `/api/evaluate` | Métriques entre un masque prédit et la vérité terrain |
| POST | `/api/compare` | Exécuter plusieurs méthodes (dont `unet`) et les comparer |
| POST | `/api/registration/optical_flow` | Recalage par flot optique de Farnebäck |

## Structure du projet

```
medical-imaging-platform/
├── backend/
│   ├── app/
│   │   ├── main.py                 # Application FastAPI, chargement des modèles
│   │   ├── config.py               # Dossiers, extensions autorisées
│   │   ├── routers/                # Routes REST (une par module)
│   │   ├── services/               # Traitements : filters, classical_seg, deep_seg,
│   │   │                           # uncertainty, metrics, morphology, edges, registration, io
│   │   └── models_registry/        # Modèles entraînés (.keras / .h5)
│   ├── training/
│   │   ├── unet_pretrained.py      # U-Net à encodeur MobileNetV2 ImageNet
│   │   ├── train_endoscopy_pretrained.py
│   │   ├── unet.py                 # U-Net from scratch (architecture TP4)
│   │   ├── train_endoscopy.py
│   │   ├── train_brain_mri.py      # IRM cérébrale (dataset LGG Kaggle)
│   │   ├── losses.py               # BCE + Dice
│   │   └── augmentation.py         # Albumentations
│   ├── tests/                      # Tests pytest
│   └── notebooks/                  # demo_api.ipynb, uncertainty_study.ipynb
├── frontend/src/
│   ├── pages/                      # Home, Workspace, Compare, Uncertainty
│   ├── components/                 # Visualiseurs, panneau latéral, graphiques
│   ├── api/client.ts               # Client HTTP
│   └── store/useSession.ts         # État (Zustand)
├── docs/                           # Compte rendu, fiche récapitulative
└── docker-compose.yml
```

## Limites connues

- **IRM cérébrale** : la modalité IRM propose le prétraitement (dont la correction N4), la segmentation conventionnelle et la comparaison, mais pas le deep learning. Les boutons U-Net y sont désactivés avec la mention « Modèle disponible pour l'endoscopie uniquement », et la page Comparaison n'y inclut pas le U-Net. Le script `train_brain_mri.py` existe, mais aucun modèle IRM n'est fourni, car le dataset LGG est hébergé sur Kaggle.
- **Page Comparaison** : les méthodes y tournent avec leurs paramètres par défaut. La croissance de régions part du centre de l'image.
- **Images à canal unique** (DICOM, NIfTI) : elles sont dupliquées sur 3 canaux pour le U-Net pré-entraîné, qui a été entraîné sur des images couleur d'endoscopie.
