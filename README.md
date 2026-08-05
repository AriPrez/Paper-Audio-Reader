# Paper Audio Reader

Lecteur Streamlit de papiers scientifiques PDF, conçu pour préserver le
vocabulaire biomédical tout en retirant le bruit gênant pour la synthèse vocale.

> Statut : bêta publique (`v0.1.0`). L'interface est volontairement centrée sur
> la sélection rectangulaire, avec prise en charge des documents à deux colonnes.

## Fonctionnalités

- Ordre de lecture adapté aux pages pleine largeur et à deux colonnes.
- Sélection rectangulaire précise avec coordonnées X/Y renvoyées à Python.
- Ordre de lecture sélectionnable : automatique, une colonne ou deux colonnes.
- Conservation des termes comme `tumor-specific`, `single-cell` et `IFN-γ+`.
- Filtres indépendants pour citations, URLs, légendes et équations.
- Synthèse Edge TTS découpée en segments courts, avec timeout, durée MP3
  vérifiable, téléchargement et cache par contenu.

L'extraction et le rendu du PDF sont locaux. **Edge TTS est un service en
ligne** : la génération audio nécessite une connexion Internet. Aucune clé API
n'est demandée.

## Confidentialité

- Le PDF chargé, son rendu et son extraction restent dans le processus local
  Streamlit et ne sont pas téléversés par l'application.
- Lorsque l'utilisateur demande un audio, **le texte nettoyé de la sélection est
  envoyé au service en ligne Microsoft Edge TTS** pour produire la voix.
- Il est donc déconseillé d'utiliser la synthèse vocale avec du contenu
  confidentiel, clinique identifiable ou non publiable.
- Aucun PDF de test ou article scientifique protégé n'est distribué avec ce
  dépôt. Les fichiers `*.pdf` sont exclus par `.gitignore`.

## Installation

Python 3.11 ou plus récent est recommandé.

```bash
cd paper_audio_reader
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Depuis la racine du dépôt, le lancement suivant fonctionne également :

```bash
streamlit run paper_audio_reader/app.py
```

L'interface est ensuite disponible à l'adresse indiquée par Streamlit,
habituellement `http://localhost:8501`.

## Lecture par rectangle

Choisir une page, puis tracer un rectangle sur le rendu. Seuls les blocs de
texte intersectant la zone sont nettoyés et proposés à la lecture.

Pour un paragraphe en deux colonnes, choisir **Two columns: left, then right**.
Le milieu du rectangle sert de séparation : toute la colonne gauche est lue
avant la colonne droite.

Le moteur expérimental de détection automatique des sections est conservé dans
le code et testé, mais il n'est pas exposé dans cette version de l'interface.

## Tests

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q test_reader.py
```

Les tests ordinaires sont hors ligne. Pour ajouter une régression sur un papier
local réel :

```bash
IMMUNOLOGY_TEST_PDF="/chemin/vers/paper.pdf" python -m pytest -q test_reader.py
```

La CI GitHub exécute automatiquement la suite hors ligne sous Python 3.11 et
3.13. Les tests nécessitant un article local sont ignorés quand le fichier n'est
pas disponible.

## Licence

Le code est distribué sous licence MIT. Voir [`LICENSE`](LICENSE).

## Limites

- Un PDF scanné sans couche texte doit d'abord passer par un OCR.
- L'ordre de lecture dépend de la qualité de la structure interne du PDF.
- Les très grandes sections sont découpées pour le TTS, mais le temps de
  génération dépend toujours du réseau et du service Edge.
