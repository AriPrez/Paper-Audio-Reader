# Paper Audio Reader

Lecteur Streamlit de papiers scientifiques PDF, conçu pour préserver le
vocabulaire biomédical tout en retirant le bruit gênant pour la synthèse vocale.

> Statut : bêta publique (`v0.1.0`). L'interface est volontairement centrée sur
> la sélection rectangulaire, avec prise en charge des documents à deux colonnes.

## Fonctionnalités

- Écran d'entrée réduit à une seule action : la barre latérale et ses réglages
  n'apparaissent qu'une fois un document ouvert.
- Ordre de lecture adapté aux pages pleine largeur et à deux colonnes.
- Sélection rectangulaire précise avec coordonnées X/Y renvoyées à Python :
  poignées de redimensionnement, déplacement, zoom, et aperçu en direct des
  paragraphes capturés.
- Ordre de lecture sélectionnable : automatique, une colonne ou deux colonnes.
- Conservation des termes comme `tumor-specific`, `single-cell` et `IFN-γ+`.
- Filtres indépendants pour citations entre crochets, entre parenthèses et
  numériques en exposant, ainsi que pour les URLs, légendes et équations.
- Protection des exposants scientifiques tels que `10⁶`, `m²`, `Ca²⁺` et
  `x²` lors du retrait des appels bibliographiques en exposant.
- Transcription éditable avant synthèse : le texte extrait peut être corrigé
  dans le champ de droite sans redessiner le rectangle.
- Prononciation biomédicale : la notation est développée avant la synthèse,
  sans modifier le texte affiché (voir plus bas).
- Synthèse Edge TTS découpée en segments courts, générés quatre à la fois et
  réassemblés dans l'ordre, avec timeout, reprise d'un segment bloqué,
  avancement, durée MP3 vérifiable, téléchargement et cache par contenu.
  Une section de 15 minutes d'audio se génère en une dizaine de secondes.
- Thème clair et sombre défini dans [`.streamlit/config.toml`](.streamlit/config.toml),
  suivi également par le composant de sélection.

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

Depuis le répertoire parent, le lancement suivant fonctionne également, mais
Streamlit ne lit `.streamlit/config.toml` que dans le répertoire courant : le
thème n'est alors pas appliqué.

```bash
streamlit run paper_audio_reader/app.py
```

L'interface est ensuite disponible à l'adresse indiquée par Streamlit,
habituellement `http://localhost:8501`.

## Lecture par rectangle

Choisir une page, puis tracer un rectangle sur le rendu. Seuls les blocs de
texte intersectant la zone sont nettoyés et proposés à la lecture. Les
paragraphes qui seront réellement capturés s'éclairent pendant le tracé : le
composant applique exactement la règle de `select_blocks_in_region` (centre du
bloc dans la zone, ou 10 % de sa surface au moins), donc l'aperçu ne ment pas.

| Geste | Effet |
| --- | --- |
| Glisser | Tracer un rectangle |
| Cliquer un paragraphe | Le sélectionner entièrement |
| Cliquer une zone vide | Effacer la sélection |
| Glisser une poignée | Ajuster un bord ou un coin |
| Glisser l'intérieur | Déplacer la sélection |
| `Ctrl` + molette, ou `+` / `−` | Zoomer |
| Flèches | Déplacer d'un pas (`Alt` : pas fin) |
| `Maj` + flèches | Redimensionner |
| `Échap` | Effacer |

Le texte extrait reste **modifiable** avant la synthèse : corriger une
extraction bancale dans le champ de droite est plus rapide que de retracer le
rectangle.

Pour un paragraphe en deux colonnes, choisir **Two columns: left, then right**.
Le milieu du rectangle sert de séparation : toute la colonne gauche est lue
avant la colonne droite.

Le moteur expérimental de détection automatique des sections est conservé dans
le code et testé, mais il n'est pas exposé dans cette version de l'interface.

## Prononciation biomédicale

Un moteur de synthèse lit la notation littéralement : `MSI-H` devient
« em-esse-i-tiret-ache », `CD8+` « cé-dé-huit-plus », `10⁶` « dix six ».
Aucune voix ne corrige cela. Avant la synthèse — et **uniquement** pour la
synthèse, le transcript affiché reste intact — la notation est développée :

| Écrit | Lu |
| --- | --- |
| `dMMR` / `pMMR` | mismatch repair deficient / proficient |
| `MSI-H` / `MSI-L` / `MSS` | M S I high / M S I low / M S S |
| `CD8+` / `CD8-` | C D 8 positive / C D 8 negative |
| `IFN-γ+` / `TNF-α` | interferon gamma positive / tumour necrosis factor alpha |
| `PD-L1` / `CTLA-4` | P D L 1 / C T L A 4 |
| `OS` / `PFS` / `HR` / `CI` | overall survival / progression free survival / hazard ratio / confidence interval |
| `p < 0.05` / `n = 42` | p less than 0.05 / n equals 42 |
| `10⁶` / `m²` / `Ca²⁺` | 10 to the power of 6 / m squared / Ca 2 plus |
| `et al.` / `Fig. 3` | and colleagues / Figure 3 |

Le bouton « What the voice will read » affiche le texte réellement envoyé au
moteur. La case **Biomedical pronunciation** désactive le tout.

Les règles vivent dans `_SPEECH_SUBSTITUTIONS` (`parser.py`) et s'appliquent
dans l'ordre : la plus spécifique de chaque famille d'abord (`MSI-H` avant
`MSI`, `PD-L1` avant `PD-1`). `CD8-positive` reste un adjectif composé, seul
`CD8-` suivi d'un espace devient « negative ». Les cas qui exigeraient de
savoir si un sigle est un gène, un élément ou un mot anglais sont volontairement
absents : `OR` (odds ratio) et `BRCA` sont laissés tels quels plutôt que
massacrés.

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
