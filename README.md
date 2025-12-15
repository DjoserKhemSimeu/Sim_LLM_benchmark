# Sim LLM Benchmark

**Sim LLM Benchmark** is a tool designed to **evaluate the environmental impact of language model (LLM) inference** across various computing infrastructures.  
It supports both **large-scale systems** (e.g., Grid5000 GPUs) and **edge devices** (e.g., Jetson AGX Orin), allowing users to simulate and analyze performance under different concurrency levels.

---

## Table of Contents
1. [How It Works](#how-it-works)
2. [Prerequisites](#prerequisites)
3. [Usage](#usage)
4. [Example](#example)
5. [JSON Input File](#json-input-file)
6. [Compatibility](#compatibility)
7. [Outputs](#outputs)

---

## How It Works

The benchmark follows an **automated pipeline** to evaluate GPU infrastructures:

1. **Configure the environment**  
   Load GPU specifications and infrastructure parameters from a JSON configuration file.

2. **Launch Ollama servers**  
   Deploy selected LLMs (e.g., *Mistral 7B*, *GPT-OSS 20B*, *Gemma3 27B*) for inference benchmarking.

3. **Measure environmental impact**  
   - Compute the *manufacturing impact* of GPUs using  
     `measure/scripts/bar_impact.py`.  
   - Run inference benchmarks for different numbers of concurrent users using  
     `scripts/multi_gpu_bench.py`.

4. **Analyze and visualize results**  
   - Aggregate and save environmental data in CSV format.  
   - Generate plots (PNG) and save them in the `images/` directory.

---

## Prerequisites
- Clone 
```bash
git clone --branch CtG-MtC-Agentic-IssueSolver --single-branch https://github.com/DjoserKhemSimeu/Sim_LLM_benchmark.git
```
- SSH key generation :
```bash
ssh-keygen -t ed25519 -C "ton_email@example.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub

```
- Python **3.8** or higher  
- **Ollama** installed and configured  
```bash
 curl -fsSL https://ollama.com/install.sh | sh

```
- Create a python 
  ```bash
   apt install python3-venv
  python3 -m venv ~/ollama
  source ~/ollama/bin/activate



  ```
  ```
- Run by using uv :
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

```
```bash
uv run  main.py --config test.json
```
- Required Python dependencies listed in `requirements.txt`

To install the dependencies:

```bash
pip install -r requirements.txt
crewai traces disable
```

- Define the location of Ollama models 
```bash
export OLLAMA_MODELS=/tmp/ollama
```
---

## Usage

Run the benchmark using:

```bash
python main.py --config <path_to_json_file>
```

### Example

```bash
python main.py --config configs/gpu_specs.json
```

---

## JSON Input File

The benchmark requires a JSON configuration file defining the computing infrastructure.

### Example

```json
{
  "infra_name": "My_Compute_Infra",
  "description": "Compute infrastructure with multiple GPUs for various use cases",
  "PUE": 1.5,
  "Nb_users": [1],
  "gpus": {
    "0": {
      "nom": "NVIDIA A100",
      "die_area": 826,
      "tdp": 400,
      "tech_node": "7",
      "type_memoire": "HBM2e",
      "taille_memoire": 40,
      "foundry": "TSMC",
      "date_sortie": "2020",
      "fu": "Large scale"
    }
  }
}
```

---

## Description of JSON Fields

| **Field**             | **Description** |
|------------------------|----------------|
| `infra_name`           | Name of the compute infrastructure. Used in logs and results. |
| `description`          | Description of the infrastructure setup. |
| `PUE`                  | *Power Usage Effectiveness*: ratio describing data center energy efficiency (e.g., 1.5 = 1 W IT + 0.5 W overhead). |
| `Nb_users`             | List of concurrent user counts to test (e.g., `[1, 5, 10]`). |
| `gpus`                 | Dictionary describing each GPU. Each key (e.g., `"0"`) corresponds to a GPU instance. |
| `nom`                  | GPU model name (e.g., `"NVIDIA A100"`). |
| `die_area`             | GPU die area in mm² — used to estimate manufacturing impact. |
| `tdp`                  | *Thermal Design Power* (in watts) — indicates peak operating power. |
| `tech_node`            | Technology node (in nm) — smaller values typically mean better efficiency. |
| `type_memoire`         | Memory type (e.g., `"HBM2e"`). |
| `taille_memoire`       | GPU memory size in gigabytes (GB). |
| `foundry`              | Semiconductor foundry (e.g., `"TSMC"`). |
| `date_sortie`          | GPU release year — provides technological context. |
| `fu`                   | *Factor Usage*: `"Large scale"` for data center GPUs, `"Edge"` for embedded/edge devices. |

---

## JSON Configuration Schema (processed by `configs/config.py`)

`configs/config.py` lit le fichier JSON principal et en dérive les variables d'environnement
utilisées par le reste des scripts. Le fichier JSON doit être un objet contenant, au minimum,
les champs suivants (les noms sont sensibles à la casse) :

- **Champs top-level requis :**
  - `PUE` (nombre) : Power Usage Effectiveness (ex. `1.5`).
  - `MANUFACTURE_DATA` (string) : identifiant du jeu de données de fabrication (ex. `more-than-carbon`).
  - `Nb_users` (tableau de nombres) : liste des nombres d'utilisateurs simultanés à tester (ex. `[1,10,100]`).
  - `Models` (tableau de strings) : liste des modèles à tester (identifiants compatibles Ollama), ex. `["mistral:7b"]`.
  - `gpus` (objet) : dictionnaire décrivant chaque GPU ; les clés sont des indices string (`"0"`, `"1"`, ...).

- **Champs par GPU (dans chaque objet `gpus["<id>"]`) :**
  - `nom` (string) : nom du GPU.
  - `die_area` (nombre) : surface du die en mm².
  - `tdp` (nombre) : TDP en watts.
  - `density` (nombre) : densité utilisée par certains scripts (ex. mémoire).
  - `tech_node` (string) : technologie du nœud (nm).
  - `type_memoire` (string) : type de mémoire (ex. `HBM2e`).
  - `taille_memoire` (nombre) : taille mémoire en GB.
  - `foundry` (string) : fondeur (ex. `TSMC`).
  - `date_sortie` (string) : année de sortie.
  - `fu` (string) : usage factor (`Large scale` / `Edge`).

- **Champs optionnels :**
  - `Iteration` (nombre) : nombre d'itérations par test (par défaut `10` si absent).

Comportement de `configs/config.py` :
- Charge le JSON et définit des variables d'environnement préfixées `BENCH_` (ex. `BENCH_PUE`, `BENCH_USERS`, `BENCH_MODELS`).
- Pour chaque GPU, il crée des variables individuelles (`BENCH_GPU_<id>_NAME`, `BENCH_GPU_<id>_DIE_AREA`, ...).
- Génère `configs/config.toml` contenant la configuration des instances Ollama (une instance par GPU disponible) et lance
  `scripts/ollama-batch-servers.sh` pour démarrer les serveurs.

Si des champs requis manquent ou sont mal nommés, `configs/config.py` lèvera une erreur ; adaptez-le si vous
modifiez la structure du JSON.


## Compatibility

Sim LLM Benchmark supports both:

- 🖥️ **Large-scale infrastructures**  
  (e.g., NVIDIA A100, V100, H100)

- 📱 **Edge devices**  
  (e.g., Jetson AGX Orin)

---

## Outputs

All results are saved automatically in two main formats:

| **Type** | **Description** |
|-----------|----------------|
| **CSV files** | Contain raw and processed environmental metrics. |
| **PNG graphs** | Visual representations (e.g., environmental impact vs. concurrent users) stored in the `images/` directory. |

---

✅ **Sim LLM Benchmark** helps researchers and engineers understand how **GPU architecture** and **inference deployment scenarios** affect the **environmental footprint** of large language models.

---

## More than carbon updates

Les fichiers ci-dessous ont été ajoutés ou modifiés localement pour étendre l'analyse des impacts :

- `measure/scripts/bar_impact_mtc.py`
  - Calcule et exporte un résumé des impacts de fabrication estimés par GPU (en utilisant la méthode de proportionnalité basée sur un A100 de référence).
  - Produit un graphique 100% empilé par catégorie d'impact montrant la part `Main dies` vs `Heatsink` et sauvegarde :
    `manufacturing_impact_main_vs_heatsink_percent.png`.

- `scripts/water_from_mix.py`
  - Nouveau script standalone pour calculer la consommation d'eau (L et m³) associée à la production de 1 kWh à partir d'un fichier CSV de mix énergétique (`data/Energetic_mix_Fr.csv`).
  - Usage : `python scripts/water_from_mix.py --csv data/Energetic_mix_Fr.csv --kwh 1`.

- `measure/scripts/perf_show_mtc.py`
  - Variante de `perf_show.py` qui combine les impacts liés à la consommation électrique (fournis dans `data/Electricity_impacts.csv`) avec les impacts de fabrication générés par `bar_impact_mtc`.
  - Calcule les impacts pour trois facteurs : `GWP`, `ADPe`, `WU` et génère un graphique consolidé unique.
  - Le graphique consolidé positionne côte-à-côte les barres par modèle et par nombre d'utilisateurs ; chaque facteur est distingué par un **motif (hachure)** et affiché sur son axe Y propre (unités converties et affichées) :
    - GWP → affiché en `g CO2 eq` (valeurs converties depuis kg→g)
    - ADPe → affiché en `mg Sb eq` (kg→mg)
    - WU → affiché en `L` (m³→L)
  - Le script trace également des lignes verticales pointillées pour séparer les groupes d'utilisateurs et sauvegarde l'image dans `images/combined_impact/global_combined_impacts.png`.

Notes d'exécution et dépendances
- Ces nouveaux scripts utilisent `pandas`, `numpy`, `matplotlib`, et `seaborn`. Installez-les si nécessaire :

```bash
pip install pandas numpy matplotlib seaborn
```

- Pour produire les graphiques combinés, exécutez :

```bash
python measure/scripts/perf_show_mtc.py
```

Si vous souhaitez que j'ajoute :
- les valeurs numériques au-dessus des barres (labels),
- une option CLI pour choisir les facteurs ou le dossier de sortie,
- ou une version sans `pandas` pour `scripts/water_from_mix.py`, dites-le et je l'ajoute.

---

## Configuration example ajouté (`test.json`)

Un fichier de configuration `test.json` est fourni en exemple dans le dépôt. Il illustre
comment activer l'utilisation des données `more-than-carbon` pour la fabrication
et comment renseigner des paramètres supplémentaires :

- `PUE`: facteur d'efficacité énergétique (ex. 1.5)
- `MANUFACTURE_DATA`: clé indiquant la source des données de fabrication (ex. `more-than-carbon`)
- `Nb_users`: liste des nombres d'utilisateurs simultanés testés (ex. `[1,10,100]`)
- `gpus` → chaque GPU contient : `nom`, `die_area`, `tdp`, `density`, `tech_node`,
  `type_memoire`, `taille_memoire`, `foundry`, `date_sortie`, `fu`.

Extrait pertinent (`test.json`):

```json
{
  "PUE": 1.5,
  "MANUFACTURE_DATA": "more-than-carbon",
  "Nb_users": [1,10,100],
  "gpus": { "0": { "nom": "NVIDIA A100", "die_area": 826, "tdp": 400, "density": 65600000, "tech_node": "7", "taille_memoire": 40 } }
}
```

Cette configuration permet aux scripts (`bar_impact_mtc.py`, `perf_show_mtc.py`) de lire
les paramètres matérielles et d'utiliser les jeux de données locaux pour estimer les
impacts de fabrication et de fonctionnement.

---

## Sources et références

- Mix énergétique (données RTE - eco2mix) :
  https://www.rte-france.com/donnees-publications/eco2mix-donnees-temps-reel/production-electricite-par-filiere
- Jeu de données "More-than-Carbon" (impacts par composant) :
  https://github.com/sophia-falk/more-than-carbon
- Impacts de production d'électricité (ADEME - Base Empreinte) :
  https://base-empreinte.ademe.fr/donnees/jeu-donnees/05585055-9742-4fff-81ff-ad2e30e1b791/0/true/null
- Consommation en eau liée à la production d'énergie (revue scientifique) :
  https://www.sciencedirect.com/science/article/pii/S1364032119305994

