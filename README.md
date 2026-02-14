# Sim LLM Benchmark

**Sim LLM Benchmark** is a tool designed to evaluate the **environmental impact of Large Language Model (LLM) inference** across different computing infrastructures.

It supports:

- Inference performance  
- Energy consumption  
- Carbon footprint (GWP)  
- Abiotic resource depletion (ADPe)  
- Water consumption  

under various concurrency levels and hardware configurations.

---

# Overview

Sim LLM Benchmark relies on a fully containerized pipeline to ensure reproducibility and portability.

## 1. Environment Initialization

The `run.sh` script:

- Initializes the SSH agent  
- Verifies that an SSH key is loaded  
- Securely forwards Git credentials into the container  

---

## 2. Container Build

A Docker image is automatically built from the provided `Dockerfile`, based on:

nvidia/cuda:12.8.0-devel-ubuntu24.04

---

if there is compatibility issue the GPU under test ; change the docker file base image with : nvidia/cuda:12.4.1-devel-ubuntu22.04

## 3. Automated Execution

Once the image is built, the container automatically runs:

python3 main.py --config test.json

---

## 4. Environmental Impact Assessment

### Manufacturing Impact

Estimated using:

- GPU die area  
- Technology node  
- Foundry information  
- Thermal Design Power (TDP)  

Data source:

More-than-Carbon dataset  
https://github.com/sophia-falk/more-than-carbon

---

### Operational Impact

Measured during live inference via Ollama:

- Energy consumption  
- Carbon emissions  
- Water usage  

Calculation sources:

French electricity mix (RTE eco2mix)  
https://www.rte-france.com/donnees-publications/eco2mix-donnees-temps-reel/production-electricite-par-filiere

ADEME Base Empreinte database  
https://base-empreinte.ademe.fr/donnees/jeu-donnees/05585055-9742-4fff-81ff-ad2e30e1b791/0/true/null

Scientific publication on water consumption in energy production  
# Sim LLM Benchmark

**Sim LLM Benchmark** is a tool designed to evaluate the **environmental impact of Large Language Model (LLM) inference** across different computing infrastructures.

It supports:

- **High-performance systems** (e.g., NVIDIA GH200, A100)
- **Edge platforms** (e.g., Jetson AGX Orin)

The benchmark enables simulation and analysis of:

- Inference performance
- Energy consumption
- Carbon footprint (GWP)
- Abiotic resource depletion (ADPe)
- Water consumption

under various concurrency levels and hardware configurations.

---

# Overview

Sim LLM Benchmark relies on a fully containerized pipeline to ensure reproducibility and portability.

## 1. Environment Initialization

The `run.sh` script:

- Initializes the SSH agent
- Verifies that an SSH key is loaded
- Securely forwards Git credentials into the container

---

## 2. Container Build

A Docker image is automatically built from the provided `Dockerfile`, based on:

```text
nvidia/cuda:12.8.0-devel-ubuntu24.04
```

---

If there is a compatibility issue with the GPU under test, change the Dockerfile base image to:

```text
nvidia/cuda:12.4.1-devel-ubuntu22.04
```

## 3. Automated Execution

Once the image is built, the container automatically runs:

```bash
python3 main.py --config test.json
```

---

## 4. Environmental Impact Assessment

### Manufacturing Impact

Estimated using:

- GPU die area
- Technology node
- Foundry information
- Thermal Design Power (TDP)

Data source:

More-than-Carbon dataset
https://github.com/sophia-falk/more-than-carbon

---

### Operational Impact

Measured during live inference via Ollama:

- Energy consumption
- Carbon emissions
- Water usage

Calculation sources:

French electricity mix (RTE eco2mix)
https://www.rte-france.com/donnees-publications/eco2mix-donnees-temps-reel/production-electricite-par-filiere

ADEME Base Empreinte database
https://base-empreinte.ademe.fr/donnees/jeu-donnees/05585055-9742-4fff-81ff-ad2e30e1b791/0/true/null

Scientific publication on water consumption in energy production
https://www.sciencedirect.com/science/article/pii/S1364032119305994

---

## 5. Data Persistence

Docker volumes ensure that all results are synchronized to the host machine:

- save_data/ → CSV reports (raw and processed data)
- images/ → Generated environmental footprint visualizations
- /tmp/ollama_host_storage → Cached Ollama models

---

# Prerequisites

- NVIDIA Container Toolkit installed and configured
- At least one accessible NVIDIA GPU (--gpus all)
- An active SSH key loaded in your agent:

```bash
ssh-add ~/.ssh/id_rsa
```

---

# Quick Start

## 1. Clone the Repository

```bash
git clone https://github.com/DjoserKhemSimeu/Sim_LLM_benchmark.git
cd Sim_LLM_benchmark
```

## 2. Run the Benchmark

```bash
chmod +x run.sh
./run.sh
```

If no SSH key is detected, the script will prompt you to add one.

---

# test.json Configuration

The `test.json` file defines the infrastructure, workload, and environmental modeling parameters used during the benchmark.

## Global Parameters

- **PUE** (float)
  Power Usage Effectiveness of the infrastructure.
  Scales IT energy to account for cooling and facility overhead.
  Example: `1.5`

- **Models** (list of strings)
  LLM models evaluated via Ollama.
  Example: `["gemma3:12b", "llama3:8b"]`

- **MANUFACTURE_DATA** (string)
  Manufacturing impact dataset key (e.g., `"more-than-carbon"`).
  Used to estimate embodied carbon and material depletion of hardware.

- **GitHub_SSH** (string)
  SSH URL of the repository accessed during the benchmark.
  Example: `"git@github.com:user/repo.git"`

---

## GPU Configuration

- **gpus** (list of objects)
  Defines hardware characteristics for operational and manufacturing impact modeling.

Fields per GPU:

- `name` → GPU model name (reporting only)
- `die_area` → Silicon die area (used for embodied impact estimation)
- `tech_node` → Manufacturing node (e.g., 7, 5)
- `tdp` → Thermal Design Power in watts (power envelope reference)
- `foundry` → Semiconductor manufacturer (e.g., TSMC, Samsung)
- `taille_memoire` → VRAM capacity


---
## Example Minimal Configuration

```json
{
  "PUE": 1.5,
  "Models": ["gemma3:12b"],
  "MANUFACTURE_DATA": "more-than-carbon",
  "GitHub_SSH": "git@github.com:user/repository.git",
  "gpus": [
    {
      "name": "NVIDIA A100",
      "die_area": 826,
      "tech_node": 7,
      "tdp": 400,
      "foundry": "TSMC",
      "taille_memoire": 80,
      "quantity": 1
    }
  ]
}
```

# Maintenance

## Docker Cleanup

```bash
docker system prune -a
docker image prune -a
```

---

## Changing Docker Storage Location

If your default Docker partition is too small, modify:

```text
/etc/docker/daemon.json
```

Example:

```json
{
  "data-root": "/tmp/docker-root"
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

---

# Data Sources & References

Electricity mix data (RTE eco2mix)
https://www.rte-france.com/donnees-publications/eco2mix-donnees-temps-reel/production-electricite-par-filiere

More-than-Carbon dataset
https://github.com/sophia-falk/more-than-carbon

Electricity production impacts (ADEME – Base Empreinte)
https://base-empreinte.ademe.fr/donnees/jeu-donnees/05585055-9742-4fff-81ff-ad2e30e1b791/0/true/null

Water consumption linked to energy production
https://www.sciencedirect.com/science/article/pii/S1364032119305994

---

# License

Specify your license here (e.g., MIT, Apache 2.0).

---

# Citation

If you use this benchmark in academic work, please cite the repository or associated publication (if available).
