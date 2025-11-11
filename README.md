# HeRoN in Crafter: Adaptive Decision Making NPC

**Progetto di Intelligenza Artificiale 2025/26**

Questo progetto estende e testa l'architettura **HeRoN (Helper-Reviewer-NPC)** nell'environment **Crafter**, dimostrando come l'integrazione di Reinforcement Learning e Large Language Models possa migliorare il comportamento adattivo degli NPC in ambienti complessi.

## 🎯 Obiettivi del Progetto

### Obiettivo Principale
Estendere e validare l'architettura HeRoN nell'ambiente Crafter, valutando le prestazioni su tutti i 22 obiettivi di gioco.

### Obiettivi Specifici
1. **Fine-tuning del Reviewer** per i task di Crafter
2. **Modifica dell'Helper** per generare sequenze di azioni coerenti (non singole azioni)
3. **Implementazione del NPC** tramite Deep Q-Network (DQN)
4. **Valutazione delle prestazioni** di HeRoN negli obiettivi di Crafter

## �️ Architettura HeRoN

L'architettura HeRoN è composta da tre componenti integrate:

### 1. **NPC (Non-Player Character)** 
- Agente di Reinforcement Learning (DQN)
- Impara una politica ottimale attraverso l'interazione con l'ambiente
- Esegue azioni in Crafter

### 2. **Helper (LLM Zero-Shot)**
- Large Language Model utilizzato senza fine-tuning
- Genera **sequenze** di azioni coerenti (non singole azioni)
- Fornisce consigli strategici basati sullo stato corrente

### 3. **Reviewer (LLM Fine-Tuned)**
- Large Language Model specializzato per Crafter
- Valuta le suggerimenti dell'Helper (rating 1-10)
- Fornisce feedback correttivi e migliora le strategie
- Addestrato su dataset specifico di Crafter

### Flusso di Interazione
```
1. NPC osserva lo stato dell'ambiente
2. Helper propone una sequenza di azioni strategiche
3. Reviewer valuta e affina/corregge la strategia
4. Feedback del Reviewer guida la politica del NPC
5. NPC esegue le azioni e impara dall'esperienza
```

## 🎮 Crafter Environment

**Crafter** è un open-world survival game per ricerca sul Reinforcement Learning, versione semplificata di Minecraft.

### 22 Obiettivi Sbloccabili
- **Raccolta risorse**: wood, stone, coal, iron, diamond, sapling, drink
- **Crafting**: wood/stone/iron pickaxe, wood/stone/iron sword
- **Costruzione**: place_stone, place_table, place_furnace, place_plant
- **Sopravvivenza**: eat_plant, eat_cow, drink, wake_up (sleep)
- **Combattimento**: defeat_zombie, defeat_skeleton

### Sfide
- Gestione risorse limitate (salute, cibo, acqua, energia)
- Esplorazione dinamica
- Crafting progressivo
- Combattimento contro nemici

## 📁 Project Structure

```
IA2025_26_progetto/
├── src/
│   ├── environment/       # Crafter environment wrapper
│   ├── agents/           # DQN NPC implementation
│   └── llm/              # Helper and Reviewer LLMs
├── configs/              # Configuration files
├── data/                 # Training datasets
├── models/               # Saved models
├── notebooks/            # Jupyter notebooks for analysis
├── scripts/              # Training and evaluation scripts
└── tests/                # Unit tests
```

## 🚀 Setup

### 1. Create Conda Environment

#### Option A: Using Conda (Recommended)

```powershell
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate ia2025
```

#### Option B: Using Micromamba (Alternative)

```powershell
# Create environment from environment.yml
micromamba create -f environment.yml

# Activate the environment
micromamba activate ia2025
```

### 2. Install Crafter Separately

```powershell
# After activating the environment
pip install git+https://github.com/danijar/crafter.git
```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```powershell
# Copy template and edit
cp .env.example .env
# Add your OPENAI_API_KEY in the .env file
```

### 4. Validate Installation

```powershell
# Run comprehensive validation
python validate_installation.py
```

If all tests pass, your installation is complete! ✨

## 📚 Quick Start

### Pipeline Completa (Raccomandato)

```powershell
# Attiva l'environment
conda activate ia2025

# Visualizza stato pipeline
python scripts/0_pipeline_manager.py --status

# Visualizza metodologia completa
python scripts/0_pipeline_manager.py --methodology

# Esegui singolo step
python scripts/0_pipeline_manager.py --step 1

# Esegui intera pipeline (10-20 ore)
python scripts/0_pipeline_manager.py --run-all
```

### Esecuzione Manuale per Step

#### Step 1: Train Baseline NPC (2-4 ore)
```powershell
python scripts/1_train_npc_baseline.py --episodes 1000
```

#### Step 2: Optimize Helper Strategy (30-60 min)
```powershell
python scripts/2_optimize_helper_strategy.py --iterations 100
```

#### Step 3: Collect Reviewer Data (1-2 ore)
```powershell
python scripts/3_collect_reviewer_data.py --episodes 200
```

#### Step 4: Fine-tune Reviewer (2-4 ore)
```powershell
python scripts/4_finetune_reviewer.py --input data/reviewer_dataset.jsonl --output_dir models/reviewer_finetuned
```

#### Step 5: Train HeRoN (4-8 ore)
```powershell
python scripts/5_train_heron.py --episodes 2000
```

#### Step 6: Evaluate and Compare (30-60 min)
```powershell
python scripts/6_evaluate_and_compare.py --heron_model models/heron/heron_best.pth --baseline_model models/baseline/npc_baseline_best.pth
```

**Documentazione completa:** Vedi `scripts/README.md`

## 🔬 Metodologia di Implementazione

### 1. Sviluppo dell'Environment Crafter
- Studio preliminare di Crafter
- Analisi dei 22 obiettivi
- Implementazione wrapper per HeRoN

### 2. Sviluppo dell'NPC
- Implementazione Deep Q-Network (DQN)
- Training baseline SENZA HeRoN
- Stabilire prestazioni di riferimento

### 3. Modifica di Helper
- **Prompt engineering** per generare **set di azioni** sequenziali
- Transizione da singola azione → sequenze coerenti
- Ottimizzazione numero di azioni per chiamata

### 4. Fine-Tuning del Reviewer
- Generazione dataset specifico per Crafter:
  - Stati dell'environment
  - Azioni suggerite dall'Helper
  - Feedback correttivi mirati
- Addestramento Reviewer su dataset Crafter
- Validazione capacità di valutazione

### 5. Analisi del Numero di Mosse
- Test diverse configurazioni (3, 5, 7, 10 mosse)
- Analisi coerenza e validità sequenze
- Determinare configurazione ottimale
- Esempio: **due chiamate da cinque mosse ciascuna**

### 6. Addestramento Iterativo HeRoN
- Integrazione NPC + Helper + Reviewer
- Ciclo RL-LLM iterativo:
  1. NPC percepisce stato
  2. Helper propone strategia
  3. Reviewer valuta/corregge
  4. Feedback guida NPC
  5. Ottimizzazione parametri
- Training completo architettura

### 7. Valutazione delle Prestazioni
- Confronto HeRoN vs Baseline
- Metriche: reward, achievements, episode length
- Analisi tutti i 22 obiettivi
- Identificazione miglioramenti e limiti

## 📖 References

- **HeRoN Paper**: "HeRoN: A Multi-Agent RL–LLM Framework for Adaptive NPC Behavior in Interactive Environments"
- **Crafter Paper**: "Benchmarking The Spectrum of Agent Capabilities"
- **HeRoN Repository**: https://github.com/Seldre99/HeRoN
- **Crafter Repository**: https://github.com/danijar/crafter

## 📊 Risultati Attesi

### Metriche di Valutazione
- **Reward medio** per episodio
- **Achievements sbloccati** (su 22 totali)
- **Success rate** per ogni achievement
- **Lunghezza episodi**
- **Confronto HeRoN vs Baseline**

### Risultati da Dimostrare
1. ✅ **Abilità del NPC** nello svolgere i task di Crafter
2. ✅ **Efficacia del Reviewer** nel fornire feedback mirati
3. ✅ **Ottimizzazione prestazioni** attraverso addestramento iterativo
4. ✅ **Miglioramenti rispetto al baseline** (o analisi dei limiti)
5. ✅ **Analisi delle sfide** e soluzioni implementate

### Output Generati
- **Modelli addestrati**: NPC baseline, NPC HeRoN, Reviewer fine-tuned
- **Dataset**: Interazioni Helper-Reviewer
- **Logs**: Training metrics, achievement progression
- **Visualizzazioni**: Grafici comparativi, heatmap achievements
- **Report**: Analisi prestazioni dettagliata

## 🛠️ Struttura Tecnica

### Codice Sorgente (`src/`)
```
src/
├── __init__.py
├── utils.py                    # Utility functions
├── agents/
│   ├── __init__.py
│   └── dqn_agent.py           # Deep Q-Network implementation
├── environment/
│   ├── __init__.py
│   └── crafter_env.py         # Crafter wrapper
└── llm/
    ├── __init__.py
    ├── helper.py              # Helper LLM (zero-shot)
    └── reviewer.py            # Reviewer LLM (fine-tuned)
```

### Configurazioni (`configs/`)
- `dqn_config.yaml`: Hyperparametri DQN agent
- `helper_config.yaml`: Configurazione Helper LLM + prompts
- `reviewer_config.yaml`: Configurazione Reviewer + fine-tuning
- `heron_config.yaml`: Integrazione architettura HeRoN

### Scripts (`scripts/`)
Pipeline sequenziale documentata in `scripts/README.md`

## 🔍 Sfide e Soluzioni

### Sfida 1: Generazione Sequenze Coerenti (Helper)
**Problema**: Helper originale genera singole azioni
**Soluzione**: Prompt engineering per sequenze di N azioni coerenti

### Sfida 2: Fine-tuning Reviewer per Crafter
**Problema**: Reviewer non specializzato per Crafter
**Soluzione**: Dataset custom con stati Crafter + feedback mirati

### Sfida 3: Integrazione RL-LLM
**Problema**: Bilanciare politica NPC e suggerimenti LLM
**Soluzione**: Threshold acceptance + reward shaping basato su Reviewer

### Sfida 4: Numero Ottimale di Mosse
**Problema**: Determinare quante azioni suggerire
**Soluzione**: Analisi empirica (script 2) → configurazione ottimale

### Sfida 5: Efficienza Computazionale
**Problema**: Fine-tuning LLM richiede molte risorse
**Soluzione**: LoRA + 4-bit quantization per efficienza

## � Riferimenti

### Articoli
- **HeRoN**: "HeRoN: A Multi-Agent RL–LLM Framework for Adaptive NPC Behavior in Interactive Environments"
- **Crafter**: "Benchmarking The Spectrum of Agent Capabilities" (Hafner, 2021)

### Repositories
- **HeRoN Original**: https://github.com/Seldre99/HeRoN
- **Crafter**: https://github.com/danijar/crafter

### Tecnologie Utilizzate
- **RL**: Deep Q-Network (DQN), Experience Replay, Target Networks
- **LLM**: OpenAI GPT-4, Llama 2, LoRA fine-tuning
- **Framework**: PyTorch, Transformers, Gymnasium
- **Environment**: Crafter (danijar/crafter)

## 👥 Progetto Accademico

**Corso**: Intelligenza Artificiale 2025/26
**Istituzione**: [Nome Università]
**Tipo**: Progetto di ricerca e implementazione

## 📝 Licenza

Progetto per scopi educativi e di ricerca.
