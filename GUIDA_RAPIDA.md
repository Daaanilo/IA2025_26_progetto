# HeRoN in Crafter - Guida Rapida

**Architettura HeRoN (Helper-Reviewer-NPC) applicata all'ambiente Crafter**

Progetto di corso - Università degli Studi di Salerno (2025/2026)  
Autori: Danilo Gisolfi, Vincenzo Maiellaro

---

## 📋 Descrizione

Sistema multi-agente che combina Reinforcement Learning (DQN) con Large Language Models per creare NPC adattivi nell'ambiente Crafter (gioco di sopravvivenza open-world ispirato a Minecraft).

**Componenti:**
- **NPC**: Agente DQN (Deep Q-Network) con Prioritized Replay
- **Helper**: LLM zero-shot (Qwen 3-4B) che suggerisce sequenze di azioni
- **Reviewer**: Modello T5 fine-tuned con PPO per validare e correggere i suggerimenti

---

## ⚙️ Installazione

```bash
# Clona il repository
git clone https://github.com/Daaanilo/IA2025_26_progetto.git
cd IA2025_26_progetto

# Installa le dipendenze
pip install -r requirements.txt

# Setup LM Studio (per Helper LLM)
# 1. Scarica e avvia LM Studio
# 2. Carica il modello: qwen/qwen3-4b-2507
# 3. Avvia il server sulla porta 1234
```

---

## 🚀 Utilizzo

### Training

```bash
# DQN baseline (senza LLM)
python training/DQN_training.py --episodes 300

# DQN + Helper (senza Reviewer)
python training/dqn_helper_training.py --episodes 300

# HeRoN completo - LLM attivo primi 100 step
python training/HERON_initial.py --episodes 300

# HeRoN completo - Probabilità LLM crescente (K=0.01)
python training/HERON_final.py --episodes 300

# HeRoN completo - Probabilità LLM 50% random
python training/HERON_random.py --episodes 300
```

### Testing

```bash
# Testa un modello allenato
python testing/test_dqn_crafter.py --model training/heron_final_output/models/heron_final_final --episodes 50

# Altri modelli disponibili:
# - training/dqn_output/models/dqn_agent_final
# - training/dqn_helper_output/models/dqn_helper_final
# - training/heron_initial_output/models/heron_initial_final
# - training/heron_random_output/models/heron_random_final
```

### Fine-tuning del Reviewer

```bash
# 1. Genera il dataset di training
python "dataset Reviewer/crafter_dataset_generation.py"

# 2. Fine-tuning supervised (T5)
python reviewer_fine_tuning/reviewer_fine_tuning.py

# 3. Reinforcement learning con PPO
python reviewer_fine_tuning/ppo_training.py
```

### Generazione Grafici

```bash
# Grafici di training (learning curves, achievement, helper calls)
python grafici/training_plots.py

# Grafici di testing (boxplot, radar chart, matrici)
python grafici/testing_plots.py
```

---

## 📁 Struttura Progetto

```
IA2025_26_progetto/
├── classes/                  # Classi principali
│   ├── agent.py              # DQN Agent (Double DQN + Prioritized Replay)
│   ├── crafter_environment.py # Wrapper Crafter con estrazione stato 43D
│   ├── crafter_helper.py     # LLM Helper per sequenze azioni
│   └── instructor_agent.py   # Reviewer T5
├── training/                 # Script di training
│   ├── DQN_training.py       # Baseline DQN
│   ├── dqn_helper_training.py # DQN + Helper
│   ├── HERON_initial.py      # HeRoN (LLM primi 100 step)
│   ├── HERON_final.py        # HeRoN (prob. crescente)
│   ├── HERON_random.py       # HeRoN (prob. 50%)
│   └── reward_shaper.py      # Custom reward shaping
├── testing/                  # Script di testing
│   └── test_dqn_crafter.py   # Valutazione modelli
├── grafici/                  # Visualizzazioni
│   ├── training_plots.py     # Grafici training
│   └── testing_plots.py      # Grafici testing
├── reviewer_fine_tuning/     # Fine-tuning Reviewer
│   ├── reviewer_fine_tuning.py # Supervised learning T5
│   └── ppo_training.py       # PPO reinforcement learning
└── dataset Reviewer/         # Generazione dataset
    └── crafter_dataset_generation.py
```

---

## 🔑 Dipendenze Principali

- **PyTorch** - Deep Q-Network
- **Transformers** (Hugging Face) - LLM (T5, Qwen)
- **LM Studio** - Server LLM locale
- **Crafter** (>=1.8.3) - Ambiente di gioco
- **Gymnasium** - API RL
- **NumPy, Matplotlib, Seaborn** - Analisi e visualizzazione

---

## 📊 Risultati

I modelli allenati e i risultati (metriche, grafici, checkpoint) vengono salvati in:
- `training/[variant]_output/` - Modelli, log JSONL, grafici
- `testing/[variant]_test_results/` - Risultati test, statistiche

---

## 📄 Licenza

Progetto accademico - Università degli Studi di Salerno (2025/2026)
