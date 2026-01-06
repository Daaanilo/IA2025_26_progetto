<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Transformers">
  <img src="https://img.shields.io/badge/Status-Research-blueviolet?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/License-Academic-green?style=for-the-badge" alt="License">
</p>

<h1 align="center">🧠 HeRoN in Crafter</h1>

<p align="center">
  <strong>Adaptive Decision Making NPC using the HeRoN Architecture</strong><br>
  <em>Extending Multi-Agent RL-LLM Framework to an Open-World Survival Environment</em>
</p>

<p align="center">
  <a href="#-project-overview-english">🇬🇧 English</a> •
  <a href="#-descrizione-del-progetto-italiano">🇮🇹 Italiano</a> •
  <a href="#-usage--utilizzo">🚀 Usage</a> •
  <a href="#-project-structure--struttura-del-progetto">📁 Structure</a>
</p>

---

## 👨‍💻 Authors / Autori

<table align="center">
  <tr>
    <td align="center">
      <strong>Danilo Gisolfi</strong><br>
      <sub>Università degli Studi di Salerno</sub>
    </td>
    <td align="center">
      <strong>Vincenzo Maiellaro</strong><br>
      <sub>Università degli Studi di Salerno</sub>
    </td>
  </tr>
</table>

---

## 📖 Project Overview (English)

This project extends and evaluates the **HeRoN (Helper–Reviewer–NPC)** architecture in the **Crafter** environment — an open-world survival game widely used in Reinforcement Learning research and inspired by Minecraft.

### 🎮 What is Crafter?

Crafter is a benchmark environment where an agent must:

| Challenge | Description |
|-----------|-------------|
| 🍖 **Survival** | Gather food and water to maintain health and stamina |
| 🔨 **Crafting** | Create tools from collected resources |
| 🏠 **Shelter** | Build protection from environmental threats |
| 👾 **Combat** | Defend against hostile creatures |
| 🎯 **Achievements** | Complete **22 hierarchical objectives** |

### 🏗️ HeRoN Architecture

The HeRoN framework combines Reinforcement Learning with Large Language Models:

```
┌─────────────────────────────────────────────────────────────────┐
│                        HeRoN Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    Suggestions    ┌─────────────┐            │
│   │   HELPER    │ ────────────────► │  REVIEWER   │            │
│   │  (LLM ZS)   │                   │ (LLM Fine-  │            │
│   │             │ ◄──────────────── │   tuned)    │            │
│   └──────┬──────┘     Feedback      └──────┬──────┘            │
│          │                                 │                    │
│          │      Corrected Actions          │                    │
│          └──────────────┬──────────────────┘                    │
│                         ▼                                       │
│                  ┌─────────────┐                                │
│                  │     NPC     │                                │
│                  │  (DQN Agent)│                                │
│                  └──────┬──────┘                                │
│                         │                                       │
│                         ▼                                       │
│              ┌──────────────────┐                               │
│              │ Crafter Env 🎮   │                               │
│              └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

| Component | Role | Technology |
|-----------|------|------------|
| **🤖 NPC** | RL agent that learns to play Crafter | Deep Q-Network (DQN) |
| **💡 Helper** | Zero-shot LLM suggesting action sequences | Prompt-engineered LLM |
| **✅ Reviewer** | Fine-tuned LLM validating/correcting suggestions | RL Fine-tuned LLM |

### 🎯 Project Goals

- [x] Study and integrate the Crafter environment
- [x] Implement NPC agent using Deep Q-Network
- [x] Adapt Helper via prompt engineering for action sequences
- [x] Fine-tune Reviewer with custom dataset
- [x] Optimize action-sequence length
- [x] Evaluate performance across 22 achievements

---

## 📖 Descrizione del Progetto (Italiano)

Questo progetto estende e valuta l'architettura **HeRoN (Helper–Reviewer–NPC)** nell'environment **Crafter** — un gioco di sopravvivenza open-world ampiamente utilizzato nella ricerca sul Reinforcement Learning e ispirato a Minecraft.

### 🎮 Cos'è Crafter?

Crafter è un ambiente benchmark dove un agente deve:

| Sfida | Descrizione |
|-------|-------------|
| 🍖 **Sopravvivenza** | Raccogliere cibo e acqua per mantenere salute e resistenza |
| 🔨 **Crafting** | Creare strumenti dalle risorse raccolte |
| 🏠 **Riparo** | Costruire protezione dalle minacce ambientali |
| 👾 **Combattimento** | Difendersi dalle creature ostili |
| 🎯 **Achievement** | Completare **22 obiettivi gerarchici** |

### 🏗️ Architettura HeRoN

Il framework HeRoN combina Reinforcement Learning con Large Language Models:

| Componente | Ruolo | Tecnologia |
|------------|-------|------------|
| **🤖 NPC** | Agente RL che impara a giocare a Crafter | Deep Q-Network (DQN) |
| **💡 Helper** | LLM zero-shot che suggerisce sequenze di azioni | LLM con prompt engineering |
| **✅ Reviewer** | LLM fine-tuned che valida/corregge i suggerimenti | LLM con RL Fine-tuning |

### 🎯 Obiettivi del Progetto

- [x] Studio e integrazione dell'environment Crafter
- [x] Implementazione dell'agente NPC tramite Deep Q-Network
- [x] Adattamento dell'Helper tramite prompt engineering
- [x] Fine-tuning del Reviewer con dataset personalizzato
- [x] Ottimizzazione della lunghezza delle sequenze di azioni
- [x] Valutazione delle prestazioni sui 22 achievement

---

## 🚀 Usage / Utilizzo

### Training the Base DQN Agent

```python
# Run base DQN training / Esegui training DQN base
python training/DQN_training.py
```

### Training with Helper Integration

```python
# Run DQN with Helper / Esegui DQN con Helper
python training/dqn_helper_training.py
```

### Full HeRoN Pipeline

```python
# Run complete HeRoN training / Esegui training HeRoN completo
python training/heron_training.py
```

### Testing the Environment

```python
# Test Crafter environment / Testa l'environment Crafter
python test_crafter_env.py
```

---

## 📁 Project Structure / Struttura del Progetto

```
IA2025_26_progetto/
├── 📂 classes/                    # Core modules / Moduli principali
│   ├── agent.py                  # DQN Agent implementation
│   ├── crafter_environment.py    # Crafter wrapper
│   ├── crafter_helper.py         # Helper LLM integration
│   └── instructor_agent.py       # Instructor agent base
│
├── 📂 training/                   # Training scripts / Script di training
│   ├── DQN_training.py           # Base DQN training
│   ├── dqn_helper_training.py    # DQN + Helper training
│   ├── heron_training.py         # Full HeRoN training
│   └── reward_shaper.py          # Custom reward shaping
│
├── 📂 evaluation/                 # Evaluation tools / Strumenti di valutazione
│
├── 📂 reviewer_fine_tuning/       # Reviewer training / Training del Reviewer
│
├── 📂 documentazione/             # Documentation & LaTeX / Documentazione
│   ├── main.pdf                  # Full report
│   └── immagini/                 # Plots and diagrams
│
├── 📄 requirements.txt            # Python dependencies
├── 📄 generate_plots_from_data.py # Visualization utilities
└── 📄 README.md                   # This file
```

---

## 📊 Results / Risultati

### Training Metrics Dashboard

The project includes comprehensive visualization tools for monitoring training progress:

- **Achievement Curves**: Track completion rates for all 22 objectives
- **Reward Distribution**: Analyze reward patterns during training
- **Helper Dependency**: Measure LLM integration effectiveness
- **Efficiency Scatter**: Compare episode length vs. achievements

---

## 📚 Resources & References / Risorse e Riferimenti

- 🔗 [Crafter Environment](https://github.com/danijar/crafter)
- 🔗 [HeRoN Official Codebase](https://github.com/Seldre99/HeRoN)
- 🔗 [Hugging Face Transformers](https://github.com/huggingface/transformers)

---

## 📜 License / Licenza

This project is developed for academic purposes as part of the Artificial Intelligence course at **Università degli Studi di Salerno** (2025/2026).

Questo progetto è sviluppato per scopi accademici come parte del corso di Intelligenza Artificiale presso l'**Università degli Studi di Salerno** (2025/2026).

---

<p align="center">
  <strong>Made with ❤️ for AI Research</strong><br>
  <sub>Università degli Studi di Salerno • Corso di Intelligenza Artificiale 2025/2026</sub>
</p>

<p align="center">
  <a href="#-heron-in-crafter">⬆️ Back to Top / Torna su</a>
</p>
