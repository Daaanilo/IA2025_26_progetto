# 🧠 Adaptive Decision Making NPC in Crafter  
### Extending the HeRoN Architecture to an Open-World RL Environment

## 👨‍💻 Autori / Authors
- **Danilo Gisolfi**  
- **Vincenzo Maiellaro**

---

## 🇮🇹 Descrizione del Progetto

Questo progetto ha l’obiettivo di estendere e testare l’architettura **HeRoN (Helper–Reviewer–NPC)** nell’environment **Crafter**, un open-world survival game utilizzato nella ricerca sul Reinforcement Learning e ispirato a Minecraft.

In Crafter il giocatore deve:  
- procurarsi cibo e acqua  
- costruire strumenti  
- trovare un riparo  
- sopravvivere a mostri  
- raccogliere risorse  
- completare fino a **22 obiettivi**

L’architettura **HeRoN** comprende:  
- **NPC** → agente RL  
- **Helper** → LLM zero-shot che suggerisce sequenze di azioni  
- **Reviewer** → LLM fine-tuned che valuta e corregge i suggerimenti dell’Helper  

---

## 🎯 Obiettivi del Progetto

- Fine-tuning del **Reviewer** per i task di Crafter  
- Adattamento dell’**Helper** per generare **sequenze di azioni**  
- Implementazione dell’**NPC** tramite **Deep Q-Network (DQN)**  
- Valutazione delle prestazioni dell’intera architettura HeRoN  

---

## ⚙️ Metodologia di Implementazione

### 1. Sviluppo dell’environment Crafter  
- Analisi preliminare  
- Comprensione degli obiettivi  
- Adattamento dell’environment a HeRoN  

### 2. Implementazione dell’NPC (DQN)  
- Definizione dello stato  
- Definizione delle azioni  
- Training e simulazioni iterative  

### 3. Modifica dell’Helper  
- Prompt engineering per generare **set di azioni coerenti**  

### 4. Fine-tuning del Reviewer  
- Creazione dataset (stati + suggerimenti + feedback)  
- Addestramento tramite RL Fine-Tuning  

### 5. Analisi del numero di azioni  
- Studio del numero ottimale di mosse per ogni chiamata all’Helper  

### 6. Addestramento iterativo  
- Miglioramento del comportamento dell’NPC nelle simulazioni  

### 7. Valutazione  
- Score sugli obiettivi  
- Confronto con agenti baseline  

---

## 📈 Risultati Attesi

- Capacità dell’NPC di eseguire task di Crafter  
- Reviewer efficace nel migliorare Helper  
- Miglioramenti progressivi tramite training iterativo  
- Analisi delle difficoltà e delle soluzioni adottate  

---

## 📚 Risorse Utilizzate

- Paper **HeRoN – A Multi-Agent RL–LLM Framework**  
- Paper **Crafter – Benchmarking the Spectrum of Agent Capabilities**  
- Codice HeRoN  
- GitHub Crafter  

---

# 🇬🇧 English Version

## 👤 Authors
- **Danilo Gisolfi**  
- **Vincenzo Maiellaro**

---

## 📝 Project Overview

This project extends and evaluates the **HeRoN (Helper–Reviewer–NPC)** architecture in the **Crafter** environment, an open-world RL survival game inspired by Minecraft.

Crafter requires the agent to:  
- gather food and water  
- craft tools  
- find shelter  
- avoid monsters  
- collect resources  
- complete **22 achievements**

The **HeRoN** architecture includes:  
- **NPC** → an RL agent (DQN)  
- **Helper** → a zero-shot LLM generating action sequences  
- **Reviewer** → a fine-tuned LLM evaluating and correcting Helper suggestions  

---

## 🎯 Project Goals

- Fine-tune the **Reviewer** for Crafter tasks  
- Adapt the **Helper** to generate **sequences** rather than single actions  
- Implement the **NPC** using **Deep Q-Network**  
- Evaluate HeRoN performance across the 22 Crafter objectives  

---

## ⚙️ Implementation Methodology

### 1. Crafter Environment Study & Integration  
### 2. NPC Development (DQN)  
### 3. Helper Modification via Prompt Engineering  
### 4. Reviewer Fine-Tuning with a Custom Dataset  
### 5. Action-Sequence Optimization  
### 6. Iterative Training Pipeline  
### 7. Performance Evaluation  

---

## 📈 Expected Outcomes

- NPC capable of addressing Crafter tasks  
- Reviewer improving Helper’s suggestions  
- Performance gains via iterative RL training  
- Insight into challenges and limitations  

---

## 📚 Resources

- **HeRoN Framework Paper**  
- **Crafter Benchmark Paper**  
- HeRoN official codebase  
- Crafter GitHub repository  

