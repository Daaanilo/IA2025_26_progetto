# Scripts Directory - HeRoN Pipeline

Questa cartella contiene gli script per l'implementazione completa dell'architettura HeRoN (Helper-Reviewer-NPC) nell'ambiente Crafter.

## 📋 Ordine di Esecuzione

### Script Manager (Raccomandato)
```powershell
# Mostra stato pipeline
python scripts/01_pipeline_manager.py --status

# Mostra metodologia completa
python scripts/01_pipeline_manager.py --methodology

# Esegui step specifico
python scripts/01_pipeline_manager.py --step 1

# Esegui intera pipeline
python scripts/01_pipeline_manager.py --run-all
```

### Esecuzione Manuale

#### **Step 1: Train NPC Baseline** ⏱️ 2-4 ore
Addestra un agente DQN baseline senza l'architettura HeRoN per stabilire prestazioni di riferimento.

```powershell
python scripts/02_train_npc_baseline.py --config configs/dqn_config.yaml --episodes 1000
```

**Output:** `models/baseline/npc_baseline_best.pth`

---

#### **Step 2: Optimize Helper Strategy** ⏱️ 30-60 minuti
Analizza e ottimizza il numero di mosse che l'Helper deve suggerire per chiamata.

```powershell
python scripts/03_optimize_helper_strategy.py --config configs/helper_config.yaml --iterations 100
```

**Output:** `data/helper_analysis/helper_optimization_*.json`

**Risultato:** Determina configurazione ottimale (es. 5 mosse per chiamata, 2 chiamate per episodio)

---

#### **Step 3: Collect Reviewer Data** ⏱️ 1-2 ore
Raccoglie interazioni Helper-Reviewer per creare dataset di fine-tuning.

```powershell
python scripts/04_collect_reviewer_data.py --episodes 200 --output data/reviewer_dataset.jsonl --use-baseline-npc models/baseline/npc_baseline_best.pth
```

**Input:** Baseline NPC (opzionale ma raccomandato)
**Output:** `data/reviewer_dataset.jsonl` (200+ esempi)

---

#### **Step 4: Fine-tune Reviewer** ⏱️ 2-4 ore
Addestra il Reviewer specializzandolo per l'ambiente Crafter.

```powershell
python scripts/05_finetune_reviewer.py --input data/reviewer_dataset.jsonl --output_dir models/reviewer_finetuned
```

**Input:** `data/reviewer_dataset.jsonl`
**Output:** `models/reviewer_finetuned/` (modello fine-tuned)

**Importante:** Aggiorna `configs/reviewer_config.yaml` per puntare al modello fine-tuned

---

#### **Step 5: Train HeRoN Architecture** ⏱️ 4-8 ore
Addestra l'architettura HeRoN completa con ciclo iterativo RL-LLM.

```powershell
python scripts/06_train_heron.py --config configs/heron_config.yaml --episodes 2000 --load_npc models/baseline/npc_baseline_best.pth
```

**Input:** 
- Baseline NPC (opzionale, come starting point)
- Fine-tuned Reviewer

**Output:** `models/heron/heron_best.pth`

**Flusso HeRoN:**
1. NPC percepisce stato
2. Helper propone strategia (sequenza azioni)
3. Reviewer valuta e affina strategia
4. Feedback guida politica NPC
5. NPC impara tramite esperienza

---

#### **Step 6: Evaluate and Compare** ⏱️ 30-60 minuti
Valuta prestazioni finali e confronta HeRoN con baseline.

```powershell
python scripts/07_evaluate_and_compare.py --heron_model models/heron/heron_best.pth --baseline_model models/baseline/npc_baseline_best.pth --episodes 100
```

**Input:**
- HeRoN model
- Baseline model

**Output:** 
- `data/evaluation/evaluation_results_*.json`
- Grafici comparativi
- Report dettagliato achievements

---

## 🎯 Metodologia di Implementazione

### 1. **Sviluppo dell'Environment Crafter**
- Studio preliminare dell'environment
- Analisi dei 22 obiettivi di gioco
- Implementazione wrapper (`src/environment/crafter_env.py`)

### 2. **Sviluppo dell'NPC (Deep Q-Network)**
- Implementazione agente RL (`src/agents/dqn_agent.py`)
- Addestramento baseline senza HeRoN
- Stabilire prestazioni di riferimento

### 3. **Modifica di Helper**
- Prompt engineering per generare SET di azioni (non singole)
- Analisi numero ottimale di mosse
- Test configurazioni diverse

### 4. **Preparazione e Fine-Tuning del Reviewer**
- Raccolta dataset specifico per Crafter
- Fine-tuning per valutazione e correzione strategie
- Validazione feedback mirati

### 5. **Analisi Preliminare**
- Determinare numero ideale di mosse da suggerire
- Bilanciare efficienza e lungimiranza
- Ottimizzare frequenza chiamate Helper

### 6. **Ciclo di Addestramento HeRoN**
- Addestramento iterativo integrato
- Ottimizzazione parametri NPC
- Ciclo RL-LLM completo

### 7. **Valutazione Finale**
- Misurare abilità del NPC nei task
- Analizzare score e achievements sbloccati
- Confrontare con baseline
- Evidenziare miglioramenti

---

## 📊 Risultati Attesi

- ✅ **Dimostrazione abilità del NPC** nei task di Crafter
- ✅ **Validazione efficacia del Reviewer** nel fornire feedback mirati
- ✅ **Ottimizzazione prestazioni** attraverso addestramento iterativo
- ✅ **Analisi comparativa** delle prestazioni
- ✅ **Evidenziazione miglioramenti** o limiti negli scenari di gioco
- ✅ **Documentazione sfide** affrontate e soluzioni implementate

---

## 🔧 Configurazioni

Tutti gli script utilizzano file di configurazione in `configs/`:

- `dqn_config.yaml` - Configurazione agente DQN
- `helper_config.yaml` - Configurazione Helper LLM
- `reviewer_config.yaml` - Configurazione Reviewer LLM
- `heron_config.yaml` - Configurazione architettura HeRoN

Modifica questi file per personalizzare:
- Hyperparametri di training
- Parametri di rete
- Configurazioni LLM
- Frequenza interazioni Helper-Reviewer

---

## 📈 Monitoraggio

Durante il training, monitora:

1. **Reward medio** - Deve aumentare nel tempo
2. **Achievements unlocked** - Quanti dei 22 obiettivi vengono sbloccati
3. **Episode length** - Durata degli episodi
4. **Acceptance rate** - % strategie Helper accettate da Reviewer
5. **Epsilon** - Tasso di esplorazione del NPC

---

## ⚠️ Note Importanti

1. **Tempo totale stimato:** 10-20 ore per pipeline completa
2. **Requisiti GPU:** Raccomandato per fine-tuning Reviewer e training HeRoN
3. **API Keys:** Assicurati di avere configurato `OPENAI_API_KEY` nel `.env`
4. **Spazio disco:** ~5-10 GB per modelli e dati

---

## 🐛 Troubleshooting

### Errore: "OPENAI_API_KEY not found"
```bash
# Crea file .env nella root del progetto
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Errore: Out of Memory durante fine-tuning
- Riduci `batch_size` in `reviewer_config.yaml`
- Abilita `load_in_4bit` per quantizzazione

### Risultati HeRoN peggiori del baseline
- Aumenta numero di episodi di training
- Riduci `acceptance_threshold` in `heron_config.yaml`
- Aumenta dataset per Reviewer fine-tuning

---

## 📞 Supporto

Per problemi o domande:
1. Controlla i log di output degli script
2. Verifica le configurazioni in `configs/`
3. Consulta `data/pipeline_status.json` per stato pipeline
- Usa `python scripts/01_pipeline_manager.py --status` per diagnostica

---

**Buona sperimentazione con HeRoN! 🚀**
