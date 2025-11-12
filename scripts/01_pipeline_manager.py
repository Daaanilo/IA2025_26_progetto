"""
Pipeline Overview and Workflow Manager

Questo script fornisce una panoramica completa della pipeline HeRoN per Crafter
e permette di eseguire l'intera metodologia in sequenza o singoli step.

METODOLOGIA DI IMPLEMENTAZIONE:

1. Sviluppo dell'environment Crafter
   - Studio preliminare dell'environment Crafter
   - Analisi degli obiettivi di gioco (22 achievements)
   - Implementazione dell'environment wrapper

2. Sviluppo dell'NPC (Baseline)
   - Implementare l'NPC tramite Deep Q-Network (DQN)
   - Addestrare baseline senza architettura HeRoN
   - Stabilire prestazioni di riferimento

3. Ottimizzazione Helper e Analisi Mosse
   - Prompt engineering su Helper per generare set di azioni
   - Analisi del numero ottimale di mosse da suggerire
   - Test configurazioni (es. due chiamate da cinque mosse ciascuna)

4. Raccolta Dati per Reviewer
   - Generare dataset contenente:
     * Stati dell'environment
     * Azioni suggerite dall'Helper
     * Feedback correttivi e mirati

5. Fine-Tuning del Reviewer
   - Preparazione dataset per training
   - Addestramento del Reviewer tramite fine-tuning
   - Validazione del modello addestrato

6. Addestramento Iterativo HeRoN
   - Integrare NPC, Helper, e Reviewer
   - Condurre sessioni di addestramento iterativo
   - Ottimizzare parametri del NPC attraverso simulazioni
   - Ciclo RL-LLM completo

7. Valutazione Finale delle Prestazioni
   - Valutare prestazioni del NPC HeRoN
   - Confrontare con baseline
   - Analisi score e obiettivi sbloccati
   - Validazione efficacia del Reviewer

Usage (PowerShell):
  # Esegui intera pipeline
  conda activate ia2025
  python scripts/01_pipeline_manager.py --run-all

  # Esegui singolo step
  python scripts/01_pipeline_manager.py --step 1

  # Mostra stato pipeline
  python scripts/01_pipeline_manager.py --status
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


class PipelineManager:
    """Gestisce l'esecuzione della pipeline HeRoN."""

    def __init__(self):
        self.steps = [
            {
                "id": 1,
                "name": "Train NPC Baseline",
                "script": "02_train_npc_baseline.py",
                "description": "Addestra agente DQN baseline senza HeRoN",
                "estimated_time": "2-4 ore",
                "outputs": ["models/baseline/npc_baseline_best.pth"],
                "dependencies": [],
            },
            {
                "id": 2,
                "name": "Optimize Helper Strategy",
                "script": "03_optimize_helper_strategy.py",
                "description": "Analizza e ottimizza numero di mosse per Helper",
                "estimated_time": "30-60 minuti",
                "outputs": ["data/helper_analysis/helper_optimization_*.json"],
                "dependencies": [],
            },
            {
                "id": 3,
                "name": "Collect Reviewer Data",
                "script": "04_collect_reviewer_data.py",
                "description": "Genera dataset sintetici per training Reviewer",
                "estimated_time": "5-10 minuti",
                "outputs": ["data/reviewer_dataset.jsonl"],
                "dependencies": [],
            },
            {
                "id": 4,
                "name": "Fine-tune Reviewer",
                "script": "05_finetune_reviewer.py",
                "description": "Addestra Reviewer con RL (PPO)",
                "estimated_time": "1-2 ore",
                "outputs": ["models/reviewer_finetuned/"],
                "dependencies": [3],
            },
            {
                "id": 5,
                "name": "Train HeRoN Architecture",
                "script": "06_train_heron.py",
                "description": "Addestra architettura HeRoN completa (NPC+Helper+Reviewer)",
                "estimated_time": "4-8 ore",
                "outputs": ["models/heron/heron_best.pth"],
                "dependencies": [1, 4],  # Può usare baseline come starting point
            },
            {
                "id": 6,
                "name": "Evaluate and Compare",
                "script": "07_evaluate_and_compare.py",
                "description": "Valuta prestazioni finali e confronta con baseline",
                "estimated_time": "30-60 minuti",
                "outputs": ["data/evaluation/evaluation_results_*.json"],
                "dependencies": [1, 5],
            },
        ]

        self.scripts_dir = Path(__file__).parent
        self.project_root = self.scripts_dir.parent
        self.status_file = self.project_root / "data" / "pipeline_status.json"

    def load_status(self):
        """Carica lo stato della pipeline."""
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                return json.load(f)
        return {"completed_steps": [], "last_run": None}

    def save_status(self, status):
        """Salva lo stato della pipeline."""
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def check_outputs_exist(self, step):
        """Verifica se gli output di uno step esistono."""
        for output_pattern in step["outputs"]:
            # Gestisci pattern con wildcard
            if "*" in output_pattern:
                parent = Path(output_pattern).parent
                pattern = Path(output_pattern).name
                if parent.exists() and list(parent.glob(pattern)):
                    return True
            else:
                if (self.project_root / output_pattern).exists():
                    return True
        return False

    def print_status(self):
        """Stampa lo stato della pipeline."""
        status = self.load_status()
        completed = set(status.get("completed_steps", []))

        print("\n" + "=" * 80)
        print("HERON PIPELINE STATUS")
        print("=" * 80)

        if status.get("last_run"):
            print(f"\nUltima esecuzione: {status['last_run']}")

        print("\nSteps:")
        for step in self.steps:
            status_icon = "✅" if step["id"] in completed else "⬜"
            outputs_exist = "📁" if self.check_outputs_exist(step) else "  "

            print(f"\n{status_icon} {outputs_exist} Step {step['id']}: {step['name']}")
            print(f"   Script: {step['script']}")
            print(f"   Descrizione: {step['description']}")
            print(f"   Tempo stimato: {step['estimated_time']}")

            if step["dependencies"]:
                deps = ", ".join([f"Step {d}" for d in step["dependencies"]])
                print(f"   Dipendenze: {deps}")

            # Controlla se le dipendenze sono soddisfatte
            deps_met = all(d in completed for d in step["dependencies"])
            if not deps_met and step["dependencies"]:
                print("   ⚠️  Dipendenze non soddisfatte")

        print("\n" + "=" * 80)
        total_steps = len(self.steps)
        completed_count = len(completed)
        progress = completed_count / total_steps * 100
        print(
            f"Progresso: {completed_count}/{total_steps} steps completati ({progress:.1f}%)"
        )
        print("=" * 80)

    def run_step(self, step_id, extra_args=None, non_interactive: bool = False):
        """Esegue uno specifico step della pipeline.

        Args:
            step_id: id dello step da eseguire
            extra_args: argomenti addizionali da passare allo script
            non_interactive: se True, accetta automaticamente le conferme sulle dipendenze
        """
        step = next((s for s in self.steps if s["id"] == step_id), None)
        if not step:
            print(f"❌ Step {step_id} non trovato")
            return False

        # Verifica dipendenze
        status = self.load_status()
        completed = set(status.get("completed_steps", []))

        for dep in step["dependencies"]:
            if dep not in completed:
                print(f"⚠️  Attenzione: Step {dep} non è stato completato")
                if non_interactive:
                    print("  -> Non-interactive mode: proseguo comunque")
                else:
                    response = input("Vuoi continuare comunque? (y/n): ")
                    if response.lower() != "y":
                        return False

        print("\n" + "=" * 80)
        print(f"ESECUZIONE STEP {step_id}: {step['name']}")
        print("=" * 80)
        print(f"Script: {step['script']}")
        print(f"Descrizione: {step['description']}")
        print(f"Tempo stimato: {step['estimated_time']}")
        print("=" * 80 + "\n")

        # Costruisci comando
        script_path = self.scripts_dir / step["script"]
        cmd = [sys.executable, str(script_path)]

        if extra_args:
            cmd.extend(extra_args)

        # Esegui
        try:
            result = subprocess.run(cmd, cwd=str(self.project_root))

            if result.returncode == 0:
                # Aggiorna status
                if step_id not in completed:
                    completed.add(step_id)
                    status["completed_steps"] = list(completed)
                    status["last_run"] = datetime.now().isoformat()
                    self.save_status(status)

                print(f"\n✅ Step {step_id} completato con successo")
                return True
            else:
                print(f"\n❌ Step {step_id} fallito con codice {result.returncode}")
                return False

        except Exception as e:
            print(f"\n❌ Errore durante l'esecuzione: {e}")
            return False

    def run_all(self, non_interactive: bool = False):
        """Esegue l'intera pipeline."""
        print("\n" + "=" * 80)
        print("ESECUZIONE COMPLETA PIPELINE HERON")
        print("=" * 80)
        print("\nQuesta operazione eseguirà tutti i 6 steps in sequenza.")
        print("Tempo totale stimato: 10-20 ore")
        print("\nSteps da eseguire:")
        for step in self.steps:
            print(f"  {step['id']}. {step['name']} ({step['estimated_time']})")

        if non_interactive:
            print(
                "\nNon-interactive mode: procedo con l'esecuzione automatica della pipeline"
            )
        else:
            response = input("\nVuoi continuare? (y/n): ")
            if response.lower() != "y":
                print("Operazione annullata")
                return

        for step in self.steps:
            success = self.run_step(step["id"], non_interactive=non_interactive)
            if not success:
                print(f"\n⚠️  Pipeline interrotta allo step {step['id']}")
                print("Puoi riprendere l'esecuzione in seguito con:")
                print(f"  python scripts/01_pipeline_manager.py --step {step['id']}")
                return

        print("\n" + "=" * 80)
        print("🎉 PIPELINE COMPLETATA CON SUCCESSO!")
        print("=" * 80)
        print("\nRisultati disponibili in:")
        print("  - models/baseline/     (NPC baseline)")
        print("  - models/heron/        (NPC con HeRoN)")
        print("  - data/evaluation/     (Risultati valutazione)")
        print("\nProssimi passi:")
        print("  1. Analizza i risultati in data/evaluation/")
        print("  2. Confronta le prestazioni HeRoN vs Baseline")
        print("  3. Visualizza i grafici generati")

    def print_methodology(self):
        """Stampa la metodologia completa."""
        print("\n" + "=" * 80)
        print("METODOLOGIA DI IMPLEMENTAZIONE HERON PER CRAFTER")
        print("=" * 80)

        methodology = [
            {
                "title": "1. Sviluppo dell'environment Crafter",
                "points": [
                    "Studio preliminare dell'environment Crafter",
                    "Analisi dei 22 obiettivi di gioco (achievements)",
                    "Implementazione wrapper per integrazione con HeRoN",
                    "Test funzionalità base dell'environment",
                ],
            },
            {
                "title": "2. Sviluppo dell'NPC (Agente RL Baseline)",
                "points": [
                    "Implementazione Deep Q-Network (DQN) agent",
                    "Addestramento baseline SENZA architettura HeRoN",
                    "Stabilire prestazioni di riferimento",
                    "Salvare modello baseline per confronti futuri",
                    "📝 Script: 02_train_npc_baseline.py",
                ],
            },
            {
                "title": "3. Modifica di Helper e Analisi Mosse",
                "points": [
                    "Prompt engineering su Helper per generare SET di azioni",
                    "Test diverse configurazioni (3, 5, 7, 10 mosse)",
                    "Analisi numero ottimale di mosse da suggerire",
                    "Esempio: due chiamate da cinque mosse ciascuna",
                    "Valutazione coerenza e validità delle sequenze",
                    "📝 Script: 03_optimize_helper_strategy.py",
                ],
            },
            {
                "title": "4. Raccolta Dati per Reviewer",
                "points": [
                    "Generazione dataset specifico per Crafter contenente:",
                    "  - Stati dell'environment",
                    "  - Azioni suggerite dall'Helper",
                    "  - Feedback correttivi e mirati",
                    "Utilizzo baseline NPC per traiettorie realistiche (opzionale)",
                    "Raccolta 200+ interazioni Helper-Reviewer",
                    "📝 Script: 04_collect_reviewer_data.py",
                ],
            },
            {
                "title": "5. Fine-Tuning del Reviewer",
                "points": [
                    "Preparazione dataset in formato training",
                    "Fine-tuning del Reviewer (LoRA/QLoRA)",
                    "Specializzazione per ambiente Crafter",
                    "Validazione capacità di valutazione e correzione",
                    "Salvataggio modello fine-tuned",
                    "📝 Script: 05_finetune_reviewer.py",
                ],
            },
            {
                "title": "6. Addestramento Iterativo HeRoN",
                "points": [
                    "Integrazione completa: NPC + Helper + Reviewer",
                    "Ciclo iterativo RL-LLM:",
                    "  1. NPC percepisce stato",
                    "  2. Helper propone strategia (sequenza azioni)",
                    "  3. Reviewer valuta e affina strategia",
                    "  4. Feedback guida politica NPC",
                    "  5. NPC impara tramite esperienza",
                    "Ottimizzazione parametri attraverso simulazioni",
                    "Monitoraggio metriche e convergenza",
                    "📝 Script: 06_train_heron.py",
                ],
            },
            {
                "title": "7. Valutazione delle Prestazioni",
                "points": [
                    "Valutazione NPC HeRoN vs Baseline",
                    "Metriche: score, achievements unlocked, episode length",
                    "Analisi 22 obiettivi di Crafter",
                    "Validazione efficacia Reviewer nel fornire feedback",
                    "Dimostrazione abilità del NPC nei task",
                    "Identificazione miglioramenti e limiti",
                    "📝 Script: 07_evaluate_and_compare.py",
                ],
            },
        ]

        for section in methodology:
            print(f"\n{section['title']}")
            print("-" * 80)
            for point in section["points"]:
                print(f"  • {point}")

        print("\n" + "=" * 80)
        print("RISULTATI ATTESI")
        print("=" * 80)
        print(
            """
  ✓ Dimostrazione dell'abilità del NPC nello svolgere i task di Crafter
  ✓ Validazione dell'efficacia del Reviewer nel fornire feedback mirati
  ✓ Ottimizzazione delle prestazioni attraverso addestramento iterativo
  ✓ Analisi comparativa delle prestazioni NPC in Crafter
  ✓ Evidenziazione di miglioramenti o limiti negli scenari di gioco
  ✓ Documentazione delle sfide affrontate e soluzioni implementate
        """
        )

        print("=" * 80)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HeRoN Pipeline Manager per Crafter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--status", action="store_true", help="Mostra stato della pipeline"
    )
    group.add_argument("--step", type=int, help="Esegui uno specifico step (1-6)")
    group.add_argument(
        "--run-all", action="store_true", help="Esegui l'intera pipeline"
    )
    group.add_argument(
        "--methodology", action="store_true", help="Mostra metodologia completa"
    )

    parser.add_argument(
        "--args",
        nargs=argparse.REMAINDER,
        help="Argomenti aggiuntivi da passare allo script",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Non-interactive: accetta automaticamente le conferme",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    manager = PipelineManager()

    if args.methodology:
        manager.print_methodology()
    elif args.status:
        manager.print_status()
    elif args.step:
        extra_args = args.args if args.args else None
        manager.run_step(args.step, extra_args, non_interactive=args.yes)
    elif args.run_all:
        manager.run_all(non_interactive=args.yes)
    else:
        # Default: mostra help e status
        print("\n" + "=" * 80)
        print("HERON PIPELINE MANAGER")
        print("=" * 80)
        print("\nGestisce l'esecuzione completa della pipeline HeRoN per Crafter")
        print("\nComandi disponibili:")
        print("  --status        Mostra stato corrente della pipeline")
        print("  --methodology   Mostra metodologia completa di implementazione")
        print("  --step N        Esegui step specifico (1-6)")
        print("  --run-all       Esegui intera pipeline")
        print("\nEsempi:")
        print("  python scripts/01_pipeline_manager.py --status")
        print("  python scripts/01_pipeline_manager.py --step 1")
        print("  python scripts/01_pipeline_manager.py --step 1 --args --episodes 500")
        print("  python scripts/01_pipeline_manager.py --run-all")

        print("\n")
        manager.print_status()


if __name__ == "__main__":
    main()
