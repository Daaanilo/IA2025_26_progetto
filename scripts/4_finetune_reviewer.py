"""
Step 4: Fine-tune Reviewer LLM

Obiettivo: Addestrare il Reviewer specializzandolo per l'ambiente Crafter.
Il Reviewer impara a valutare e correggere i suggerimenti dell'Helper con coerenza
specifica per il contesto di gioco.

Questo script:
1. Carica il dataset di interazioni Helper-Reviewer
2. Prepara i dati per il fine-tuning
3. Esegue il fine-tuning del modello Reviewer
4. Salva il modello addestrato

Usage (PowerShell):
  conda activate ia2025
  python scripts/4_finetune_reviewer.py --input data/reviewer_dataset.jsonl --output_dir models/reviewer_finetuned
"""
import argparse
import json
from pathlib import Path
import sys

# Add repo root
sys.path.append(str(Path(__file__).parent.parent))

from src.llm import Reviewer
import yaml


def parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser('Fine-tune Reviewer from collected JSONL data')
    p.add_argument('--input', type=str, required=True, 
                   help='Input JSONL file with collected interactions')
    p.add_argument('--output_dir', type=str, required=True, 
                   help='Directory to save fine-tuned model')
    p.add_argument('--reviewer-config', type=str, default='configs/reviewer_config.yaml',
                   help='Reviewer configuration file')
    p.add_argument('--max-examples', type=int, default=None, 
                   help='Limit number of training examples (for testing)')
    p.add_argument('--train-split', type=float, default=0.9,
                   help='Fraction of data to use for training (rest for validation)')
    p.add_argument('--validate', action='store_true',
                   help='Run validation after training')
    return p.parse_args()


def convert_record_to_training_format(record: dict) -> str:
    """
    Convert a collected interaction record into training text format.
    
    The format is:
    Input: [State + Suggested Actions]
    Target: [Review Feedback]
    
    Args:
        record: Dictionary with state, suggested_actions, review_feedback
        
    Returns:
        Formatted training text
    """
    state = record.get('state', {})
    suggested = record.get('suggested_actions', [])
    feedback = record.get('review_feedback', {})

    parts = []
    
    # Input section
    parts.append('### Input:')
    parts.append('\n**Current State:**')
    
    # Format state information
    state_str_parts = []
    for key, value in state.items():
        if key not in ['step', 'episode', 'timestamp']:
            if isinstance(value, dict):
                state_str_parts.append(f"{key}: {json.dumps(value)}")
            else:
                state_str_parts.append(f"{key}: {value}")
    parts.append('\n'.join(state_str_parts))
    
    parts.append('\n\n**Suggested Actions:**')
    parts.append(json.dumps(suggested, ensure_ascii=False))
    
    # Target section
    parts.append('\n\n### Target:')
    try:
        parts.append(json.dumps(feedback, ensure_ascii=False, indent=2))
    except Exception:
        parts.append(str(feedback))
    
    return '\n'.join(parts)


def load_and_prepare_data(input_path: Path, max_examples: int = None, 
                         train_split: float = 0.9):
    """
    Load JSONL data and prepare for training.
    
    Args:
        input_path: Path to JSONL file
        max_examples: Maximum number of examples to use
        train_split: Fraction for training set
        
    Returns:
        Tuple of (train_examples, val_examples)
    """
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    print(f"Loading data from: {input_path}")
    
    all_examples = []
    skipped = 0
    
    with input_path.open('r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            if max_examples and i >= max_examples:
                break
            try:
                rec = json.loads(line)
                text = convert_record_to_training_format(rec)
                all_examples.append({'text': text})
            except Exception as e:
                print(f"  Warning: Skipped line {i+1}: {e}")
                skipped += 1

    print(f"Loaded {len(all_examples)} examples ({skipped} skipped)")
    
    if not all_examples:
        raise SystemExit('No valid examples found in input file.')
    
    # Split into train and validation
    split_idx = int(len(all_examples) * train_split)
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]
    
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    return train_examples, val_examples


def validate_model(reviewer: Reviewer, val_examples: list, num_samples: int = 10):
    """
    Validate the fine-tuned model on validation examples.
    
    Args:
        reviewer: Fine-tuned Reviewer instance
        val_examples: Validation examples
        num_samples: Number of samples to validate
    """
    print("\n" + "="*60)
    print("VALIDATION")
    print("="*60)
    
    import random
    samples = random.sample(val_examples, min(num_samples, len(val_examples)))
    
    for i, example in enumerate(samples):
        print(f"\n--- Validation Sample {i+1} ---")
        # Extract state and actions from text
        text = example['text']
        # This is simplified - in practice you'd parse the text properly
        print(f"Input (truncated): {text[:200]}...")
        print(f"Expected output (truncated): {text[-200:] if len(text) > 200 else text}")
        # Note: Actual validation would involve calling the model and comparing outputs


def main():
    """Main fine-tuning process."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("STEP 4: FINE-TUNE REVIEWER")
    print("="*60)
    
    # Load configuration
    print(f"\nLoading configuration from: {args.reviewer_config}")
    with open(args.reviewer_config, 'r') as f:
        reviewer_cfg = yaml.safe_load(f)
    
    # Load and prepare data
    input_path = Path(args.input)
    train_examples, val_examples = load_and_prepare_data(
        input_path=input_path,
        max_examples=args.max_examples,
        train_split=args.train_split
    )
    
    # Save to temporary JSON file
    tmp_train_json = Path('data') / 'reviewer_training_tmp.json'
    tmp_val_json = Path('data') / 'reviewer_validation_tmp.json'
    tmp_train_json.parent.mkdir(parents=True, exist_ok=True)
    
    with tmp_train_json.open('w', encoding='utf-8') as fh:
        json.dump(train_examples, fh, ensure_ascii=False, indent=2)
    
    with tmp_val_json.open('w', encoding='utf-8') as fh:
        json.dump(val_examples, fh, ensure_ascii=False, indent=2)
    
    print(f"\nTemporary training file: {tmp_train_json}")
    print(f"Temporary validation file: {tmp_val_json}")
    
    # Initialize Reviewer
    print("\nInitializing Reviewer...")
    reviewer = Reviewer(reviewer_cfg, device=reviewer_cfg.get('device', 'cuda'))
    
    try:
        print("Loading base model...")
        reviewer.load_model(use_fine_tuned=False)
    except Exception as e:
        print(f"Warning: Could not load base model: {e}")
        print("Fine-tuning will attempt to initialize the model")

    # Create output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n" + "="*60)
    print("STARTING FINE-TUNING")
    print("="*60)
    print(f"Training examples: {len(train_examples)}")
    print(f"Output directory: {out_dir}")
    
    # Fine-tune
    try:
        reviewer.fine_tune(str(tmp_train_json), str(out_dir))
        print(f"\n✓ Fine-tuning completed successfully")
        print(f"Model saved to: {out_dir}")
        
        # Optional validation
        if args.validate and val_examples:
            try:
                # Reload fine-tuned model
                reviewer.load_model(use_fine_tuned=True)
                validate_model(reviewer, val_examples)
            except Exception as e:
                print(f"Validation failed: {e}")
        
    except Exception as e:
        print(f"\n✗ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*60)
    print("REVIEWER FINE-TUNING COMPLETED")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Test the fine-tuned Reviewer manually")
    print(f"2. Run script 5 to train the full HeRoN architecture")
    print(f"3. Update reviewer_config.yaml to point to: {out_dir}")


if __name__ == '__main__':
    main()
