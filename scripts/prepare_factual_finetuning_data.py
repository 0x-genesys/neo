#!/usr/bin/env python3
"""
Script to prepare fine-tuning data for factual correction.
Uses SciQ (Science Questions) and Dolly-15k (General QA) datasets
formatted for factual fine-tuning with the User/Assistant messaging format.

This script helps fix factual errors in a pretrained model by:
1. SciQ: 2,000 multiple-choice science questions converted to direct answers
   - Forces model to associate concepts with correct domains (e.g., "Gravity" -> "Physics/Mass")
2. Dolly-15k: 2,000 filtered open_qa and general_qa samples
   - Cures geographic/historical drift (e.g., "France = Scotland")
"""
import argparse
import json
from pathlib import Path
import sys
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.finetuning.data_utils import validate_cot_format, inject_identity_samples


def format_user_assistant(instruction: str, response: str, thought: str = None) -> str:
    """
    Format a Q&A pair into the User/Assistant messaging format.
    
    Format:
        User: {question}
        Assistant: {answer}
    
    If thought is provided:
        User: {question}
        Thought: {thought}
        Assistant: {answer}
    
    Args:
        instruction: The question/prompt
        response: The answer/response
        thought: Optional thought process
    
    Returns:
        Formatted string with User/Assistant (and optional Thought) blocks
    """
    if thought:
        return f"User: {instruction}\nThought: {thought}\nAssistant: {response}"
    return f"User: {instruction}\nAssistant: {response}"


def create_sciq_data(output_path: str, num_samples: int = 2000):
    """
    Create SciQ (Science Questions) dataset for factual fine-tuning.
    
    SciQ contains multiple-choice science questions with:
    - question: The science question
    - correct_answer: The correct answer
    - distractors: Incorrect answer options
    - evidence: Scientific evidence supporting the answer
    
    We convert these to direct Q&A format to force factual associations.
    
    Args:
        output_path: Path to save JSONL file
        num_samples: Number of samples to generate (default: 2000)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets library not installed. Install with: pip install datasets")
        return None, None
    
    print("\n" + "="*80)
    print("📥 Loading SciQ Dataset (The Science Fix)")
    print("="*80 + "\n")
    print("   Target: ~2,000 science questions with direct answers")
    print("   Purpose: Force factual associations (e.g., 'Gravity' -> 'Physics/Mass')")
    
    try:
        dataset = load_dataset("sciq", split="train")
    except Exception as e:
        print(f"   ⚠️  Error loading SciQ: {e}")
        return None, None
    
    examples = []
    
    for item in dataset:
        if len(examples) >= num_samples:
            break
        
        question = item.get('question', '')
        correct_answer = item.get('correct_answer', '')
        # FIX: The actual key in the HuggingFace dataset is 'support'
        evidence = item.get('support', '') 
        
        if not question or not correct_answer:
            continue
        
        # Create thought based on evidence for factual grounding
        # NEVER let thought remain None/null
        if evidence and len(evidence) > 10:
            # Use evidence as thought for factual grounding
            thought = f"Let's recall the scientific evidence: {evidence[:400]}..."
        else:
            # FALLBACK: If no support exists, force the model to state the topic
            thought = f"To answer this, I need to recall scientific facts regarding the topic of: {question[:50]}..."
        
        example = {
            "instruction": question,
            "thought": thought,
            "response": correct_answer,
        }
        
        examples.append(example)
    
    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"   ✅ Processed {len(examples)} SciQ samples")
    print(f"   Saved to: {output_path}")
    
    return str(output_path), len(examples)


def create_dolly_filtered_data(output_path: str, num_samples: int = 2000):
    """
    Create filtered Dolly-15k dataset for factual fine-tuning.
    
    Filters to only include:
    - open_qa: Open-ended questions requiring factual answers
    - general_qa: General knowledge questions
    
    Excludes:
    - summarization (not Q&A format)
    - information_extraction (not direct Q&A)
    - creative_writing (not factual)
    - brainstorming (not factual)
    - classification (not direct Q&A)
    
    Args:
        output_path: Path to save JSONL file
        num_samples: Number of samples to generate (default: 2000)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets library not installed. Install with: pip install datasets")
        return None, None
    
    print("\n" + "="*80)
    print("📥 Loading Dolly-15k Dataset (The Encyclopedia Fix)")
    print("="*80 + "\n")
    print("   Target: ~2,000 open-ended and general knowledge questions")
    print("   Purpose: Cure geographic/historical drift (e.g., 'France = Scotland')")
    print("   Filtering: Only open_qa and general_qa categories")
    
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
    except Exception as e:
        print(f"   ⚠️  Error loading Dolly: {e}")
        return None, None
    
    # Allowed categories for factual fine-tuning
    allowed_categories = {'open_qa', 'general_qa'}
    
    examples = []
    
    for item in dataset:
        if len(examples) >= num_samples:
            break
        
        instruction = item.get('instruction', '')
        context = item.get('context', '')
        response = item.get('response', '')
        category = item.get('category', '')
        
        # Filter by category
        if category not in allowed_categories:
            continue
        
        if not instruction or not response:
            continue
        
        # Combine instruction and context
        user_message = instruction
        if context:
            user_message = f"{context}\n\n{instruction}"
        
        # Create thought based on category - use diverse, fact-based thoughts
        # Randomize prefixes to prevent the model from memorizing a single 'filler' sentence
        import random
        open_qa_prefixes = [
            f"Analyzing the query regarding",
            f"Let me look into",
            f"Looking at the question about",
            f"Examining the topic of",
            f"Considering the question on"
        ]
        general_qa_prefixes = [
            f"Providing information about",
            f"Focusing on the topic of",
            f"Delivering facts about",
            f"Sharing knowledge about",
            f"Addressing the topic of"
        ]
        
        if category == 'open_qa':
            thought = f"{random.choice(open_qa_prefixes)}: {instruction[:100]}..."
        elif category == 'general_qa':
            thought = f"{random.choice(general_qa_prefixes)}: {instruction[:100]}..."
        else:
            thought = f"I will analyze this request and provide a helpful, factual response. Category: {category}."
        
        example = {
            "instruction": user_message,
            "thought": thought,
            "response": response,
        }
        
        examples.append(example)
    
    # Save to JSONL
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"   ✅ Processed {len(examples)} Dolly samples (open_qa + general_qa)")
    print(f"   Saved to: {output_path}")
    
    return str(output_path), len(examples)


def merge_datasets(input_paths: list, output_path: str):
    """Merge multiple JSONL datasets into one."""
    all_examples = []
    
    for input_path in input_paths:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    all_examples.append(json.loads(line))
    
    # Shuffle
    import random
    random.shuffle(all_examples)
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Merged {len(all_examples)} examples to {output_path}")


def split_dataset(input_path: str, train_path: str, val_path: str, val_ratio: float = 0.1):
    """Split dataset into train and validation sets."""
    import random
    
    # Load all examples
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    # Shuffle
    random.shuffle(examples)
    
    # Split
    val_size = int(len(examples) * val_ratio)
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]
    
    # Save train
    train_path = Path(train_path)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save val
    val_path = Path(val_path)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_path, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ Split dataset:")
    print(f"   Train: {len(train_examples)} examples -> {train_path}")
    print(f"   Val: {len(val_examples)} examples -> {val_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare fine-tuning data for factual correction using SciQ and Dolly-15k'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # SciQ command
    sciq_parser = subparsers.add_parser('sciq', help='Create SciQ dataset')
    sciq_parser.add_argument('--output', default='data/sciq.jsonl', help='Output path')
    sciq_parser.add_argument('--num-samples', type=int, default=2000, help='Number of samples')
    
    # Dolly command
    dolly_parser = subparsers.add_parser('dolly', help='Create filtered Dolly-15k dataset')
    dolly_parser.add_argument('--output', default='data/dolly_filtered.jsonl', help='Output path')
    dolly_parser.add_argument('--num-samples', type=int, default=2000, help='Number of samples')
    
    # Merge command
    merge_parser = subparsers.add_parser('merge', help='Merge multiple datasets')
    merge_parser.add_argument('--inputs', nargs='+', required=True, help='Input JSONL files')
    merge_parser.add_argument('--output', required=True, help='Output path')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val')
    split_parser.add_argument('--input', required=True, help='Input JSONL file')
    split_parser.add_argument('--train', default='data/factual_train.jsonl', help='Train output path')
    split_parser.add_argument('--val', default='data/factual_val.jsonl', help='Val output path')
    split_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate format')
    validate_parser.add_argument('input', help='Input JSONL file')
    
    # All command (create complete dataset)
    all_parser = subparsers.add_parser('all', help='Create complete factual dataset (SciQ + Dolly)')
    all_parser.add_argument('--output-dir', default='data/factual', help='Output directory')
    all_parser.add_argument('--sciq-samples', type=int, default=2000, help='SciQ samples')
    all_parser.add_argument('--dolly-samples', type=int, default=2000, help='Dolly samples')
    all_parser.add_argument('--identity-samples', type=int, default=50, help='Identity anchor samples')
    all_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation ratio')
    
    # HuggingFace datasets command
    hf_parser = subparsers.add_parser('hf', help='Pull and process HuggingFace datasets')
    hf_parser.add_argument('--config', default='config/finetuning_config.yaml', help='YAML config path')
    hf_parser.add_argument('--output-dir', default='data/factual_hf', help='Output directory')
    hf_parser.add_argument('--max-tokens', type=int, default=480, help='Max tokens per sample')
    hf_parser.add_argument('--sciq-samples-target', type=int, default=None, help='Target SciQ sample count')
    hf_parser.add_argument('--dolly-samples-target', type=int, default=None, help='Target Dolly sample count')
    hf_parser.add_argument('--identity-samples', type=int, default=None, help='Number of identity samples')
    
    args = parser.parse_args()
    
    if args.command == 'sciq':
        create_sciq_data(args.output, args.num_samples)
    
    elif args.command == 'dolly':
        create_dolly_filtered_data(args.output, args.num_samples)
    
    elif args.command == 'merge':
        merge_datasets(args.inputs, args.output)
    
    elif args.command == 'split':
        split_dataset(args.input, args.train, args.val, args.val_ratio)
    
    elif args.command == 'validate':
        validate_cot_format(args.input)
    
    elif args.command == 'all':
        print("\n" + "="*80)
        print("Creating Complete Factual Fine-Tuning Dataset")
        print("="*80 + "\n")
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual datasets
        sciq_path = output_dir / 'sciq.jsonl'
        dolly_path = output_dir / 'dolly_filtered.jsonl'
        
        print("\n" + "="*80)
        print("Step 1: Creating SciQ Dataset (Science Questions)")
        print("="*80)
        create_sciq_data(str(sciq_path), args.sciq_samples)
        
        print("\n" + "="*80)
        print("Step 2: Creating Filtered Dolly-15k Dataset")
        print("="*80)
        create_dolly_filtered_data(str(dolly_path), args.dolly_samples)
        
        # Merge
        merged_path = output_dir / 'factual_all.jsonl'
        merge_datasets([str(sciq_path), str(dolly_path)], str(merged_path))
        
        # Split
        train_path = output_dir / 'factual_train.jsonl'
        val_path = output_dir / 'factual_val.jsonl'
        split_dataset(str(merged_path), str(train_path), str(val_path), args.val_ratio)
        
        print("\n" + "="*80)
        print("✅ Complete Factual Dataset Created!")
        print("="*80)
        print(f"\nFiles created:")
        print(f"  - {train_path}")
        print(f"  - {val_path}")
        print(f"\nReady for fine-tuning!")
        print("="*80 + "\n")
    
    elif args.command == 'hf':
        hf_config = {}
        config_path = Path(args.config)
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                hf_config = (config.get('data', {}) or {}).get('hf_mix', {}) or {}
                print(f"✅ Loaded HF mix config from {config_path}")
            except Exception as e:
                print(f"⚠️  Could not load HF mix config from {config_path}: {e}")
                print("   Falling back to built-in defaults / CLI args")
        else:
            print(f"⚠️  Config file not found: {config_path}")
            print("   Falling back to built-in defaults / CLI args")
        
        sciq_samples_target = (
            args.sciq_samples_target
            if args.sciq_samples_target is not None
            else hf_config.get('sciq_samples_target', 2000)
        )
        dolly_samples_target = (
            args.dolly_samples_target
            if args.dolly_samples_target is not None
            else hf_config.get('dolly_samples_target', 2000)
        )
        identity_samples = (
            args.identity_samples
            if args.identity_samples is not None
            else hf_config.get('identity_samples', 50)
        )
        
        print("\nHF dataset mix:")
        print(f"  - SciQ target: {sciq_samples_target}")
        print(f"  - Dolly target: {dolly_samples_target}")
        print(f"  - Identity samples: {identity_samples}")
        
        # Load tokenizer for length filtering
        try:
            from src.tokenizer_utils import load_tokenizer
            tokenizer = load_tokenizer()
            print("✅ Loaded tokenizer for length filtering")
        except Exception as e:
            print(f"⚠️  Could not load tokenizer: {e}")
            print("   Proceeding without length filtering")
            tokenizer = None
        
        train_path, val_path = pull_and_process_factual_hf_datasets(
            output_dir=args.output_dir,
            max_tokens=args.max_tokens,
            tokenizer=tokenizer,
            sciq_samples_target=sciq_samples_target,
            dolly_samples_target=dolly_samples_target,
            identity_samples=identity_samples,
        )
        
        if train_path and val_path:
            print("\n✅ HuggingFace datasets processed successfully!")
            print(f"\nUse these files for fine-tuning:")
            print(f"  python src/finetuning/gpu_finetune.py \\")
            print(f"    --train-data {train_path} \\")
            print(f"    --val-data {val_path}")
    
    else:
        parser.print_help()


def pull_and_process_factual_hf_datasets(
    output_dir: str = "data/factual_hf",
    max_tokens: int = 480,
    tokenizer=None,
    sciq_samples_target: int = 2000,
    dolly_samples_target: int = 2000,
    identity_samples: int = 50,
) -> tuple:
    """
    Pull and process HuggingFace datasets into factual fine-tuning format.
    
    Sources:
    - allenai/sciq (SciQ) - Science questions with direct answers
    - databricks/databricks-dolly-15k (Dolly) - Filtered open_qa and general_qa
    
    Processing:
    - Filter by length (max 480 tokens for safety buffer)
    - Map to instruction/thought/response format
    - Inject identity samples (50 samples)
    - Shuffle and merge
    - Split into train/val (90/10)
    
    Args:
        output_dir: Output directory for processed data
        max_tokens: Maximum token count per sample (default: 480)
        tokenizer: Tokenizer for length filtering (optional)
        sciq_samples_target: Target number of SciQ samples (default: 2000)
        dolly_samples_target: Target number of Dolly samples (default: 2000)
        identity_samples: Number of identity samples to inject
    
    Returns:
        (train_path, val_path) tuple
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets library not installed. Install with: pip install datasets")
        return None, None
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_examples = []
    
    print("\n" + "="*80)
    print("📥 Pulling and Processing HuggingFace Datasets for Factual Fine-Tuning")
    print("="*80 + "\n")
    
    sciq_samples = []
    dolly_samples = []
    
    # 1. SciQ (Science Questions) - Factual Knowledge Anchor
    print("1️⃣  Processing allenai/sciq (Science Questions).")
    print(f"   Target: {sciq_samples_target:,} samples")
    print("   Purpose: Force factual associations (e.g., 'Gravity' -> 'Physics/Mass')")
    try:
        sciq_dataset = load_dataset("allenai/sciq", split="train")
        
        for item in sciq_dataset:
            if len(sciq_samples) >= sciq_samples_target:
                break
            
            question = item.get('question', '')
            correct_answer = item.get('correct_answer', '')
            evidence = item.get('evidence', '')
            
            if not question or not correct_answer:
                continue
            
            # Create thought based on evidence for factual grounding
            thought = None
            if evidence and len(evidence) > 50:
                thought = evidence[:400]
            
            example = {
                "instruction": question,
                "thought": thought,
                "response": correct_answer,
            }
            
            # Filter by length if tokenizer provided
            if tokenizer:
                text = format_user_assistant(question, correct_answer, thought)
                tokens = tokenizer.encode(text)
                if len(tokens) > max_tokens:
                    continue
            
            sciq_samples.append(example)
        
        all_examples.extend(sciq_samples)
        print(f"   ✅ Processed {len(sciq_samples)} science samples")
    
    except Exception as e:
        print(f"   ⚠️  Error processing SciQ: {e}")
    
    # 2. Databricks Dolly (filtered) - General Assistant
    print("\n2️⃣  Processing databricks/databricks-dolly-15k (General Assistant).")
    print(f"   Target: {dolly_samples_target:,} samples")
    print("   Purpose: Cure geographic/historical drift (e.g., 'France = Scotland')")
    print("   Filtering: Only open_qa and general_qa categories")
    try:
        dolly_dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        
        allowed_categories = {'open_qa', 'general_qa'}
        
        for item in dolly_dataset:
            if len(dolly_samples) >= dolly_samples_target:
                break
            
            instruction = item.get('instruction', '')
            context = item.get('context', '')
            response = item.get('response', '')
            category = item.get('category', '')
            
            # Filter by category
            if category not in allowed_categories:
                continue
            
            if not instruction or not response:
                continue
            
            # Combine instruction and context
            user_message = instruction
            if context:
                user_message = f"{context}\n\n{instruction}"
            
            # Create thought based on category - use diverse, fact-based thoughts
            # Randomize prefixes to prevent the model from memorizing a single 'filler' sentence
            import random
            open_qa_prefixes = [
                f"Analyzing the query regarding",
                f"Let me look into",
                f"Looking at the question about",
                f"Examining the topic of",
                f"Considering the question on"
            ]
            general_qa_prefixes = [
                f"Providing information about",
                f"Focusing on the topic of",
                f"Delivering facts about",
                f"Sharing knowledge about",
                f"Addressing the topic of"
            ]
            
            if category == 'open_qa':
                thought = f"{random.choice(open_qa_prefixes)}: {instruction[:100]}..."
            elif category == 'general_qa':
                thought = f"{random.choice(general_qa_prefixes)}: {instruction[:100]}..."
            else:
                thought = f"I will analyze this request and provide a helpful, factual response. Category: {category}."
            
            example = {
                "instruction": user_message,
                "thought": thought,
                "response": response,
            }
            
            # Filter by length
            if tokenizer:
                text = format_user_assistant(user_message, response, thought)
                tokens = tokenizer.encode(text)
                if len(tokens) > max_tokens:
                    continue
            
            dolly_samples.append(example)
        
        all_examples.extend(dolly_samples)
        print(f"   ✅ Processed {len(dolly_samples)} general Q&A samples")
    
    except Exception as e:
        print(f"   ⚠️  Error processing Dolly: {e}")
    
    # 3. Inject Identity Samples
    print("\n3️⃣  Injecting Identity Samples...")
    all_examples = inject_identity_samples(all_examples, num_samples=identity_samples)
    
    # Shuffle all examples
    print(f"\n4️⃣  Shuffling {len(all_examples)} total examples...")
    import random
    random.shuffle(all_examples)
    
    # Split into train/val (90/10)
    val_size = int(len(all_examples) * 0.1)
    val_examples = all_examples[:val_size]
    train_examples = all_examples[val_size:]
    
    # Save train
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    
    # Save val
    val_path = output_dir / "val.jsonl"
    with open(val_path, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"\n{'='*80}")
    print("✅ Dataset Processing Complete!")
    print(f"{'='*80}")
    print(f"\nDataset Composition:")
    print(f"  - Science Questions (SciQ): {len(sciq_samples):,} samples")
    print(f"  - General Assistant (Dolly): {len(dolly_samples):,} samples")
    print(f"  - Identity Anchors: {identity_samples:,} samples")
    print(f"  - Total: {len(all_examples)} samples")
    print(f"\nSplit:")
    print(f"  - Train: {len(train_examples)} samples -> {train_path}")
    print(f"  - Val: {len(val_examples)} samples -> {val_path}")
    print(f"\nMax tokens per sample: {max_tokens}")
    print(f"Format: User: {{question}}\\nThought: {{thought}}\\nAssistant: {{answer}}")
    print(f"{'='*80}\n")
    
    return str(train_path), str(val_path)


if __name__ == '__main__':
    main()
