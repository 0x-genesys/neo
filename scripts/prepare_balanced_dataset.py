"""
Prepare a 300M token "Conversational Scholar" binary dataset.

This script creates a high-quality, balanced dataset combining:
1. WikiText-103 (encyclopedic knowledge)
2. UltraChat (conversational AI)

Total: 300M tokens for training, 10M tokens for validation
Tokenizer: tiktoken cl100k_base (GPT-4)
Format: Binary np.uint32 with np.memmap for memory efficiency
"""

import numpy as np
import tiktoken
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm
import random
from typing import List
import json

# Configure tqdm to avoid newline issues
tqdm.pandas(
    desc="Processing",
    unit="items", 
    dynamic_ncols=True,
    leave=True,
    position=0,
    ascii=False,
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
)


class BalancedDatasetBuilder:
    """Build the 80/20 Conversational Scholar dataset."""
    
    def __init__(self, output_dir: str = "data/balanced_300m", seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPT-4 tokenizer (cl100k_base)
        print("Loading tiktoken cl100k_base (GPT-4) tokenizer...")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.tokenizer.n_vocab
        print(f"✅ Tokenizer loaded: vocab_size={self.vocab_size:,}")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Target token counts for the 80/20 Conversational Scholar mix.
        self.target_tokens = {
            'wikitext': 240_000_000,
            'ultrachat': 60_000_000,
        }
        
        self.total_target = 300_000_000
        self.val_target = 10_000_000
        
        # Statistics
        self.stats = {
            'wikitext': {'docs': 0, 'tokens': 0},
            'ultrachat': {'docs': 0, 'tokens': 0},
        }
    
    def format_conversation(self, user_text: str, assistant_text: str) -> str:
        """Format conversational data with consistent template."""
        return f"User: {user_text}\nAssistant: {assistant_text}"
    
    def tokenize_text(self, text: str) -> List[int]:
        """Tokenize text using tiktoken."""
        return self.tokenizer.encode(text, allowed_special='all')

    def count_tokens(self, documents: List[List[int]]) -> int:
        """Count tokens in a list of tokenized documents."""
        return sum(len(doc) for doc in documents)

    def fit_documents_to_target(
        self,
        documents: List[List[int]],
        target_tokens: int,
        source_name: str,
    ) -> List[List[int]]:
        """Trim or repeat documents so a source lands exactly on its token target."""
        if not documents:
            raise RuntimeError(f"No documents available for {source_name}")

        fitted = []
        total_tokens = 0

        for doc in documents:
            if total_tokens >= target_tokens:
                break
            remaining = target_tokens - total_tokens
            fitted_doc = doc if len(doc) <= remaining else doc[:remaining]
            fitted.append(fitted_doc)
            total_tokens += len(fitted_doc)

        if total_tokens < target_tokens:
            print(
                f"WARNING: {source_name} only provided {total_tokens:,} tokens; "
                f"repeating shuffled documents to reach {target_tokens:,}."
            )
            while total_tokens < target_tokens:
                shuffled_docs = list(documents)
                random.shuffle(shuffled_docs)
                for doc in shuffled_docs:
                    remaining = target_tokens - total_tokens
                    fitted_doc = doc if len(doc) <= remaining else doc[:remaining]
                    fitted.append(fitted_doc)
                    total_tokens += len(fitted_doc)
                    if total_tokens >= target_tokens:
                        break

        return fitted
    
    def load_wikitext(self) -> List[List[int]]:
        """Load and tokenize WikiText-103 (full train split)."""
        print("\n" + "="*80)
        print("Loading WikiText-103-raw-v1 (full train split)")
        print("="*80)
        
        dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
        
        documents = []
        total_tokens = 0
        
        print(f"Processing {len(dataset):,} examples...")
        
        # Use tqdm with proper configuration to avoid newlines
        progress_bar = tqdm(
            dataset, 
            desc="WikiText-103",
            unit="docs",
            dynamic_ncols=True,
            leave=False,
            position=0
        )
        
        for example in progress_bar:
            text = example['text'].strip()
            
            # Skip empty or very short texts
            if len(text) < 50:
                continue
            
            tokens = self.tokenize_text(text)
            
            # Skip if too few tokens
            if len(tokens) < 10:
                continue
            
            documents.append(tokens)
            total_tokens += len(tokens)
            self.stats['wikitext']['docs'] += 1
            self.stats['wikitext']['tokens'] += len(tokens)
            
            # Update progress bar description with current stats
            progress_bar.set_postfix({
                'docs': len(documents),
                'tokens': f"{total_tokens/1e6:.1f}M"
            })
            
            # Stop if we've reached target
            if total_tokens >= self.target_tokens['wikitext']:
                break
        
        progress_bar.close()
        
        print(f"✅ WikiText-103: {len(documents):,} documents, {total_tokens:,} tokens")
        return documents
    
    def load_ultrachat(self, extra_tokens: int = 0) -> List[List[int]]:
        """
        Load and tokenize UltraChat training tokens plus optional validation reserve.
        
        Args:
            extra_tokens: Additional tokens to load if other sources fail
        """
        target = self.target_tokens['ultrachat'] + extra_tokens
        
        print("\n" + "="*80)
        print(f"Loading stingning/ultrachat (streaming first {target:,} tokens)")
        print("="*80)
        
        # Load in streaming mode for efficiency
        dataset = load_dataset('stingning/ultrachat', split='train', streaming=True)
        
        documents = []
        total_tokens = 0
        
        print("Processing UltraChat conversations...")
        
        # Use tqdm with proper configuration
        progress_bar = tqdm(
            dataset, 
            desc="UltraChat",
            total=150000,  # Estimate
            unit="convs",
            dynamic_ncols=True,
            leave=False,
            position=0
        )
        
        for example in progress_bar:
            try:
                # UltraChat format: {'data': [user_msg, assistant_msg, ...]}
                if 'data' not in example or len(example['data']) < 2:
                    continue
                
                conversation_data = example['data']
                
                # Process conversation pairs
                for i in range(0, len(conversation_data) - 1, 2):
                    user_msg = conversation_data[i]
                    assistant_msg = conversation_data[i + 1]
                    
                    # Format as User/Assistant dialogue
                    formatted = self.format_conversation(user_msg, assistant_msg)
                    tokens = self.tokenize_text(formatted)
                    
                    if len(tokens) < 10:
                        continue
                    
                    documents.append(tokens)
                    total_tokens += len(tokens)
                    self.stats['ultrachat']['docs'] += 1
                    self.stats['ultrachat']['tokens'] += len(tokens)
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'docs': len(documents),
                        'tokens': f"{total_tokens/1e6:.1f}M"
                    })
                    
                    # Stop if we've reached target
                    if total_tokens >= target:
                        progress_bar.close()
                        print(f"\n✅ Reached target: {total_tokens:,} tokens")
                        return documents
            
            except Exception as e:
                # Skip malformed examples
                continue
        
        progress_bar.close()
        
        print(f"✅ UltraChat: {len(documents):,} documents, {total_tokens:,} tokens")
        return documents
    
    def shuffle_and_pack(self, documents: List[List[int]], output_file: str):
        """
        Shuffle documents and pack into binary format using memmap.
        
        Args:
            documents: List of tokenized documents
            output_file: Output binary file path
        """
        print(f"\n{'='*80}")
        print(f"Shuffling and packing {len(documents):,} documents")
        print(f"{'='*80}")
        
        # Shuffle documents
        print("Shuffling documents...")
        random.shuffle(documents)
        
        # Calculate total tokens
        total_tokens = sum(len(doc) for doc in documents)
        print(f"Total tokens: {total_tokens:,}")
        
        # Create memory-mapped array
        output_path = self.output_dir / output_file
        print(f"Creating memory-mapped file: {output_path}")
        
        # Use memmap for memory efficiency
        mmap = np.memmap(output_path, dtype=np.uint32, mode='w+', shape=(total_tokens,))
        
        # Pack tokens
        print("Packing tokens...")
        offset = 0
        for doc in tqdm(documents, desc="Packing"):
            doc_len = len(doc)
            mmap[offset:offset + doc_len] = doc
            offset += doc_len
        
        # Flush to disk
        mmap.flush()
        del mmap
        
        print(f"✅ Saved: {output_path}")
        print(f"   Size: {output_path.stat().st_size / 1e9:.2f} GB")
        print(f"   Tokens: {total_tokens:,}")
    
    def build_dataset(self):
        """Build the complete 80/20 WikiText/UltraChat dataset."""
        print("\n" + "="*80)
        print("BUILDING CONVERSATIONAL SCHOLAR DATASET (300M TOKENS)")
        print("="*80)
        
        # Load all sources
        print("\n📚 Loading data sources...")
        
        wikitext_docs = self.fit_documents_to_target(
            self.load_wikitext(),
            self.target_tokens['wikitext'],
            "WikiText/factual",
        )
        
        print("\n" + "="*80)
        print("Loading UltraChat (with extra for validation)")
        print("="*80)
        ultrachat_docs = self.load_ultrachat(extra_tokens=self.val_target)
        
        # Combine training documents
        print("\n" + "="*80)
        print("Combining training documents")
        print("="*80)
        
        # Reserve last portion of UltraChat for validation
        val_tokens_needed = self.val_target
        val_docs = []
        val_tokens = 0
        train_ultrachat = []
        
        print(f"Reserving {val_tokens_needed:,} tokens from UltraChat for validation...")
        
        # Take from end of UltraChat for validation
        for doc in reversed(ultrachat_docs):
            if val_tokens >= val_tokens_needed:
                train_ultrachat.insert(0, doc)
            else:
                val_docs.insert(0, doc)
                val_tokens += len(doc)
        
        print(f"✅ Validation: {len(val_docs):,} documents, {val_tokens:,} tokens")
        print(f"✅ Training UltraChat: {len(train_ultrachat):,} documents")

        train_ultrachat = self.fit_documents_to_target(
            train_ultrachat,
            self.target_tokens['ultrachat'],
            "UltraChat",
        )
        val_docs = self.fit_documents_to_target(
            val_docs,
            self.val_target,
            "UltraChat validation",
        )

        self.stats = {
            'wikitext': {
                'docs': len(wikitext_docs),
                'tokens': self.count_tokens(wikitext_docs),
            },
            'ultrachat': {
                'docs': len(train_ultrachat),
                'tokens': self.count_tokens(train_ultrachat),
            },
        }
        
        # Combine all training documents
        all_train_docs = wikitext_docs + train_ultrachat
        
        print(f"Total training documents: {len(all_train_docs):,}")
        print("Final training mix:")
        print(f"  WikiText/factual: {self.stats['wikitext']['tokens']:,} tokens (80.0%)")
        print(f"  UltraChat:        {self.stats['ultrachat']['tokens']:,} tokens (20.0%)")

        print("\n" + "="*80)
        print("Packing curriculum source files")
        print("="*80)
        self.shuffle_and_pack(list(wikitext_docs), 'wikitext_train.bin')
        self.shuffle_and_pack(list(train_ultrachat), 'ultrachat_train.bin')
        
        # Pack training data
        self.shuffle_and_pack(list(all_train_docs), 'train.bin')
        
        # Pack validation data
        print("\n" + "="*80)
        print("Packing validation data (UltraChat conversational subset)")
        print("="*80)
        self.shuffle_and_pack(val_docs, 'val.bin')
        
        # Save statistics
        self.save_statistics()
        
        print("\n" + "="*80)
        print("✅ DATASET BUILD COMPLETE!")
        print("="*80)
        self.print_summary()
    
    def save_statistics(self):
        """Save dataset statistics to JSON."""
        stats_file = self.output_dir / 'dataset_stats.json'
        
        # Calculate totals
        total_docs = sum(s['docs'] for s in self.stats.values())
        total_tokens = sum(s['tokens'] for s in self.stats.values())
        
        stats_output = {
            'sources': self.stats,
            'totals': {
                'documents': total_docs,
                'tokens': total_tokens,
            },
            'targets': self.target_tokens,
            'tokenizer': 'tiktoken cl100k_base (GPT-4)',
            'vocab_size': self.vocab_size,
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_output, f, indent=2)
        
        print(f"\n✅ Statistics saved: {stats_file}")
    
    def print_summary(self):
        """Print final summary."""
        print("\n📊 Dataset Statistics:")
        print("-" * 80)
        
        total_docs = sum(s['docs'] for s in self.stats.values())
        total_tokens = sum(s['tokens'] for s in self.stats.values())
        
        for source, stats in self.stats.items():
            docs = stats['docs']
            tokens = stats['tokens']
            pct = (tokens / total_tokens * 100) if total_tokens > 0 else 0
            print(f"{source:15s}: {docs:8,} docs | {tokens:12,} tokens ({pct:5.1f}%)")
        
        print("-" * 80)
        print(f"{'TOTAL':15s}: {total_docs:8,} docs | {total_tokens:12,} tokens (100.0%)")
        print("-" * 80)
        
        print("\n📁 Output Files:")
        print(f"  Training:   {self.output_dir / 'train.bin'}")
        print(f"  Validation: {self.output_dir / 'val.bin'}")
        print(f"  Statistics: {self.output_dir / 'dataset_stats.json'}")
        
        print("\n🎯 Training Configuration:")
        print(f"  Total tokens: 300M")
        print(f"  Composition:")
        print(f"    - WikiText/factual: 240M (80%) - Knowledge")
        print(f"    - UltraChat: 60M (20%) - Conversation")
        print(f"  Epochs: 8")
        print(f"  Total training tokens: 2.4B (300M x 8)")
        print(f"  Tokens per step: 65,536 (batch_size × context_length × grad_accum)")
        print(f"  Max steps: 36,621 (2.4B / 65,536)")
        
        print("\n💡 Next Steps:")
        print("  1. Update config to use this dataset:")
        print("     data:")
        print("       train_file: 'data/balanced_300m/train.bin'")
        print("       val_file: 'data/balanced_300m/val.bin'")
        print("       dataset_type: 'binary'")
        print("  2. Start training:")
        print("     python train.py --config config/auto_training_200m_modern.yaml")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Prepare 300M token Conversational Scholar dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/balanced_300m',
        help='Output directory for binary files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Build dataset
    builder = BalancedDatasetBuilder(
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    builder.build_dataset()


if __name__ == '__main__':
    main()
