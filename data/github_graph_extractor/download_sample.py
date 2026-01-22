#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GitHub Data Sampler: Download and sample Python repositories from The Stack dataset.
Streams 10,000 items from the bigcode/the-stack-dedup dataset (Python repositories only).
"""
import argparse
import json
from itertools import islice
from datasets import load_dataset
from tqdm import tqdm

def download_sample(output_path: str, limit: int = 10_000):
    """
    Download a sample of Python repositories from The Stack dataset.

    Args:
        output_path: Path to save the sampled data as JSONL
        limit: Number of items to sample (default: 10,000)
    """
    print(f"Loading bigcode/the-stack-dedup dataset (Python repositories only)...")

    # Stream the dataset - only Python repositories
    ds = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/python",
        split="train",
        streaming=True,
        token=True,  # uses HF auth token if available
    )

    print(f"Sampling {limit} items from the dataset...")
    sampled_items = []

    # Use tqdm for progress tracking
    with tqdm(total=limit, desc="Sampling repositories") as pbar:
        for item in islice(ds, limit):
            sampled_items.append(item)
            pbar.update(1)

    print(f"Saving {len(sampled_items)} items to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for item in sampled_items:
            # Optional: truncate very large content to prevent gigantic files
            if "content" in item and len(item["content"]) > 50000:  # 50KB limit
                item["content"] = item["content"][:50000] + "...[truncated]"
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Successfully saved {len(sampled_items)} repositories to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        prog="GitHubDataSampler",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file for the sampled data (default: auto-generated based on limit: sample_10k.jsonl, sample_100k.jsonl, sample_1M.jsonl, or sample_10M.jsonl)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100_000,
        help="Number of items to sample (default: 100,000)"
    )

    args = parser.parse_args()

    # Auto-generate output filename if not specified
    if args.output is None:
        if args.limit >= 10000000:
            args.output = "sample_10M.jsonl"
        elif args.limit >= 1000000:
            args.output = "sample_1M.jsonl"
        elif args.limit >= 100000:
            args.output = "sample_100k.jsonl"
        elif args.limit >= 10000:
            args.output = "sample_10k.jsonl"
        else:
            args.output = f"sample_{args.limit}.jsonl"

    download_sample(args.output, args.limit)

if __name__ == '__main__':
    main()
