# Text-Attributed Graph Sequence to Text-Attributed Graph Sequence (TAGSeq2TAGSeq)

This experiment explores training language models on graph-structured data. It implements custom data loading pipelines that traverse the text-attributed document graph to construct packed sequences.

## Usage

### Inspecting Data

To see how the graph traversal works and what the packed batches look like:

```bash
python experiments/dagseq2dagseq/demo_traversal.py /path/to/pretokenized/dataset --strategy random_walk
```

### Training

To run the training script:

```bash
# Using default config
python experiments/dagseq2dagseq/main.py --dataset-dir data/pretokenized_datasets/simplewiki_full

# With custom settings
python experiments/dagseq2dagseq/main.py \
  --dataset-dir data/pretokenized_datasets/simplewiki_full \
  --strategy random_walk \
  --max-seq-len 4096 \
  --seed 42
```
