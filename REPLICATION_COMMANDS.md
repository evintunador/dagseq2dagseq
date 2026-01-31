# Replication Commands

## Wikipedia Dataset

```bash
# Extract from dump
python -m data.wiki_graph_extractor.dump_extractor \
  /Users/evintunador/Desktop/wiki_dumps/simplewiki-20251027-cirrussearch-content.json.gz \
  -o demo_datasets/wiki/extracted \
  --limit 100

# Build graph
python -m data.extractors.cli wiki \
  demo_datasets/wiki/extracted \
  -o demo_datasets/wiki/wiki_graph.jsonl

# Pretokenize
python -m data.pretokenize \
  demo_datasets/wiki/extracted \
  demo_datasets/wiki/wiki_graph.jsonl \
  -o demo_datasets/wiki/runs \
  --dataset-name "Wikipedia Sample" \
  --source-type markdown \
  -p 4

# Visualize
python -m model.graph_traversal.block_mask_creator \
  demo_datasets/wiki/runs \
  --mask-type cross_doc_link \
  --strategy random_walk \
  --token-budget 4096 \
  --seed 10
```

## GitHub Dataset

```bash
# Download sample
python -m data.github_graph_extractor.download_sample \
  -o demo_datasets/github/sample.jsonl \
  --limit 50000

# Build graph
python -m data.extractors.cli github \
  demo_datasets/github/sample.jsonl \
  -o demo_datasets/github/github_graph.jsonl

# Pretokenize (using ContentSource directly - no intermediate markdown!)
python -m data.pretokenize \
  demo_datasets/github/github_graph.jsonl \
  -o demo_datasets/github/runs \
  --dataset-name "GitHub Sample" \
  --source-type github \
  --input-file demo_datasets/github/sample.jsonl \
  -p 4

# Visualize
python -m model.graph_traversal.block_mask_creator \
  demo_datasets/github/runs \
  --mask-type cross_doc_link \
  --strategy bfs \
  --token-budget 8192 \
  --seed 4
```

## Output Locations

- Wikipedia (3 matched links): `model/graph_traversal/artifacts/mask_viz_cross_doc_link_seed10.png`
- GitHub (47 matched links): `model/graph_traversal/artifacts/mask_viz_cross_doc_link_seed4.png`

## Notes

- No intermediate markdown generation needed! The `ContentSource` abstraction handles reading from different formats.
- GitHub uses `--source-type jsonl` to read directly from the sample file
- The graph stores `source_identifier` metadata to map back to original documents
- To add a new dataset type (e.g., LaTeX), just implement a new `ContentSource`
