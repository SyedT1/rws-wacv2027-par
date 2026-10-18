# EVA02-L/14 domain-balanced sampling

Open [upar-track1-eva02-l14-domain-balanced.ipynb](upar-track1-eva02-l14-domain-balanced.ipynb) in Kaggle, enable **GPU and Internet**, and run all cells.

This uses the strongest recorded single backbone, EVA02-L/14 (hidden-test Avg 0.7825), with its global classifier. The existing EVA02-L + SigLIP 2 ensemble scores 0.7856; this notebook trains one model that could later be evaluated as an ensemble member. No new score is claimed.

## Google Drive quota / reusable data

`FileURLRetrievalError: Too many users have viewed or downloaded this file recently` is a quota block on the public Google Drive file, before training. The frozen-module debugger warnings are unrelated.

To continue while the host is blocked:

1. In Kaggle, use **Add Input** to attach a previous successful notebook's saved data or a dataset containing the original source images/ZIPs. A submission ZIP containing only weights is insufficient.
2. Keep the prepared `Market1501/`, `PA100k/`, and `PETA/` directory layout, or the original archive layout. Market1501's original `Market-1501-v15.09.15/` layout is supported. Original PETA files are converted using the organizer's filename mapping.
3. Run the updated notebook from the first cell. It scans `/kaggle/input` automatically, copies only missing annotation-referenced images, and skips downloading complete sources. Add any other cache directories to `LOCAL_DATA_ROOTS` in the setup cell. `UPAR_DATA_ROOT`, if set, is also searched as an image/archive cache; training uses the prepared writable `DATA` directory.

Local ZIPs can have different filenames; members are matched by known source paths rather than basename alone. Annotation labels always come from the organizer repository. Cached ZIPs are retained, and image coverage is checked before training. If no complete local copy is available, the public host must become available again; the notebook cannot remove its quota restriction.

The local-reuse helper is embedded in the notebook, so uploading the `.ipynb` alone is sufficient. `data_sources.py` is the readable copy of that helper. Small temporary-file checks run before download attempts and exercise extracted data, ZIP layouts, PETA mapping, repeat execution, and unused unsafe archive members.

## Comparison runs

Run each arm from the same generic pretrained initialization in a fresh kernel. Preserve outputs between runs.

| Setting | Sampling | Replacement | Purpose |
|---|---|---|---|
| `SAMPLING_MODE = "image"` | Uniform images | No | Fresh ordinary-sampling control |
| `SAMPLING_MODE = "mixture"`, `DOMAIN_MIX = 0.5` | Half uniform images, half uniform domains | Yes | Default experiment |
| `SAMPLING_MODE = "image_replacement"` | Uniform images | Yes | Separate replacement from domain reweighting |
| `SAMPLING_MODE = "domain"` | Equal expected domain shares | Yes | Optional stronger balancing |

For image i in source d, the mixture probability is:

```text
p(i) = (1 - DOMAIN_MIX) / N + DOMAIN_MIX / (K * n_d)
```

Counts come from the training CSV. Balancing holds in expectation, not in every minibatch. With replacement, some images repeat and others are omitted during an epoch. Every mode uses `floor(N / BATCH_SIZE)` optimizer steps per epoch, so oversampling does not extend the update budget. At the recorded 97,669 training images and batch 16, that is 6,104 steps and 97,664 draws per epoch.

## Fixed settings

- EVA02-L/14 global head, generic ImageNet-pretrained initialization.
- Direct resize to 224 x 224; original normalization and augmentation.
- Five epochs, batch 16, peak LR 5e-5, layer decay 0.85, weight decay 0.05, drop-path 0.2, one warm-up epoch, seed 42.
- Flip-logit averaging for validation and submission in every arm.
- `LOSS_WEIGHT_MODE = "fixed_image"`: positive weights use the original training-image distribution and remain identical across sampling arms.

The optional `LOSS_WEIGHT_MODE = "sampling"` uses the expected sampled positive prevalence instead. Treat it as a separate sampling-plus-loss experiment. Keep the default fixed weights for the primary comparison.

The historical EVA02-L notebook used single-view validation for checkpoint selection, so its recorded hidden-test score is context rather than a matched control. Compare fresh arms using per-source validation Challenge_Avg. Source-held-out evaluation and another seed are useful follow-ups; no improvement is guaranteed.

## Checks and outputs

Small synthetic checks verify sampling probabilities, equal draw budgets, seeded reproducibility, no replacement for the ordinary control, and positive-weight calculations. A nonzero-head offline/timm equivalence check runs before training. Packaging checks strict loading, finite probability ranges, single/batch agreement, attribute ordering, and offline/timm equivalence.

Outputs under `/kaggle/working` have a prefix containing sampling mode, mixture fraction, loss-weight mode, and `RUN_SUFFIX`:

- `<prefix>_config.json`: settings, versions, annotation hashes, training counts, expected domain shares, and positive weights.
- `<prefix>_history.json`: epoch losses, all seven metrics by source and their equal-source means, realized draw counts, and cumulative optimizer steps.
- `<prefix>_best_val_logits.npz`: selected-epoch flip-averaged logits, labels, paths, sources, attribute order, and epoch.
- `<prefix>_best.pth`: selected checkpoint and experiment configuration.
- `<prefix>_submission.zip`: plain-PyTorch inference, unchanged submission metadata, and checkpoint.

Change `RUN_SUFFIX` before repeating an existing configuration; the notebook stops if its output prefix already exists. The ZIP excludes validation logits, labels, and images. Inference uses neither domain identities nor timm and requires no network access.

The notebook has been inspected locally; training and the PyTorch runtime checks must be executed in Kaggle. No local GPU-training result is available.
