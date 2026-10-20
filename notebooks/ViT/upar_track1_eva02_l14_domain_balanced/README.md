# EVA02-L/14 domain-balanced sampling

Open [upar-track1-eva02-l14-domain-balanced.ipynb](upar-track1-eva02-l14-domain-balanced.ipynb) in Kaggle, enable **GPU and Internet**, and run all cells.

This trains a single EVA02-L/14 backbone with a global 40-attribute classifier and domain-balanced sampling.

## Reported scores

The notebook run achieved the following reported evaluation scores (values on a 0–1 scale):

| Avg | mA | Label_F1 | Inst_Acc | Inst_Prec | Inst_Rec | Inst_F1 |
|---|---|---|---|---|---|---|
| **0.7839** | 0.7875 | 0.6014 | 0.6651 | 0.7469 | 0.8311 | 0.7803 |

For historical context, the EVA02-L/14 baseline recorded hidden-test Avg 0.7825 and the EVA02-L + SigLIP 2 ensemble recorded 0.7856. These are separate runs, not matched sampling controls.

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

## Model and training configuration

The notebook's default configuration is:

| Setting | Value |
|---|---|
| Backbone | `eva02_large_patch14_448.mim_in22k_ft_in22k_in1k` (timm EVA02-L/14) |
| Initialization | ImageNet-22k MIM pretraining, then ImageNet-22k and ImageNet-1k fine-tuning |
| Classifier | Global linear head with 40 outputs; head weights and bias initialized to zero |
| Input | RGB, direct resize to 224 x 224; 14 x 14 patches |
| Normalization mean | `(0.48145466, 0.4578275, 0.40821073)` |
| Normalization standard deviation | `(0.26862954, 0.26130258, 0.27577711)` |
| Training sources | Market1501, PA100K, PETA; organizer training split |
| Sampling | `SAMPLING_MODE = "mixture"`, `DOMAIN_MIX = 0.5`, with replacement |
| Loss | `BCEWithLogitsLoss`; `LOSS_WEIGHT_MODE = "fixed_image"`, positive weights `(negative / positive)` clipped to `[1, 20]` |
| Epochs / batch size | 5 / 16 |
| Optimizer | AdamW; peak head learning rate `5e-5`, layer-wise LR decay `0.85`, weight decay `0.05` (no decay on norms/biases) |
| LR schedule | One epoch of linear warm-up, then cosine decay |
| Regularization | Drop-path `0.2`; gradient norm clipping at `1.0` |
| Training augmentation | Random horizontal flip, edge padding of 10 pixels, random crop to 224 x 224, random erasing with probability `0.25` |
| Precision | CUDA FP16 automatic mixed precision with gradient scaling |
| Checkpoint selection | Best equal-source mean validation `Challenge_Avg` |
| Validation / inference | Average original and horizontally flipped logits, then sigmoid; validation threshold `> 0.5` |
| Inference batch size | 16 |
| Seed | 42 |

Fixed-image positive weights use the original training-image distribution and remain identical across sampling arms.

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

The scores above were supplied from the completed notebook run. Training and the PyTorch runtime checks execute in Kaggle; this README update does not rerun training.
