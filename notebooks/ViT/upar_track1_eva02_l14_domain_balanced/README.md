# EVA02-L/14 domain-balanced sampling

Open [upar-track1-eva02-l14-domain-balanced.ipynb](upar-track1-eva02-l14-domain-balanced.ipynb) in Kaggle, enable **GPU and Internet**, and run all cells.

This trains a single EVA02-L/14 backbone with a global 40-attribute classifier and domain-balanced sampling.

## Reported scores

The user-reported evaluation scores are below (values on a 0–1 scale). These are separate from the validation metrics saved in the executed notebook.

| Avg | mA | Label_F1 | Inst_Acc | Inst_Prec | Inst_Rec | Inst_F1 |
|---|---|---|---|---|---|---|
| **0.7839** | 0.7875 | 0.6014 | 0.6651 | 0.7469 | 0.8311 | 0.7803 |

For historical context, the EVA02-L/14 baseline recorded hidden-test Avg 0.7825 and the EVA02-L + SigLIP 2 ensemble recorded 0.7856. These are separate runs, not matched sampling controls.

## Recorded notebook execution

The saved outputs show a completed five-epoch run with the mixture configuration below, 303.3M model parameters, 97,669 training images, and 33,407 validation images. The run uses `timm==1.0.22` and artifact prefix `upar_track1_eva02_l14_domain_sampling_mixture_mix0.5_fixed_image_v1`.

| Epoch | Train BCE | Validation BCE | Validation mA | Validation Inst_F1 | Validation Challenge_Avg |
|---|---|---|---|---|---|
| 1 | 0.4413 | 0.3320 | 0.8590 | 0.8694 | 0.8637 |
| 2 | 0.1764 | 0.4069 | 0.8724 | 0.8937 | 0.8823 |
| **3 (selected)** | **0.1264** | **0.4657** | **0.8719** | **0.8981** | **0.8842** |
| 4 | 0.1000 | 0.5053 | 0.8666 | 0.9032 | 0.8841 |
| 5 | 0.0886 | 0.5138 | 0.8652 | 0.9042 | 0.8838 |

Validation metrics are equal-source means. At the selected epoch, per-source results and training sampling shares were:

| Source | Training images | Expected sampling share | Validation mA | Validation Inst_F1 | Validation Challenge_Avg |
|---|---|---|---|---|---|
| Market1501 | 10,000 | 21.7860% | 0.8049 | 0.8921 | 0.8462 |
| PA100K | 79,001 | 57.1099% | 0.8825 | 0.8757 | 0.8791 |
| PETA | 8,668 | 21.1041% | 0.9284 | 0.9265 | 0.9274 |

The saved outputs confirm that sampling self-checks, the pre-training offline/timm equivalence check, and packaged inference checks passed. Both recorded offline/timm comparisons had maximum logit difference `0.0`. Packaging produced `upar_track1_eva02_l14_domain_sampling_mixture_mix0.5_fixed_image_v1_submission.zip` (561 MB as printed by the notebook), containing `run.py`, `metadata.yaml`, and the selected checkpoint under `assets/`.

## Data downloads

The executed notebook downloads Market1501 and PA100K from Google Drive and PETA from Dropbox, copies organizer annotations, checks image coverage, and removes the source ZIPs after successful preparation. Its download cells use `/kaggle/working/upar_2027/data` and skip downloads for sources whose required images are already present.

`FileURLRetrievalError: Too many users have viewed or downloaded this file recently` indicates a public Google Drive quota block. If this occurs on a repeat run, retry when the host becomes available or manually prepare the required images in the notebook's data directory. This notebook does not automatically scan `/kaggle/input` or embed the separate `data_sources.py` cache helper.

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

The executed notebook uses the following configuration:

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

The reported evaluation scores were supplied by the user; the validation history and successful runtime checks above were read from the notebook's saved execution outputs. Training was not rerun for this documentation update.
