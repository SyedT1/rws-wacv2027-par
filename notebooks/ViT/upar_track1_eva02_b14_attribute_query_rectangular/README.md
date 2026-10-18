# Experiment 2: rectangular EVA02-B attribute queries

Open [upar-track1-eva02-b14-attribute-query-rectangular.ipynb](upar-track1-eva02-b14-attribute-query-rectangular.ipynb) in Kaggle with **GPU and Internet enabled**, then run all cells.

This tests tall pedestrian inputs after the [square attribute-query experiment](../upar_track1_eva02_b14_attribute_query/upar-track1-eva02-b14-attribute-query.ipynb). It trains from generic pretrained EVA02 weights using the provided training split.

## Reported hidden-test result

| Avg | mA | Label_F1 | Inst_Acc | Inst_Prec | Inst_Rec | Inst_F1 |
|---|---|---|---|---|---|---|
| 0.7739 | 0.7814 | 0.5906 | 0.6469 | 0.7229 | 0.8320 | 0.7666 |

Reported on October 18, 2026. Avg improves by 0.0025 over the square query run (0.7714) and by 0.0036 over the historical global baseline (0.7703). Geometry and token count change together, and these are single-run comparisons, so this does not isolate an aspect-ratio effect. The selected checkpoint epoch was not reported; retain the saved configuration and submission report to confirm the exact run settings.

## Default run

- Image height: **280**; width: **140**.
- Patch grid: **20 × 10**, giving **200 patch tokens**.
- Head: equally averaged global and attribute-query logits.
- Decoder: 40 queries, width 256, eight heads, one layer.
- Training: eight epochs, microbatch 32, two accumulation steps, effective batch 64.
- Loss, learning rate, layer decay, augmentation operations, and seed match the square query experiment.
- Validation and submission use horizontal-flip logit averaging.

The earlier recommendation incorrectly counted 280 × 140 as 256 patch tokens. Its correct count is 200. This changes geometry and token count together; it does not isolate shape at identical compute. Direct resizing also does not preserve every original crop's exact aspect ratio.

## Comparison runs

Set `IMAGE_HEIGHT`, `IMAGE_WIDTH`, and the same `HEAD_MODE` as the square comparison. Restart and run all cells after changing configuration.

| Height × width | Grid | Patch tokens | Purpose |
|---|---|---:|---|
| 224 × 224 | 16 × 16 | 256 | Square control |
| 280 × 140 | 20 × 10 | 200 | Default tall-input experiment |
| 308 × 168 | 22 × 12 | 264 | Tall input with a closer token budget |
| 336 × 224 | 24 × 16 | 384 | Subsequent resolution experiment |

Run the square control and tall inputs with the same head mode, data, optimizer settings, effective batch, and seed. Select configurations using development data and compare per-source Challenge_Avg. Source-held-out validation is still needed to assess unseen-domain generalization. The global branch within a hybrid run is jointly trained and is not an independent global-only baseline.

Both dimensions must be positive multiples of 14. If memory is tight, use `BATCH_SIZE = 16` and `ACCUMULATION_STEPS = 4`. Lower `INFERENCE_BATCH_SIZE` separately if needed.

## Rectangular inference support

timm interpolates pretrained absolute positions to the target grid during loading. Fine-tuned absolute positions are saved in the checkpoint. The offline encoder reads explicit height/width metadata and reconstructs axis-specific rotary positions against the reference 16 × 16 grid. Grid shape is never inferred with a square root of token count. Input and checkpoint shape assertions catch reversed or inconsistent dimensions.

The notebook includes a CPU check against pinned timm rotary embeddings for square and rectangular grids. Before training, it checks full offline/timm equivalence using nonzero classifiers. After training, packaging checks real-image logits, finite probabilities, single/batch consistency, and attribute order. These numerical checks execute when the notebook is run; no local training result is claimed.

## Outputs

The default prefix is `upar_track1_eva02_b14_attribute_query_280x140_hybrid`. `/kaggle/working` receives:

- `<prefix>_best.pth`
- `<prefix>_submission.zip`
- `<prefix>_config.json`
- `<prefix>_history.json`
- `<prefix>_best_val_logits.npz`

Geometry and head mode are included in names to keep square and rectangular runs separate. Preserve outputs before rerunning an identical geometry/head combination with changed settings. Only the offline runtime, unchanged submission metadata, and selected model checkpoint enter the ZIP. No timm dependency, validation labels, or training images are shipped.
