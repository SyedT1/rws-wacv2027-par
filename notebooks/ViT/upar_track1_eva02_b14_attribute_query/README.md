# EVA02-B/14 attribute-query experiment

Open [upar-track1-eva02-b14-attribute-query.ipynb](upar-track1-eva02-b14-attribute-query.ipynb) in Kaggle, enable **GPU** and **Internet**, and run all cells. The notebook downloads the organizer's development data and generic pretrained EVA02 weights, trains on the provided training split, evaluates on validation, and builds an offline Track 1 submission ZIP.

The default model combines the original global classifier with 40 learned attribute queries. Each query attends to patch tokens through one Transformer decoder layer (width 256, eight heads). The encoder is fine-tuned throughout. Inputs, weighted BCE, augmentation, eight epochs, peak LR, layer decay, and seed follow the EVA02-B baseline. Microbatch 32 with two accumulation steps retains effective batch 64. Validation and submission both average original/flipped logits.

## Run settings

- Start with `HEAD_MODE = "hybrid"` for the proposed model.
- For a controlled comparison, also train `HEAD_MODE = "global"` with the same settings. `"query"` is the optional third ablation. Each mode writes a different output prefix.
- If GPU memory is insufficient, set `BATCH_SIZE = 16` and `ACCUMULATION_STEPS = 4`; keep the effective batch at 64. Lower `INFERENCE_BATCH_SIZE` separately if validation exceeds memory.
- Test `DECODER_LAYERS = 2` only as a separate follow-up run. Preserve outputs before rerunning the same mode with changed settings.
- `USE_IMAGENET_WEIGHTS = False` trains from scratch and is not a pretrained-baseline comparison.

The self-check cell tests query-head gradients. Before training, the notebook compares timm with the plain-PyTorch offline encoder, query head, and fused predictions using nonzero readouts. Packaging checks the trained implementation again, including attribute-order and single/batch consistency. A failed check stops the run.

## Outputs

For the default hybrid run, `/kaggle/working` contains:

- `upar_track1_eva02_b14_attribute_query_hybrid_best.pth`
- `upar_track1_eva02_b14_attribute_query_hybrid_submission.zip`
- `upar_track1_eva02_b14_attribute_query_hybrid_config.json`
- `upar_track1_eva02_b14_attribute_query_hybrid_history.json`
- `upar_track1_eva02_b14_attribute_query_hybrid_best_val_logits.npz`

The checkpoint is selected by mean per-source validation Challenge_Avg with flip TTA. History also records per-source and jointly trained branch diagnostics; the global branch within a hybrid run is not an independently trained baseline. The NPZ contains best-epoch combined/global/query logits as available, labels, paths, source names, and attribute order. It supports later local calibration experiments.

Only `run.py`, metadata, and the selected checkpoint enter the ZIP. Inference needs PyTorch, torchvision, numpy, and Pillow from the evaluation image; it does not import timm or access the network. The notebook pins training timm to 1.0.22 and records versions, annotation hashes, and the organizer repository revision.

## Interpretation

This experiment has no measured result yet. Compare a fresh global control and the query variants with the same evaluator. Historical EVA02-B hidden-test Avg 0.7703 is context; its checkpoint selection did not use the new flip-aware validation procedure. Additional source-held-out experiments are needed to assess domain generalization.

Inspired by [Query2Label](https://github.com/SlongLiu/query2labels) and [ML-Decoder](https://github.com/Alibaba-MIIL/ML_Decoder); this notebook is a small adaptation, not a complete reproduction of either method.
