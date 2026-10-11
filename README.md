# RWS2027 – UPAR Challenge Track 1: Pedestrian Attribute Recognition @ WACV2027

Real-World Surveillance: Applications and Challenges, 7th @ WACV2027

- **Organized by:** spcke (andreas.specker@iosb.fraunhofer.de)
- **Platform:** Codabench
- **Current phase ends:** October 22, 2026 at 6:00 PM GMT+6
- **Docker image:** `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`

## News

- **Oct 6, 2026:** We have clarified the rules. Please review them to ensure compliance.
- **Oct 2, 2026:** New code & fact sheet submission deadline: Oct 26, 2026, AoE. Due to strict WACV deadlines, we must move up the deadline.
- **Sep 30, 2026:** Track 3 on Human Pose Estimation is now live.

## Challenge & Workshop

Track 1 of the UPAR Challenge at the Real-World Surveillance workshop (RWS), WACV 2027.

Everything about this track lives on the Codabench page: task, data, evaluation, submission format, schedule and rules. Announcements go out by email to registered participants.

The RWS workshop welcomes both regular submissions on the topic of real-world surveillance and submissions as part of the Challenge:

- RWS Workshop
- Challenge Task 1: Pedestrian Attribute Recognition
- Challenge Task 2: Attribute-Based Person Retrieval
- Challenge Task 3: Human Pose Estimation

## The Task

Pedestrian attribute recognition under domain shift. For every pedestrian crop predict **40 binary attributes**: age, gender, hair length, upper and lower body clothing length, colour and type, and accessories.

The challenge builds on the UPAR dataset, which unifies the annotations of several public pedestrian datasets. The test images come from a domain your model has never seen, so the method has to generalise beyond its training data.

You submit **code, not predictions**. Your model runs inside the evaluation container on a hidden test set, and only the resulting metrics are published.

## Getting Started

1. Read the Data page and download the training data and example submission (Starter Kit).
2. Take the example submission as your template. It shows the functions the platform calls.
3. Put your model and weights into the submission folder, zip the contents of the folder so `run.py` is at the archive root, and upload it under My Submissions.
4. See the Submission section for the API your `run.py` must expose, and the Evaluation section for the metrics.

Submissions run in `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime` (Python 3.11, torch 2.4.1, torchvision, numpy, Pillow) with **no network access**. Ship everything your model needs inside the zip.

## Our Baselines: ImageNet Vision Transformers

The [`notebooks/`](notebooks) folder holds Kaggle notebooks that download the data, train a model, and package a Track 1 submission ZIP. Candidate backbones for later experiments are listed in [`models_to_use.txt`](models_to_use.txt).

| Notebook | Backbone | Params | Pretrained weights | ImageNet-1K top-1 | Epochs | Batch | Peak LR $`\eta_{\max}`$ | Layer decay $`\lambda`$ | ZIP size |
|---|---|---|---|---|---|---|---|---|---|
| [`upar-track1-vit-b16.ipynb`](notebooks/ViT/upar_track1_vit_b16/upar-track1-vit-b16.ipynb) | ViT-B/16 (12 blocks, width 768) | 86.6M | `torchvision` `ViT_B_16_Weights.IMAGENET1K_V1` | 81.07% | 8 | 64 | $`10^{-4}`$ | 0.75 | 159 MB |
| [`upar-track1-vit-l16.ipynb`](notebooks/ViT/upar_track1_vit_l16/upar-track1-vit-l16.ipynb) | ViT-L/16 (24 blocks, width 1024) | 304.3M | `torchvision` `ViT_L_16_Weights.IMAGENET1K_V1` | 79.66% | 5 | 32 | $`5 \cdot 10^{-5}`$ | 0.85 | 553 MB |
| [`upar-track1-deit3-b16.ipynb`](notebooks/ViT/upar_track1_deit3_b16/upar-track1-deit3-b16.ipynb) | DeiT III-B/16 (12 blocks, width 768) | 86.6M | `timm` `deit3_base_patch16_224.fb_in22k_ft_in1k` | 85.7% | 8 | 64 | $`10^{-4}`$ | 0.75 | ~159 MB |
| [`upar-track1-deit3-l16.ipynb`](notebooks/ViT/upar_track1_deit3_l16/upar-track1-deit3-l16.ipynb) | DeiT III-L/16 (24 blocks, width 1024) | 304.4M | `timm` `deit3_large_patch16_224.fb_in22k_ft_in1k` | 87.0% | 5 | 32 | $`5 \cdot 10^{-5}`$ | 0.85 | ~559 MB |
| [`upar-track1-eva02-b14.ipynb`](notebooks/ViT/upar_track1_eva02_b14/upar-track1-eva02-b14.ipynb) | EVA02-B/14 (12 blocks, width 768) | 87.1M | `timm` `eva02_base_patch14_448.mim_in22k_ft_in22k_in1k` | 88.7% (448 px) | 8 | 64 | $`10^{-4}`$ | 0.75 | ~159 MB |
| [`upar-track1-eva02-l14.ipynb`](notebooks/ViT/upar_track1_eva02_l14/upar-track1-eva02-l14.ipynb) | EVA02-L/14 (24 blocks, width 1024) | 305.1M | `timm` `eva02_large_patch14_448.mim_in22k_ft_in22k_in1k` | 89.9% (448 px) | 5 | 16 | $`5 \cdot 10^{-5}`$ | 0.85 | ~559 MB |
| [`upar-track1-siglip2-b16.ipynb`](notebooks/CLIP-Style/upar_track1_siglip2_b16/upar-track1-siglip2-b16.ipynb) | SigLIP 2 ViT-B/16 (12 blocks, width 768, attention pooling) | 92.9M | `timm` `vit_base_patch16_siglip_224.v2_webli` | n/a (contrastive, WebLI) | 8 | 64 | $`10^{-4}`$ | 0.75 | 172 MB |
| [`upar-track1-siglip2-so400m14.ipynb`](notebooks/CLIP-Style/upar_track1_siglip2_so400m14/upar-track1-siglip2-so400m14.ipynb) | SigLIP 2 So400m/14 (27 blocks, width 1152, attention pooling) | 427.7M | `timm` `vit_so400m_patch14_siglip_224.v2_webli` | n/a (contrastive, WebLI) | 5 | 16 | $`5 \cdot 10^-5`$ | 0.85 | 791 MB |
| [`upar-track1-dinov3-b16.ipynb`](notebooks/Self-supervised/upar_track1_dinov3_b16/upar-track1-dinov3-b16.ipynb) | DINOv3 ViT-B/16 (12 blocks, width 768, RoPE, 4 register tokens, LayerScale) | 85.7M | `timm` `vit_base_patch16_dinov3.lvd1689m` | n/a (self-supervised, LVD-1689M) | 8 | 64 | $`10^{-4}`$ | 0.75 | 159 MB |

The notebooks follow the structure and download cells of the organizers' Kaggle example notebook, which was written for Track 2 (attribute-based retrieval), but they package a **Track 1** submission. Both ViT checkpoints are trained on ImageNet-1K only, which the rules permit, and `torchvision` is already in the evaluation image, so nothing besides the weights needs to ship in the ZIP. The DeiT III checkpoints are pretrained on ImageNet-22k and fine-tuned on ImageNet-1K, and timm downloads them from the Hugging Face Hub. timm is not in the evaluation image, so the DeiT III `run.py` defines the network in plain PyTorch with timm's parameter names, and the packaging step checks that it gives the same logits as the timm model. DeiT III training also uses stochastic depth (drop-path rate 0.1 for B, 0.2 for L). The EVA02 checkpoints are pretrained on ImageNet-22k with masked image modeling (EVA-CLIP features as the target) and fine-tuned on ImageNet-22k and ImageNet-1K at 448 px. The notebooks train them at 224 px with the CLIP mean and standard deviation, the same drop-path rates as DeiT III, and the same plain-PyTorch `run.py` check. The SigLIP 2 checkpoint is the image encoder of Google's SigLIP 2 (Apache-2.0), pretrained on WebLI image-text pairs with a sigmoid contrastive loss, a captioning decoder, self-distillation and masked prediction. It is a contrastive model, so it has no ImageNet-1K top-1. The notebook trains it at its native 224 px resolution with mean and standard deviation 0.5 (pixels scaled to $`[-1, 1]`$), drop-path rate 0.1, and the DeiT III-B hyperparameters; the packaged `run.py` again defines the network in plain PyTorch and checks that it matches timm (max logit difference 0). The SigLIP 2 So400m/14 notebook uses the same recipe with the larger-model hyperparameters of DeiT III-L (5 epochs, batch 16, peak LR $`5 \cdot 10^{-5}`$, layer decay 0.85, drop-path 0.2); its 14 px patches give $16 \times 16$ tokens at 224 px. EVA02-L/14 (24 blocks, width 1024, 305.1M parameters) uses the DeiT III-L hyperparameters: 224 px input ($16 \times 16$ tokens), 5 epochs, batch size 16 so that training fits in the 16 GB of a T4, peak LR $`5 \cdot 10^{-5}`$, layer decay 0.85, drop-path 0.2, weight decay 0.05, 1 warm-up epoch and seed 42. Submission ZIPs exceed GitHub's 100 MB file limit, so they are stored with Git LFS (`*.zip` in `.gitattributes`); run `git lfs pull` after cloning to download them. The DINOv3 ViT-B/16 checkpoint is Meta's self-supervised model (DINOv3 licence), pretrained on LVD-1689M without labels using self-distillation with Gram anchoring. It has no ImageNet-1K top-1 here, because only the backbone is released with this tag. The notebook fine-tunes it at 224 px ($14 \times 14$ patches) with the ImageNet mean and standard deviation, class-token pooling, drop-path rate 0.1, and the DeiT III-B hyperparameters. Its positions are encoded with 2D rotary embeddings (RoPE) instead of learned position embeddings, so the network has no position-embedding parameters; the packaged `run.py` defines it in plain PyTorch and checks that it matches timm (max logit difference $2.3 \cdot 10^{-6}$). The learning rate warms up linearly over all 8 epochs (`WARMUP_EPOCHS = 8`), so it peaks only at the end of training.

### Results on the provided validation split

Each model was trained once on Kaggle (seed 42). Each value is the mean over the Market1501, PA-100K and PETA parts of the validation split. The best value in each column is in bold, and the bold epoch is the packaged checkpoint.

**ViT-B/16** (8 epochs, packaged checkpoint from epoch 6):

| Epoch | Train BCE | Val BCE | mA | Instance F1 | Challenge_Avg |
|---|---|---|---|---|---|
| 1 | 0.6099 | 0.4596 | 0.7931 | 0.7361 | 0.7632 |
| 2 | 0.3059 | 0.3790 | 0.8297 | 0.7946 | 0.8113 |
| 3 | 0.2322 | **0.3636** | 0.8341 | 0.8206 | 0.8268 |
| 4 | 0.1901 | 0.3920 | **0.8396** | 0.8351 | 0.8367 |
| 5 | 0.1584 | 0.4287 | 0.8295 | 0.8438 | 0.8360 |
| **6** | 0.1359 | 0.4495 | 0.8327 | 0.8500 | **0.8408** |
| 7 | 0.1205 | 0.4740 | 0.8250 | 0.8541 | 0.8389 |
| 8 | **0.1130** | 0.4796 | 0.8237 | **0.8546** | 0.8385 |

Validation BCE is lowest at epoch 3 and rises afterwards while Instance F1 keeps improving, so the model becomes over-confident as it fits the training data. mA peaks at epoch 4.

**ViT-L/16** (5 epochs, packaged checkpoint from epoch 4):

| Epoch | Train BCE | Val BCE | mA | Instance F1 | Challenge_Avg |
|---|---|---|---|---|---|
| 1 | 0.6093 | 0.4569 | 0.7867 | 0.7601 | 0.7729 |
| 2 | 0.2829 | **0.3860** | 0.8190 | 0.8112 | 0.8147 |
| 3 | 0.1918 | 0.4081 | **0.8257** | 0.8262 | 0.8255 |
| **4** | 0.1404 | 0.4708 | 0.8183 | 0.8414 | **0.8293** |
| 5 | **0.1152** | 0.5058 | 0.8105 | **0.8453** | 0.8272 |

ViT-L/16 shows the same pattern sooner: validation BCE is lowest at epoch 2 and mA peaks at epoch 3. Its best Challenge_Avg (0.8293) is below ViT-B/16's (0.8408), even though it has 3.5 times as many parameters. Its torchvision weights are also weaker on ImageNet (79.66% vs 81.07% top-1).

Both runs averaged per-image F1 scores to get Instance F1. The notebooks now compute it from the image-averaged precision and recall, as the evaluation script does (see [Evaluation](#evaluation)), so later runs can show slightly different Instance F1 and Challenge_Avg values. Both DeiT III models, EVA02-B/14, SigLIP 2 ViT-B/16 and SigLIP 2 So400m/14 have been trained and scored (see below); DINOv3 ViT-B/16 and EVA02-L/14 have also been trained and scored (see below).

**SigLIP 2 ViT-B/16** (8 epochs, packaged checkpoint from epoch 7):

| Epoch | Train BCE | Val BCE | mA | Instance F1 | Challenge_Avg |
|---|---|---|---|---|---|
| 1 | 0.6430 | 0.4625 | 0.7978 | 0.7565 | 0.7762 |
| 2 | 0.3137 | 0.3732 | 0.8410 | 0.8206 | 0.8302 |
| 3 | 0.2465 | **0.3518** | 0.8489 | 0.8329 | 0.8403 |
| 4 | 0.2102 | 0.3557 | 0.8541 | 0.8531 | 0.8530 |
| 5 | 0.1836 | 0.3544 | **0.8596** | 0.8610 | 0.8598 |
| 6 | 0.1613 | 0.3778 | 0.8585 | 0.8679 | 0.8627 |
| **7** | 0.1473 | 0.3978 | 0.8595 | 0.8730 | **0.8656** |
| 8 | **0.1393** | 0.4092 | 0.8568 | **0.8743** | 0.8649 |

SigLIP 2 reaches a validation Challenge_Avg of 0.8656, above ViT-B/16 (0.8408) and ViT-L/16 (0.8293). Validation BCE is lowest at epoch 3 and mA peaks at epoch 5, while Instance F1 keeps rising until epoch 8, the same over-confidence pattern as the other models. Its Instance F1 is computed from the image-averaged precision and recall, as in the evaluation script, so it is not directly comparable with the older ViT-B/16 and ViT-L/16 values.

**SigLIP 2 So400m/14** (5 epochs, packaged checkpoint from epoch 5):

| Epoch | Train BCE | Val BCE | mA | Instance F1 | Challenge_Avg |
|---|---|---|---|---|---|
| 1 | 0.5594 | 0.3834 | 0.8346 | 0.8028 | 0.8176 |
| 2 | 0.2732 | 0.3319 | 0.8578 | 0.8591 | 0.8580 |
| 3 | 0.2138 | **0.3233** | 0.8682 | 0.8674 | 0.8674 |
| 4 | 0.1762 | 0.3455 | 0.8694 | 0.8791 | 0.8738 |
| **5** | **0.1558** | 0.3588 | **0.8735** | **0.8830** | **0.8777** |

Validation BCE is lowest at epoch 3, while mA, Instance F1 and Challenge_Avg still rise at epoch 5, so the last epoch is packaged. The packaged `run.py` matches timm (max logit difference 0).

**DINOv3 ViT-B/16** (8 epochs, packaged checkpoint from epoch 8):

| Epoch | Train BCE | Val BCE | mA | Instance F1 | Challenge_Avg |
|---|---|---|---|---|---|
| 1 | 0.8573 | 0.7295 | 0.6710 | 0.6138 | 0.6411 |
| 2 | 0.5582 | 0.4883 | 0.7931 | 0.7407 | 0.7657 |
| 3 | 0.3962 | 0.3947 | 0.8272 | 0.7898 | 0.8078 |
| 4 | 0.3192 | 0.3474 | 0.8446 | 0.8195 | 0.8315 |
| 5 | 0.2764 | 0.3286 | 0.8547 | 0.8281 | 0.8407 |
| 6 | 0.2502 | **0.3113** | 0.8636 | 0.8370 | 0.8497 |
| 7 | 0.2293 | 0.3120 | **0.8651** | 0.8458 | 0.8549 |
| **8** | **0.2138** | 0.3172 | 0.8604 | **0.8516** | **0.8555** |

DINOv3 learns more slowly at the start than the supervised models (epoch 1 Challenge_Avg 0.6411, against 0.7632 for ViT-B/16), because the learning rate warms up over the whole run. It then improves steadily: validation BCE is lowest at epoch 6, mA peaks at epoch 7, and Challenge_Avg is highest at epoch 8, so the last epoch is packaged. Its train BCE (0.2138) is still well above the other B-size models' (0.1130 to 0.1393), so it is less over-fitted at the end. The packaged `run.py` matches timm (max logit difference $2.3 \cdot 10^{-6}$).

### Results on the hidden test set (Codabench leaderboard)

Scores of the submitted ZIPs, using the packaged checkpoints above. Each value is the mean over the hidden test datasets, and the best value in each column is in bold.

| Model | Checkpoint | Avg | mA | Label_F1 | Inst_Acc | Inst_Prec | Inst_Rec | Inst_F1 |
|---|---|---|---|---|---|---|---|---|
| [ViT-B/16](notebooks/ViT/upar_track1_vit_b16/upar-track1-vit-b16.ipynb) | epoch 6 | 0.7305 | 0.7402 | 0.5202 | 0.5895 | 0.6805 | 0.7846 | 0.7210 |
| [ViT-L/16](notebooks/ViT/upar_track1_vit_l16/upar-track1-vit-l16.ipynb) | epoch 4 | 0.7210 | 0.7274 | 0.5007 | 0.5820 | 0.6788 | 0.7742 | 0.7148 |
| [DeiT III-B/16](notebooks/ViT/upar_track1_deit3_b16/upar-track1-deit3-b16.ipynb) | best validation epoch | 0.7541 | 0.7641 | 0.5565 | 0.6187 | 0.7067 | 0.8031 | 0.7443 |
| [DeiT III-L/16](notebooks/ViT/upar_track1_deit3_l16/upar-track1-deit3-l16.ipynb) | best validation epoch | 0.7716 | **0.7849** | 0.5684 | 0.6361 | 0.7101 | 0.8320 | 0.7587 |
| [EVA02-B/14](notebooks/ViT/upar_track1_eva02_b14/upar-track1-eva02-b14.ipynb) | best validation epoch | 0.7703 | 0.7775 | 0.5777 | 0.6435 | 0.726 | 0.8206 | 0.7632 |
| [SigLIP 2 ViT-B/16](notebooks/CLIP-Style/upar_track1_siglip2_b16/upar-track1-siglip2-b16.ipynb) | epoch 7 | 0.7570 | 0.7716 | 0.5572 | 0.6172 | 0.6939 | 0.8174 | 0.7430 |
| [SigLIP 2 So400m/14](notebooks/CLIP-Style/upar_track1_siglip2_so400m14/upar-track1-siglip2-so400m14.ipynb) | epoch 5 | 0.7774 | **0.7849** | 0.583 | 0.652 | 0.7236 | **0.8394** | 0.77 |
| [DINOv3 ViT-B/16](notebooks/Self-supervised/upar_track1_dinov3_b16/upar-track1-dinov3-b16.ipynb) | epoch 8 | 0.7638 | 0.7843 | 0.5516 | 0.6165 | 0.6827 | 0.8386 | 0.7443 |
| [EVA02-L/14](notebooks/ViT/upar_track1_eva02_l14/upar-track1-eva02-l14.ipynb) | best validation epoch | **0.7825** | 0.7809 | **0.6024** | **0.6719** | **0.7557** | 0.8288 | **0.7842** |

EVA02-B/14 (87.1M parameters) scores Avg 0.7703, only 0.0013 below DeiT III-L/16 (304.4M) and 0.0162 above DeiT III-B/16. It has the best Label_F1, Inst_Acc, Inst_Prec and Inst_F1; DeiT III-L/16 keeps the best Avg, mA and Inst_Rec. EVA02-B/14's gain over DeiT III-L/16 comes from precision (0.726 vs 0.7101) with lower recall (0.8206 vs 0.8320), so it over-predicts attributes less. DeiT III-L/16 is the best of the other models (Avg 0.7716), +0.0175 over DeiT III-B/16 and +0.0411 over ViT-B/16. DeiT III-B/16 in turn beats ViT-B/16 by 0.0236, so the ImageNet-22k pretraining helps under domain shift. Unlike the torchvision models, the larger DeiT III backbone also helps: mA rises by 0.0208 and Instance F1 by 0.0144. Most of the Instance F1 gain comes from recall (+0.0289) rather than precision (+0.0034), so DeiT III-L/16 over-predicts attributes even more (recall 0.8320 vs precision 0.7101). Among the torchvision models, ViT-B/16 beats ViT-L/16 on every metric, so the larger backbone does not help. For ViT-B/16 vs ViT-L/16, the gap is largest on the label-based metrics: mA is 0.0128 higher and Label_F1 0.0195 higher. Instance F1 differs by only 0.0062.

SigLIP 2 ViT-B/16 (92.9M parameters) scores Avg 0.7570. It beats DeiT III-B/16 by 0.0029 (mA +0.0075, Label_F1 +0.0007), but is below EVA02-B/14 by 0.0133 and DeiT III-L/16 by 0.0146. Although its validation Challenge_Avg (0.8656) is above the ViT-B/16 and ViT-L/16 runs, it drops by 0.1086 on the test domain: mA falls by 0.0879 (0.8595 → 0.7716) and Instance F1 by 0.130 (0.8730 → 0.7430). Its instance recall (0.8174) is well above its precision (0.6939), and its precision and Instance F1 are slightly below DeiT III-B/16's (0.7067 and 0.7443). So the larger image-text pretraining set did not give the best robustness under domain shift here, and validation score is a poor predictor of the leaderboard ranking.

SigLIP 2 So400m/14 (427.7M parameters) scores Avg 0.7774, the best of the earlier models: +0.0058 over DeiT III-L/16, +0.0071 over EVA02-B/14 and +0.0204 over SigLIP 2 ViT-B/16. It has the best Label_F1 (0.583), Inst_Acc (0.652), Inst_Rec (0.8394) and Inst_F1 (0.77), and ties DeiT III-L/16 on mA (0.7849); EVA02-B/14 keeps the best Inst_Prec (0.726 vs 0.7236). Compared with SigLIP 2 ViT-B/16, the larger backbone raises mA by 0.0133, Label_F1 by 0.0258 and Instance F1 by 0.0270, so unlike the torchvision models, scaling the SigLIP 2 encoder helps under domain shift. Its instance recall (0.8394) is again well above its precision (0.7236), so it over-predicts attributes, and its gain over the other models comes mostly from recall. Its validation Challenge_Avg (0.8777, epoch 5) is also the highest, and the test-domain drop is 0.1003 (mA 0.8735 → 0.7849, Instance F1 0.8830 → 0.77), smaller than SigLIP 2 ViT-B/16's 0.1086.

DINOv3 ViT-B/16 (85.7M parameters) scores Avg 0.7638. It is the second-best B-size model, +0.0097 over DeiT III-B/16 and +0.0068 over SigLIP 2 ViT-B/16, but 0.0065 below EVA02-B/14 and 0.0136 below SigLIP 2 So400m/14. Its mA (0.7843) is only 0.0006 below the best (0.7849), and its Inst_Rec (0.8386) is second only to SigLIP 2 So400m/14 (0.8394), which shows that self-supervised pretraining transfers well to the test domain. Its weaker side is precision: Inst_Prec (0.6827) is the lowest apart from ViT-B/16 and ViT-L/16, and it is far below its recall, so it over-predicts attributes like the other models. Label_F1 (0.5516) is below DeiT III-B/16 (0.5565) and SigLIP 2 ViT-B/16 (0.5572), and Instance F1 (0.7443) ties DeiT III-B/16. Its validation Challenge_Avg (0.8555) is below both SigLIP 2 models' (0.8656 and 0.8777) but above ViT-B/16's (0.8408), and the test-domain drop is 0.0917 (mA 0.8604 → 0.7843, Instance F1 0.8516 → 0.7443), smaller than SigLIP 2 ViT-B/16's 0.1086.

EVA02-L/14 (305.1M parameters) scores Avg 0.7825, the best of all models: +0.0051 over SigLIP 2 So400m/14, +0.0109 over DeiT III-L/16 and +0.0122 over EVA02-B/14. It has the best Label_F1 (0.6024), Inst_Acc (0.6719), Inst_Prec (0.7557) and Inst_F1 (0.7842). SigLIP 2 So400m/14 and DeiT III-L/16 keep the best mA (0.7849, 0.0040 above its 0.7809), and SigLIP 2 So400m/14 keeps the best Inst_Rec (0.8394 vs 0.8288). Its gain comes from precision (0.7557 vs 0.7236 for SigLIP 2 So400m/14), so it over-predicts attributes less: its recall exceeds its precision by 0.0731, against 0.1158 for SigLIP 2 So400m/14. Compared with EVA02-B/14, the larger backbone raises Label_F1 by 0.0247, Inst_Prec by 0.0297 and Instance F1 by 0.0210, so scaling EVA02 helps under domain shift, unlike the torchvision ViT models. The training log of this run is not saved in the notebook, so its per-epoch validation scores and packaged epoch are not listed here.

For ViT-B/16, Challenge_Avg drops from 0.8408 on validation to 0.7305. mA falls by 0.093 (0.8327 → 0.7402) and Instance F1 by 0.129 (0.8500 → 0.7210). The validation Instance F1 of this run is the older per-image average, so that drop is approximate. Instance recall (0.7846) is higher than precision (0.6805), so the model predicts too many attributes on the test domain. This fits the positive-weighted loss, which pushes rare attributes above the 0.5 threshold. ViT-L/16 drops from 0.8293 to 0.7210, and its recall (0.7742) is also well above its precision (0.6788).

### Running a notebook

1. Open the notebook in Kaggle with **Internet** and a **GPU** enabled, and run all cells.
2. Section 1 downloads Market1501, PA-100K and PETA into `/kaggle/working/upar_2027/data` (97,669 training and 33,407 validation images).
3. Sections 2–4 fine-tune the ViT and keep the checkpoint with the best validation $`\text{Challenge\_Avg}`$.
4. Sections 5–6 write `run.py` and `metadata.yaml`, build `/kaggle/working/upar_track1_<model>_submission.zip`, and smoke-test the packaged `run.py`. Upload that ZIP to Codabench.

The L/16 models are roughly three times slower than the B/16 models per epoch. Lower `BATCH_SIZE` to 16 if the GPU runs out of memory.

### Model

Each crop $`x`$ is resized to $`224 \times 224`$ (torchvision's ViT needs square inputs) and normalised with the ImageNet mean and standard deviation.

**Encoder.** The image is cut into $`n = (224/16)^2 = 196`$ patches of $`16 \times 16`$ pixels. Each patch $`x_p^{(i)}`$ is flattened and linearly projected to width $`d`$ (a $`16 \times 16`$ convolution with stride 16). A learnable class token is prepended, and learnable position embeddings are added:

```math
X_0 = \big[\, x_{\text{cls}};\ E\,x_p^{(1)};\ \dots;\ E\,x_p^{(196)} \,\big] + E_{\text{pos}}, \qquad E \in \mathbb{R}^{d \times 768},\ E_{\text{pos}} \in \mathbb{R}^{197 \times d}.
```

Each of the $`D`$ pre-norm transformer blocks applies multi-head self-attention (MSA) with $`H`$ heads and a two-layer GELU MLP of width $`4d`$, each with a residual connection:

```math
X'_k = X_{k-1} + \mathrm{MSA}\big(\mathrm{LN}(X_{k-1})\big), \qquad X_k = X'_k + \mathrm{MLP}\big(\mathrm{LN}(X'_k)\big), \qquad k = 1, \dots, D,
```

```math
\mathrm{MSA}(X) = \big[\mathrm{head}_1; \dots; \mathrm{head}_H\big] W_O, \qquad \mathrm{head}_m = \mathrm{softmax}\!\left(\frac{Q_m K_m^{\top}}{\sqrt{d/H}}\right) V_m .
```

The encoder output is the final layer norm of the class token, $`h = f_\theta(x) = \mathrm{LN}(X_D)_{\text{cls}} \in \mathbb{R}^{d}`$.

DeiT III uses the same blocks with two changes. Each residual branch is scaled by a learnable per-channel LayerScale vector $`\gamma`$, as in $`X'_k = X_{k-1} + \gamma_{1,k} \odot \mathrm{MSA}(\cdot)`$. The position embeddings cover only the 196 patch tokens, not the class token.

| Model | Blocks $`D`$ | Width $`d`$ | Heads $`H`$ | MLP width $`4d`$ | Steps per epoch | Total steps $`T`$ | Warm-up $`T_w`$ | Lowest LR factor $`\lambda^{D+1}`$ |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16, DeiT III-B/16 | 12 | 768 | 12 | 3072 | 1,526 | 12,208 | 1,526 | $`0.75^{13} = 0.024`$ |
| ViT-L/16, DeiT III-L/16 | 24 | 1024 | 16 | 4096 | 3,052 | 15,260 | 3,052 | $`0.85^{25} = 0.017`$ |
| SigLIP 2 ViT-B/16 | 12 | 768 | 12 | 3072 | 1,526 | 12,208 | 1,526 | $`0.75^{13} = 0.024`$ |
| SigLIP 2 So400m/14 | 27 | 1152 | 16 | 4304 | 6,104 | 30,520 | 6,104 | $`0.85^{28} = 0.011`$ |

Steps per epoch are $`\lfloor 97{,}669 / B \rfloor`$, because the last incomplete batch is dropped. The lowest LR factor applies to the patch embedding, class token and position embeddings, so their peak learning rate is $`2.4 \cdot 10^{-6}`$ for the B/16 models and $`8.6 \cdot 10^{-7}`$ for the L/16 models.

SigLIP 2 ViT-B/16 differs from the blocks above in three ways. It has no class token and no `[CLS]` position, so the 196 position embeddings cover only the patch tokens. Inputs are normalised with mean and standard deviation $`0.5`$ instead of the ImageNet statistics. The encoder output is pooled by SigLIP's pretrained attention-pooling head (multi-head attention pooling, MAP): a learned query $`q`$ attends to all 196 final-norm tokens, followed by a residual MLP, which is fine-tuned together with the encoder:

```math
h = u + \mathrm{MLP}\big(\mathrm{LN}(u)\big), \qquad u = \mathrm{MSA}\big(q,\ \mathrm{LN}(X_D),\ \mathrm{LN}(X_D)\big), \qquad h \in \mathbb{R}^{768}.
```

The pooling head has the same layer index as the final norm and classifier, $`l = D + 1`$. The text encoder is not used. Configuration: `timm` `vit_base_patch16_siglip_224.v2_webli`, 224 px input, 8 epochs, batch 64, peak LR $`10^{-4}`$, layer decay 0.75, drop-path 0.1, weight decay 0.05, 1 warm-up epoch, seed 42.

SigLIP 2 So400m/14 uses the same design (no class token, attention-pooling head, mean and standard deviation 0.5) with shape-optimised dimensions: 27 blocks, width 1152, 16 heads, MLP width 4304, patch size 14 (256 patch tokens, $h \in \mathbb{R}^{1152}$). Configuration: `timm` `vit_so400m_patch14_siglip_224.v2_webli`, 224 px input, 5 epochs, batch 16, peak LR $`5 \cdot 10^{-5}`$, layer decay 0.85, drop-path 0.2, weight decay 0.05, 1 warm-up epoch, seed 42. Its embedding-layer peak learning rate is $`5.3 \cdot 10^{-7}`$.

**Head.** A zero-initialised linear head predicts one logit per attribute, and a sigmoid turns each logit into an independent probability:

```math
z = W h + b, \qquad W \in \mathbb{R}^{40 \times d},\ b \in \mathbb{R}^{40}, \qquad p_j = \sigma(z_j) = \frac{1}{1 + e^{-z_j}}, \quad j = 1, \dots, 40.
```

### Loss: positive-weighted binary cross-entropy

Attributes such as *Hair-Length-Bald* or *Accessory-Glasses-Sun* are rare, so positives are up-weighted. With $`N`$ training images and $`P_j`$ positives for attribute $`j`$, the weight is the clipped negative-to-positive ratio:

```math
w_j = \mathrm{clip}\!\left(\frac{N - P_j}{\max(P_j, 1)},\ 1,\ 20\right).
```

For a batch of $`B`$ images with labels $`y_{ij} \in \{0, 1\}`$, the loss is

```math
\mathcal{L} = -\frac{1}{40\,B} \sum_{i=1}^{B} \sum_{j=1}^{40} \Big[ w_j\, y_{ij} \log \sigma(z_{ij}) + (1 - y_{ij}) \log\big(1 - \sigma(z_{ij})\big) \Big].
```

### Optimisation

AdamW with weight decay $`0.05`$ (none on biases, norms, the class token and positional embeddings), mixed-precision training, and gradient clipping at $`\lVert g \rVert_2 \le 1`$.

**Layer-wise learning-rate decay.** With $`D`$ transformer blocks ($`D = 12`$ for ViT-B, $`D = 24`$ for ViT-L), the patch embedding, class token and positional embeddings get layer index $`l = 0`$, block $`k`$ gets $`l = k + 1`$, and the final norm and head get $`l = D + 1`$. Earlier layers learn more slowly, which preserves the pretrained features:

```math
\eta_l(t) = \eta(t) \cdot \lambda^{\,D + 1 - l}.
```

**Warm-up and cosine schedule.** With $`T`$ total steps and $`T_w`$ warm-up steps (one epoch):

```math
\eta(t) =
\begin{cases}
\eta_{\max} \, \dfrac{t + 1}{T_w}, & t < T_w, \\[2ex]
\dfrac{\eta_{\max}}{2} \left( 1 + \cos\!\left( \pi \, \dfrac{t - T_w}{T - T_w} \right) \right), & t \ge T_w.
\end{cases}
```

**Augmentation.** Random horizontal flip, 10-pixel edge padding followed by a random $`224 \times 224`$ crop, and random erasing ($`p = 0.25`$). There is no colour jitter, because it would corrupt the colour labels.

### Inference and test-time augmentation

`run.py` averages the logits of each image and its horizontal flip $`\tilde{x}`$, then thresholds at 0.5 as the evaluation does:

```math
p_j(x) = \sigma\!\left( \frac{z_j(x) + z_j(\tilde{x})}{2} \right), \qquad \hat{y}_j = \mathbb{1}\big[p_j(x) > 0.5\big].
```

Checkpoints are stored in float16 to halve the ZIP size and are cast back to float32 when loaded.

### Checkpoint selection

After every epoch, the notebooks compute $`\text{mA}`$, $`\text{Instance\_F1}`$ and $`\text{Challenge\_Avg}`$ (see [Evaluation](#evaluation)) separately on the Market1501, PA-100K and PETA parts of the validation split, average them, and keep the epoch with the highest $`\text{Challenge\_Avg}`$. The hidden test set comes from an unseen domain, so expect the leaderboard score to be lower than the validation score.

## Code Submission and Winner Selection

After the challenge deadline, participating teams are required to submit their code, models, and a short description of their method (fact sheet) by **Oct 26, 2026, AoE** to upar.challenge@gmail.com. A panel of experts will review the entries. The executable/source code should include trained models, training code, and necessary parameters to ensure reproducibility. Teams that do not comply are removed from the final ranking.

> Note: Submitted codes will not be saved on the evaluation servers. Final submission codebases provided to the organizers will be used only for reproducibility checks and winner verification, and will not be shared publicly or with other participants at any stage.

## Paper Submission

All challenge participants are strongly encouraged to submit a technical paper to the RWS workshop. The paper should detail your methodology, experiments, and results on the challenge dataset. Accepted papers will be presented at the workshop and published in the WACV workshop proceedings.

- **Format:** WACV format
- **Submission system:** OpenReview
- **Challenge paper deadline:** Oct 23, 2026, AoE

## Schedule

| Date | Event |
|------|-------|
| Sep 21, 2026 | Challenge opens, training data available |
| Oct 21, 2026, AoE | Challenge submission deadline |
| Oct 23, 2026, AoE | Paper submission deadline |
| Oct 26, 2026, AoE | Code & fact sheet submission deadline, code verification starts |
| Jan 4/5, 2027 | RWS Workshop at WACV 2027 |

## Data

### Training data

Training data and starter kit are available via the Starter Kit link on the Codabench page.

- The use of external data to fine-tune PAR is **not permitted**.
- Pre-trained models trained on common datasets such as ImageNet, or foundation models/VLMs that were not specifically trained for PAR, are **permitted**.

### Attributes

40 binary attributes, in this exact order. Your model returns one probability per attribute in this order. All 40 attributes are scored; always return all 40 values.

| # | Attribute | # | Attribute |
|---|-----------|---|-----------|
| 0 | Age-Young | 20 | LowerBody-Length-Short |
| 1 | Age-Adult | 21 | LowerBody-Color-Black |
| 2 | Age-Old | 22 | LowerBody-Color-Blue |
| 3 | Gender-Female | 23 | LowerBody-Color-Brown |
| 4 | Hair-Length-Short | 24 | LowerBody-Color-Green |
| 5 | Hair-Length-Long | 25 | LowerBody-Color-Grey |
| 6 | Hair-Length-Bald | 26 | LowerBody-Color-Orange |
| 7 | UpperBody-Length-Short | 27 | LowerBody-Color-Pink |
| 8 | UpperBody-Color-Black | 28 | LowerBody-Color-Purple |
| 9 | UpperBody-Color-Blue | 29 | LowerBody-Color-Red |
| 10 | UpperBody-Color-Brown | 30 | LowerBody-Color-White |
| 11 | UpperBody-Color-Green | 31 | LowerBody-Color-Yellow |
| 12 | UpperBody-Color-Grey | 32 | LowerBody-Color-Other |
| 13 | UpperBody-Color-Orange | 33 | LowerBody-Type-Trousers&Shorts |
| 14 | UpperBody-Color-Pink | 34 | LowerBody-Type-Skirt&Dress |
| 15 | UpperBody-Color-Purple | 35 | Accessory-Backpack |
| 16 | UpperBody-Color-Red | 36 | Accessory-Bag |
| 17 | UpperBody-Color-White | 37 | Accessory-Glasses-Normal |
| 18 | UpperBody-Color-Yellow | 38 | Accessory-Glasses-Sun |
| 19 | UpperBody-Color-Other | 39 | Accessory-Hat |

### Test data

The test set is hidden. It is mounted into your container, read-only, with this layout:

```
input_data/
├── images.csv      one image path per line, no labels
├── images/         the image files, at the paths listed in images.csv
│   └── <dataset>/...
└── task.json       attribute names, number of images
```

The images are pedestrian crops of varying size and aspect ratio. You do not have to parse these files yourself: the ingestion program does it and hands your code one ready-made sample dict per image.

## Evaluation

The protocol and metrics are those of the previous UPAR challenge edition (WACV 2024). The scoring program reproduces the original evaluation script exactly.

### Decision threshold

Your model returns a probability in [0, 1] per attribute. An attribute counts as present if its probability is **above 0.5**. Calibrate your outputs accordingly. Values outside [0, 1] fail the submission.

### Metrics

| Leaderboard | Key | Meaning |
|-------------|-----|---------|
| Avg | Challenge_Avg | harmonic mean of mA and Instance_F1 – **the ranking metric** |
| mA | mA | label-based mean accuracy |
| Label_F1 | Label_F1 | label-based F1 |
| Inst_Acc | Instance_Acc | instance-based accuracy |
| Inst_Prec | Instance_Prec | instance-based precision |
| Inst_Rec | Instance_Rec | instance-based recall |
| Inst_F1 | Instance_F1 | instance-based F1 |

Label-based metrics are computed per attribute and then averaged over the $`M = 40`$ attributes, so rare attributes weigh as much as frequent ones. $`\text{TP}_j, \text{FN}_j, \text{TN}_j, \text{FP}_j`$ are the counts for attribute $`j`$ over all images:

```math
\text{mA} = \frac{1}{M} \sum_{j=1}^{M} \frac{1}{2} \left( \frac{\text{TP}_j}{\text{TP}_j + \text{FN}_j} + \frac{\text{TN}_j}{\text{TN}_j + \text{FP}_j} \right)
```

```math
\text{Label\_F1} = \frac{1}{M} \sum_{j=1}^{M} \frac{2\,\text{TP}_j}{2\,\text{TP}_j + \text{FP}_j + \text{FN}_j}
```

Instance-based metrics are computed per image $`i`$, over the set of attributes predicted present $`P_i`$ and the set actually present $`G_i`$, and then averaged over the $`N`$ images:

```math
\text{Instance\_Acc} = \frac{1}{N} \sum_{i=1}^{N} \frac{\lvert P_i \cap G_i \rvert}{\lvert P_i \cup G_i \rvert},
\qquad
\text{Instance\_Prec} = \frac{1}{N} \sum_{i=1}^{N} \frac{\lvert P_i \cap G_i \rvert}{\lvert P_i \rvert},
\qquad
\text{Instance\_Rec} = \frac{1}{N} \sum_{i=1}^{N} \frac{\lvert P_i \cap G_i \rvert}{\lvert G_i \rvert}
```

```math
\text{Instance\_F1} = \frac{2 \cdot \text{Instance\_Prec} \cdot \text{Instance\_Rec}}{\text{Instance\_Prec} + \text{Instance\_Rec}}
```

The ranking metric is the harmonic mean of the two views:

```math
\text{Challenge\_Avg} = \frac{2 \cdot \text{mA} \cdot \text{Instance\_F1}}{\text{mA} + \text{Instance\_F1}}
```

With test datasets $`\mathcal{D}_1, \dots, \mathcal{D}_K`$ (see [Datasets](#datasets)), every leaderboard value is the unweighted mean over the datasets, for example

```math
\text{Challenge\_Avg} = \frac{1}{K} \sum_{k=1}^{K} \text{Challenge\_Avg}(\mathcal{D}_k).
```

### Datasets

If the test set contains images from several datasets, every dataset is scored on its own, over all 40 attributes. Every leaderboard value is the mean over the datasets, so a small dataset weighs as much as a large one. The standard deviation over the datasets is reported in the detailed results.

Attributes whose ground truth is marked invalid in a dataset are excluded from the evaluation of that dataset.

### Leaderboard

Your **latest** submission is the one shown on the leaderboard, not your best.

## Submission

You upload a zip containing your model; the platform runs it against data you never see.

### The zip

Zip the contents of your submission folder, not the folder itself:

```bash
cd my_submission
zip -r ../my_submission.zip . -x "*/__pycache__/*" "*.DS_Store" ".git/*"
unzip -l ../my_submission.zip | head    # run.py must be at the root
```

Required at the root of the archive:

- `run.py` – your entry point
- `metadata.yaml` – copy it unchanged from the example submission
- anything else – weights, modules, configs

### The API

The ingestion program imports `run.py` and calls:

```python
def load_model() -> None:
    """Optional. Called once before inference. Load your weights here."""

def predict_image(sample: dict) -> list[float]:
    """Called once per image. Return 40 probabilities."""
```

`sample` is:

```python
{
  "index":           0,                        # row in images.csv
  "image":           "<dataset>/images/0.jpg",     # path as listed in images.csv
  "image_path":      "/app/input_data/images/<dataset>/images/0.jpg",   # absolute, readable
  "attribute_names": ["Age-Young", "Age-Adult", ...],   # the 40 names, in order
}
```

Return one probability per attribute, in the order of `attribute_names`: a list, tuple, numpy array or tensor of 40 floats in [0, 1], or a dict mapping every attribute name to its probability. An attribute counts as present above 0.5. NaN, values outside [0, 1] or a wrong length fail the submission with a message in the ingestion log.

### Batched inference

Per-image calls are simple but slow on a GPU. Define `predict_batch` instead of, or in addition to, `predict_image` and it is used:

```python
BATCH_SIZE = 64    # optional, default 32

def predict_batch(samples: list[dict]) -> list[list[float]]:
    """One row of 40 probabilities per sample, in order."""
```

### Runtime environment

- No network access. Ship your weights inside the zip. Nothing is downloaded at run time.
- The image is `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime`: Python 3.11, torch 2.4.1, torchvision, numpy, Pillow. Anything else your code imports must travel in the zip.
- There is a wall-clock limit for the whole run. The ingestion program logs its progress and per-image timing.
- The working directory is your submission folder, so relative paths to your weights work.

## Terms and Conditions

### 1. The challenge

The UPAR Challenge at the Real-World Surveillance workshop, WACV 2027, is a skill-based contest with three independent tracks. These terms cover Track 1: Pedestrian Attribute Recognition. The challenge is run entirely on Codabench: the benchmark page is the official source for the task description, data, evaluation, schedule, rules and announcements.

### 2. Eligibility

- Participation is open to everyone except the organizers and members of their research groups.
- You may participate alone or in a team. Each person may belong to one team only, and each team may use one Codabench account only.
- To be eligible for ranking and prizes you must register with your real identity and affiliation.

### 3. Training data, pre-training, and post-processing

- Models must be trained only on the training data provided by the organizers.
- The use of external data to fine-tune PAR is not permitted.
- Pre-trained models trained on common datasets such as ImageNet, or foundation models/VLMs that were not specifically trained for PAR, are permitted.
- Any use of the test images is prohibited, including training, pseudo-labelling, manual annotation and hyper-parameter selection, and leads to disqualification. Test images are never released. Attempts to extract them from the evaluation container lead to disqualification.
- The use of information about the structure of the hidden test set is not allowed. Using information, e.g., from the image directories to apply separate models/hyperparameters to different subsets, leads to disqualification.

### 4. Submissions

- Submissions are code submissions as described in the Submission section. The submitted code must not access the network or tamper with the evaluation environment.
- Submission limits per day and in total are shown on the phase.
- The ranking metric is Challenge_Avg. The latest valid submission of a team counts.
- To win, a submission must outperform the organizers' baseline.

### 5. Verification and publication

- The top-ranked teams must provide their training code and a short description of the method (fact sheet) so the organizers can verify the result and compliance with these terms. Teams that do not comply are removed from the final ranking.
- The top three teams release their code under an OSI-approved open-source licence.
- All participants are invited to submit papers to the RWS workshop by Oct 23, 2026 and to present their work at the workshop.

### 6. Data licence

The UPAR annotations are published under the Creative Commons Attribution-NonCommercial-ShareAlike 3.0 licence. The underlying image datasets keep their own licences and are for non-commercial research use only. If you use the data, cite the UPAR paper and the papers of the sub-datasets.

### 7. General conditions

- The organizers may change the schedule, clarify the rules, or disqualify participants who violate these terms or the spirit of the challenge.
- The organizers' decisions on ranking and eligibility are final.
- The data and the platform are provided as is, without warranty.
- By registering you accept these terms.

## Citation

```bibtex
@inproceedings{specker2023upar,
  title={UPAR: Unified Pedestrian Attribute Recognition and Person Retrieval},
  author={Specker, Andreas and Cormier, Mickael and Beyerer, Jurgen},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2023}
}

@inproceedings{cormier2023upar,
  title={UPAR challenge: pedestrian attribute recognition and attribute-based person retrieval-dataset, design, and results},
  author={Cormier, Mickael and Specker, Andreas and Jacques, Julio CS and Florin, Lucas and Metzler, J{\"u}rgen and Moeslund, Thomas B and Nasrollahi, Kamal and Escalera, Sergio and Beyerer, J{\"u}rgen},
  booktitle={2023 IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)},
  pages={166--175},
  year={2023},
  organization={IEEE}
}

@inproceedings{cormier2024upar,
  title={Upar challenge 2024: Pedestrian attribute recognition and attribute-based person retrieval-dataset, design, and results},
  author={Cormier, Mickael and Specker, Andreas and Junior, Julio and CS, Jacques and Moritz, Lennart and Metzler, J{\"u}rgen and Moeslund, Thomas B and Nasrollahi, Kamal and Escalera, Sergio and Beyerer, J{\"u}rgen},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={359--367},
  year={2024}
}
```
