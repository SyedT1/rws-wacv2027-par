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

The notebooks follow the structure and download cells of the organizers' Kaggle example notebook, which was written for Track 2 (attribute-based retrieval), but they package a **Track 1** submission. Both ViT checkpoints are trained on ImageNet-1K only, which the rules permit, and `torchvision` is already in the evaluation image, so nothing besides the weights needs to ship in the ZIP. The DeiT III checkpoints are pretrained on ImageNet-22k and fine-tuned on ImageNet-1K, and timm downloads them from the Hugging Face Hub. timm is not in the evaluation image, so the DeiT III `run.py` defines the network in plain PyTorch with timm's parameter names, and the packaging step checks that it gives the same logits as the timm model. DeiT III training also uses stochastic depth (drop-path rate 0.1 for B, 0.2 for L). The EVA02 checkpoints are pretrained on ImageNet-22k with masked image modeling (EVA-CLIP features as the target) and fine-tuned on ImageNet-22k and ImageNet-1K at 448 px. The notebooks train them at 224 px with the CLIP mean and standard deviation, the same drop-path rates as DeiT III, and the same plain-PyTorch `run.py` check. EVA02-L uses batch size 16 so that training fits in the 16 GB of a T4. Submission ZIPs exceed GitHub's 100 MB file limit, so they are stored with Git LFS (`*.zip` in `.gitattributes`); run `git lfs pull` after cloning to download them.

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

Both runs averaged per-image F1 scores to get Instance F1. The notebooks now compute it from the image-averaged precision and recall, as the evaluation script does (see [Evaluation](#evaluation)), so later runs can show slightly different Instance F1 and Challenge_Avg values. Both DeiT III models have been trained and scored (see below); the EVA02 models have not been trained yet.

### Results on the hidden test set (Codabench leaderboard)

Scores of the submitted ZIPs, using the packaged checkpoints above. Each value is the mean over the hidden test datasets, and the best value in each column is in bold.

| Model | Checkpoint | Avg | mA | Label_F1 | Inst_Acc | Inst_Prec | Inst_Rec | Inst_F1 |
|---|---|---|---|---|---|---|---|---|
| [ViT-B/16](notebooks/ViT/upar_track1_vit_b16/upar-track1-vit-b16.ipynb) | epoch 6 | 0.7305 | 0.7402 | 0.5202 | 0.5895 | 0.6805 | 0.7846 | 0.7210 |
| [ViT-L/16](notebooks/ViT/upar_track1_vit_l16/upar-track1-vit-l16.ipynb) | epoch 4 | 0.7210 | 0.7274 | 0.5007 | 0.5820 | 0.6788 | 0.7742 | 0.7148 |
| [DeiT III-B/16](notebooks/ViT/upar_track1_deit3_b16/upar-track1-deit3-b16.ipynb) | best validation epoch | 0.7541 | 0.7641 | 0.5565 | 0.6187 | 0.7067 | 0.8031 | 0.7443 |
| [DeiT III-L/16](notebooks/ViT/upar_track1_deit3_l16/upar-track1-deit3-l16.ipynb) | best validation epoch | **0.7716** | **0.7849** | **0.5684** | **0.6361** | **0.7101** | **0.8320** | **0.7587** |

DeiT III-L/16 has the best score on every metric (Avg 0.7716), +0.0175 over DeiT III-B/16 and +0.0411 over ViT-B/16. DeiT III-B/16 in turn beats ViT-B/16 by 0.0236, so the ImageNet-22k pretraining helps under domain shift. Unlike the torchvision models, the larger DeiT III backbone also helps: mA rises by 0.0208 and Instance F1 by 0.0144. Most of the Instance F1 gain comes from recall (+0.0289) rather than precision (+0.0034), so DeiT III-L/16 over-predicts attributes even more (recall 0.8320 vs precision 0.7101). Among the torchvision models, ViT-B/16 beats ViT-L/16 on every metric, so the larger backbone does not help. For ViT-B/16 vs ViT-L/16, the gap is largest on the label-based metrics: mA is 0.0128 higher and Label_F1 0.0195 higher. Instance F1 differs by only 0.0062.

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

Steps per epoch are $`\lfloor 97{,}669 / B \rfloor`$, because the last incomplete batch is dropped. The lowest LR factor applies to the patch embedding, class token and position embeddings, so their peak learning rate is $`2.4 \cdot 10^{-6}`$ for the B/16 models and $`8.6 \cdot 10^{-7}`$ for the L/16 models.

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
