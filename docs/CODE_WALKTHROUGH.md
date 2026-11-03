# Code walkthrough

This document follows one image through the code, file by file, in the order the data
flows. Read it next to the source; every section names the file and the lines that matter.
At the end are the questions an examiner or interviewer is likely to ask, with short answers.

---

## 1. The idea in four sentences

1. Labelled medical data is expensive, so we segment **without any training**: the user only
   describes the target in words.
2. A medical CLIP model (**BMC-CLIP**) knows what "polyp" or "malignant tumor" looks like, but
   its per-pixel output is noisy; **CorrCLIP** (with help from **DINO** and **MedSAM2**
   regions) cleans it into a *target map* that shows **where** the target is, blurrily.
3. **MedSAM2** draws sharp outlines when it is given a point, so we give it the **five
   highest, well-separated peaks** of the target map and get 15 candidate outlines.
4. We keep the candidate that MedSAM2 itself is **most confident** about (highest predicted IoU).

In one line: **CLIP points, MedSAM2 draws.**

---

## 2. One image through the code

```
image (H x W x 3, BGR)
 │
 │  pipeline.py  ZeroShotSegmenter.segment()          ← the whole method, ~40 lines
 │
 ├─1─► saliency/regions.py   RegionGenerator           MedSAM2 automatic masks
 │        └ models/medsam2.py AutomaticRegionGenerator  → region map (H x W int)
 │
 ├─2─► saliency/estimator.py SaliencyEstimator
 │        data/preprocessing.py   rescale (short side 336), CLIP normalisation
 │        for every 336 x 336 window (stride 112):
 │           models/dino.py       DINO ViT-B/8 patch features → patch similarity
 │           models/corrclip.py   BMC-CLIP features rebuilt from DINO + regions
 │           cosine with the class embeddings (models/text_encoder.py)
 │        average overlapping windows, resize, softmax(40 · cos) → target map (H x W, 0..1)
 │
 ├─3─► prompting/strategies.py MultiPeakPrompts         → 5 points (x, y)
 │
 ├─4─► models/medsam2.py     PointPromptSegmenter       → 5 x 3 = 15 masks + predicted IoU
 │
 └─5─► selection/rules.py    MaskSelector               → final mask (H x W bool)
```

Before any image, once: `models/clip_backbone.py` loads BMC-CLIP, `models/text_encoder.py`
turns the text prompts into two vectors (background, target), and `models/medsam2.py` builds
MedSAM2. All of this happens in `ZeroShotSegmenter.from_config()`.

---

## 3. File by file

### `pipeline.py`: start here

`ZeroShotSegmenter` has two methods that matter.

* `from_config(cfg)` loads the frozen models and builds the five stages. Note that one MedSAM2
  model is used twice: by `RegionGenerator` (automatic mode) and by `PointPromptSegmenter`
  (point mode).
* `segment(image)` runs the five steps. Read it top to bottom; each step is one or two lines:

```python
regions = self.region_generator(image_rgb)                    # 1
saliency = self.saliency_estimator(image_bgr, regions).target  # 2
points = self.prompter(saliency)                              # 3
for index, (x, y) in enumerate(points):                       # 4
    candidates.extend(self.point_segmenter.segment_point(x, y, prompt_index=index))
kept = self.selector.choose(candidates)                       # 5
```

`set_prompts(background, target)` changes what is segmented without reloading any model; the
website and `scripts/evaluate_all.py` use it.

### `config.py`: every setting in one place

Dataclasses, one per section (`weights`, `saliency`, `corrclip`, `prompting`, `selection`,
`dataset`, `evaluation`, `runtime`). The default values **are** the method; a dataset YAML
only sets where the files are and the text prompts. A typo such as `prompting.num_point`
raises an error instead of being silently ignored (`_build_dataclass`).

### `models/text_encoder.py`: words to vectors (formula 1)

`encode_classes` puts each phrase into 12 templates ("a medical image showing {}.",
"an ultrasound image of {}.", ...), embeds each sentence with BMC-CLIP's text encoder,
normalises, averages over templates, then averages over the synonyms of the class. Result:
one unit vector for the background, one for the target.

Why templates: CLIP was trained on captions, so a full sentence is closer to what it saw than
a bare word. Why averaging synonyms: "benign tumor" and "malignant tumor" together describe
a breast lesion better than either alone.

### `saliency/regions.py` and `models/medsam2.py → AutomaticRegionGenerator`

MedSAM2's automatic mode places a 16 x 16 grid of points, keeps confident masks, and we paint
them into one integer map (smallest first, so a larger mask wins where they overlap). These
regions tell CorrCLIP which patches belong to the same object.

### `saliency/estimator.py`: the target map (formula 3)

`SaliencyEstimator.__call__`:

1. `prepare_clip_input` resizes the image so its short side is 336 (the size BMC-CLIP was
   trained on) and normalises it with CLIP's mean and std.
2. `sliding_windows` cuts 336 x 336 windows with stride 112, so every pixel is seen by
   several windows; their logits are averaged (`logit_sum / counts`).
3. Per window, `_window_logits`: DINO features, then CorrCLIP features, then the cosine of
   every patch feature with the two class vectors (`feats @ self.class_embeddings.T`).
4. The averaged logits are resized to the original image and turned into probabilities:
   `(logits * 40).softmax(dim=0)`. Channel 1 is the target map.

The factor 40 (`saliency.logit_scale`) is a temperature: cosine similarities differ only in
the second decimal, and multiplying by 40 before the softmax turns small differences into a
clear map.

### `models/dino.py` and `models/corrclip.py`: clean patch features (formula 2)

Plain CLIP is trained on whole images, so its per-patch features are noisy: a patch on a
polyp may "look at" the whole image in the last attention layer. CorrCLIP replaces only the
**last** attention layer:

* `similarity = dino_feats @ dino_feats.T`: DINO, a self-supervised model, is very good at
  saying which patches look alike.
* `_merge_regions`: regions with similar mean DINO features are merged (DBSCAN).
* `_scope_attention`: similarities between patches of **different** regions are set to
  −∞, then `softmax(4 · similarity)` gives the attention weights. A patch may only average
  similar patches of its own region.
* In `forward`, those weights multiply CLIP's **value** vectors of the last layer
  (`torch.bmm(weights, v)`); CLIP's own query/key attention is not used.
* Two refinements from the paper are added: a *spatial branch* (the same reconstruction on
  the mean of intermediate layers, weight 1.0) and a *semantic branch* (one summary token per
  region, weight 0.5). Then `ln_post` and the projection into CLIP's joint text-image space.

`_block_forward` exists because open_clip would turn a boolean "may not attend" mask into a
small float bias; calling the attention directly keeps the mask a hard block.

### `prompting/strategies.py`: five points (formula 4)

`MultiPeakPrompts`: take the highest pixel of the target map, blank a square of half-side
0.1 · min(H, W) around it, repeat until five points. Why several points: the hottest peak is
sometimes on the wrong structure; the others are backups. Why the square: without it all five
points would sit next to each other on the same blob.

### `models/medsam2.py → PointPromptSegmenter`: outlines

`set_image` encodes the full-resolution image once; `segment_point` asks for three masks for
one positive point (`multimask_output=True`): small, medium and large interpretations of the
click, each with MedSAM2's own estimate of its quality (predicted IoU). Five points give 15
candidates.

### `selection/rules.py`: the choice (formula 5)

`MaskSelector.choose` sorts the 15 candidates by predicted IoU and keeps the first. For the
lungs (`selection.num_objects: 2`) it also keeps the next most confident mask that comes from
a different point and overlaps the first by at most 10 %, and `merge` joins them. Ties go to the
earliest candidate, which makes the result deterministic.

### `evaluation/`: scoring (formula 6)

`metrics.py`: per image, Dice = 2|M ∩ G| / (|M| + |G|) and IoU, averaged over the images that
contain the target (this is the headline number), plus dataset-aggregate IoU / Dice and,
for the MedCLIP-SAMv2 datasets, NSD (the share of boundary within 2 px of the true boundary).
`runner.py` loops over a dataset (images are read ahead in background threads), writes
`metrics.json`, `per_image.csv` and the exact `config.yaml` of the run.

### The rest

| File | Role |
|---|---|
| `data/dataset.py` | finds `images/*` and the matching `masks/*`, reads them |
| `data/preprocessing.py` | the resize rule and CLIP / DINO normalisation |
| `models/clip_backbone.py` | builds open_clip ViT-L/14 and loads the BMC-CLIP weights |
| `tools/download_weights.py` | BMC-CLIP and MedSAM2 from the Hugging Face Hub |
| `tools/download_data.py`, `tools/prepare_data.py` | get every dataset and convert it to `images/` + `masks/` |
| `cli.py` | the `python -m zsmedseg ...` commands |
| `webapp/server.py`, `webapp/static/` | the demo website: FastAPI backend, plain HTML/CSS/JS page |
| `utils/visualization.py` | overlays and heat maps |

---

## 4. The formulas and where they are in the code

| # | Formula (slides) | Code |
|---|---|---|
| 1 | t_c = norm( mean over synonyms s of norm( mean over templates τ of g(τ(s)) ) ) | `models/text_encoder.py: encode_classes` |
| 2 | A_pq = softmax over q of τ_s · sim_DINO(p, q), only where R(p) = R(q); f_p = Σ_q A_pq v_q | `models/corrclip.py: _scope_attention` and `torch.bmm(weights, v)` in `forward` |
| 3 | P_p = exp(λ cos(f_p, t_fg)) / Σ_c exp(λ cos(f_p, t_c)), λ = 40 | `saliency/estimator.py`, last three lines of `__call__` |
| 4 | x_k = argmax of P outside the squares around x_1..x_{k-1}, k = 1..5, half-side 0.1 · min(H, W) | `prompting/strategies.py: MultiPeakPrompts.__call__` |
| 5 | M = M_{k*, m*} with (k*, m*) = argmax over k, m of predicted IoU s_{k,m} | `selection/rules.py: MaskSelector.choose` |
| 6 | Dice(M, G) = 2 \|M ∩ G\| / (\|M\| + \|G\|) | `evaluation/metrics.py: ImageRecord.dice` |

---

## 5. Settings and where the numbers come from

| Setting | Value | Origin |
|---|---|---|
| CLIP input short side, window, stride | 336, 336, 112 | BMC-CLIP's training resolution; stride = one third of a window |
| Softmax temperature λ | 40 | CorrCLIP's default |
| CorrCLIP τ, spatial weight, semantic weight, merge eps | 4, 1.0, 0.5, 0.2 | CorrCLIP paper defaults |
| Number of points | 5 | chosen on BUSI, frozen |
| Suppression half-side | 0.1 · min(H, W) | chosen on BUSI, frozen |
| MedSAM2 automatic grid | 16 x 16, thresholds 0.5 | chosen on BUSI, frozen |
| Objects | 1 (2 for both lungs) | the target's anatomy |

Every other dataset uses exactly these values; only the text prompts change.

---

## 6. Design decisions

* **Nothing is trained, nothing is fine-tuned.** All weights are loaded and frozen
  (`torch.inference_mode` everywhere).
* **Points, not boxes or the map itself.** The target map finds the right place but is too
  blurry and too large to be a mask, and a box drawn from it inherits that error. A point only
  has to land *inside* the target, which the map does well; MedSAM2 then does the precise part.
* **Unmodified libraries.** CorrCLIP's original code shipped a patched copy of open_clip and ran
  inside mmsegmentation. Here the modified forward pass lives in `models/corrclip.py` and only
  reads the standard model's weights, so `pip install open_clip_torch` is enough and it runs
  on Colab and Kaggle.
* **Exact reproduction.** The re-implementation was checked image by image against the original
  research code: the same predictions on every image.
* **One image at a time.** Images have different sizes and CorrCLIP builds one attention mask per
  image, so batching would not help; reading images is done ahead in background threads.
* **Models loaded once.** Changing the text (`set_prompts`) costs milliseconds, which is how the
  website and `evaluate_all.py` switch between tasks.

---

## 7. Questions you may be asked

**Is it really zero-shot?** Yes in the usual sense: no image of any test dataset, and no label,
was used to train or fine-tune anything. The few settings in section 5 marked "chosen on BUSI"
were picked once by looking at BUSI results and then kept fixed for every other dataset. Honest caveats: the pretrained models may have seen
similar public images, and "two lungs" is anatomical knowledge put into the config.

**Is it unsupervised?** Not exactly. The models were pretrained (BMC-CLIP on image-caption pairs,
MedSAM2 on labelled masks), but *we* use no labelled data. "Training-free and zero-shot" is
the precise term.

**Why not just threshold the target map?** It is blurry and spills over the edges (compare
panels B and D on the website). Points from the map only need to land inside the target, and
MedSAM2 then draws the sharp outline.

**Why five points?** One point fails whenever the strongest peak is on the wrong structure. Five
well-separated points give backups; more points add mostly noise candidates.

**Why trust MedSAM2's predicted IoU?** It is trained to estimate the quality of its own masks,
and it does not depend on the blurry map. Picking by it is simple and needs no labels.

**Why CorrCLIP and DINO?** CLIP's last attention layer mixes information from the whole image,
which ruins per-patch features. DINO is very good at grouping similar patches, and restricting
the mixing to similar patches of the same region gives clean features, without training.

**Where does it fail, and why?** When the target looks like its surroundings (brain tumours in
MRI), CLIP cannot point to it. In large, textured organs (lungs in CT, the heart in
echocardiography) one point makes MedSAM2 outline only a small part. And if no point lands on
the target, the result is wrong (all or nothing).

**How would you improve it?** Combine several candidates instead of picking one; use box prompts
for large organs; add a check for images without a target; a better medical CLIP would
directly improve the map.

**How long does it take?** About 1.5 s per image on a 4 GB laptop GPU (RTX 3050).

**How do I add a new dataset?** Put images and masks in two folders, run `prepare-data generic`,
copy `configs/datasets/_template.yaml` and write the target and background phrases. No code changes.
