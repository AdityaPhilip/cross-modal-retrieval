# Attention-driven cross-modal retrieval: EuroSAT RGB <-> SAR

Course project, ECPE49 Foundations of Artificial Intelligence (NIT Tiruchirappalli, July 2026).
Implements an adaptation of **Att-CMIR-Net** (Chaudhuri, Banerjee, Bhattacharya, Datcu, *Attention-Driven
Cross-Modal Remote Sensing Image Retrieval*, IGARSS 2021) for **RGB query -> SAR gallery** and
**SAR query -> RGB gallery** retrieval on EuroSAT, evaluated with mAP.

> **Status of this package.** The data, split, pairing and retrieval/mAP code was executed and tested on
> synthetic data. The PyTorch parts (model, losses, `train.py`, `test.py`) were written and statically
> checked but **not executed** (the authoring environment had no PyTorch and no EuroSAT data).
> No trained model and no real results ship with the source package: you produce them with the commands below.
> See [Verification status](#verification-status).

## 1. Quick start (Windows, from this folder)

Commands are single lines for `cmd.exe`/PowerShell. Replace paths with yours.

| Step | Command |
|---|---|
| **Train** (one line) | `python train.py --rgb_root "C:\Users\adity\Downloads\EuroSAT_RGB\EuroSAT_RGB" --sar_root "C:\Users\adity\Downloads\EuroSAT-SAR\EuroSAT-SAR"` |
| **Test** (one line) | `python test.py` |

If the datasets are extracted to `data\EuroSAT_RGB` and `data\EuroSAT-SAR` inside this folder (or the environment
variables `EUROSAT_RGB_ROOT` / `EUROSAT_SAR_ROOT` are set), the training command is simply `python train.py`.

* `train.py` trains, validates every epoch, saves `checkpoints/best.pt` (best validation mAP) and `last.pt`, writes
  `plots/loss_curves.png`, `plots/loss_components.png`, `plots/val_map.png`, `plots/history.json`, `reports/train_summary.json`,
  and finally evaluates the best checkpoint on the test split (`reports/test_results_after_training.json`).
  **It overwrites `checkpoints/best.pt`** (this is what the evaluator's step E2 expects); back up the submitted model first if you need it.
* `test.py` needs only `splits/test.csv`, `test_dataset/` and `checkpoints/best.pt`. It does not need the training data and does not
  download anything. It prints mAP / P@10 / P@20 / P@50 for RGB->SAR and SAR->RGB (plus same-modality retrieval and a chance-level
  reference) and writes `reports/test_results.json`.

## 2. Datasets and organisation

* EuroSAT RGB (and MS): <https://zenodo.org/records/7711810>, 27,000 Sentinel-2 patches, 64x64 px, 10 classes.
* EuroSAT-SAR: <https://huggingface.co/datasets/wangyi111/EuroSAT-SAR>, 27,000 dual-polarisation Sentinel-1 GeoTIFFs.

Both have `<root>/<ClassName>/<ClassName>_<n>.jpg|tif`. The code accepts a wrapper folder (`EuroSAT_RGB\EuroSAT_RGB`).

**Pairing.** The EuroSAT-SAR dataset description states that each EuroSAT Sentinel-2 image was matched with one Sentinel-1 patch
according to geospatial coordinates, with the same file structure as EuroSAT. Samples are therefore paired by *(class folder, filename stem)*.
A check on the author's copy found 27,000 matching stems and no unmatched files. What the documentation does **not** establish is
pixel-level alignment or same-day acquisition, so the pairs are treated as geographically matched, not pixel-registered.
`python inspect_data.py --rgb_root ... --sar_root ...` re-checks the file matching, prints the SAR array shape/dtype/value range
and runs a weak empirical check (true pairs vs same-class shuffled pairs); a null result there is inconclusive, not a refutation.

**Split.** `python make_splits.py --rgb_root ... --sar_root ...` creates `splits/{train,val,test}.csv` + `classes.json`:
stratified by class, 70/15/15, `numpy.default_rng(42)` over a naturally sorted file list, performed over *pairs* (the RGB and SAR
member of a pair are always in the same split). It is deterministic: re-running on the same data reproduces identical CSVs.
`train.py` creates the split automatically if the CSVs are missing. **Commit/ship the CSVs** you used.

**Test data for the ZIP.** `python prepare_test_dataset.py --rgb_root ... --sar_root ...` copies only the test-split pairs to
`test_dataset/RGB/<class>/...` and `test_dataset/SAR/<class>/...` (about 4,050 pairs; exact size is printed). The full training data is
not packaged; the evaluator needs the EuroSAT downloads for the training command (place at `data\EuroSAT_RGB`, `data\EuroSAT-SAR`).

## 3. Environment setup (Windows 11, Python 3.11, CUDA GPU)

Your current PyTorch is CPU-only (`2.12.0+cpu`), so install a CUDA build in a fresh Python 3.11 environment:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
:: copy the exact CUDA command for your setup from https://pytorch.org/get-started/locally/  (example below)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The last line must print `True`; `nvidia-smi` already shows your driver. The first `train.py` run downloads the ImageNet VGG-16 weights
(about 528 MB) through torchvision; `test.py` never downloads weights. The code also runs on CPU (slowly).
The code targets Python 3.11; I did not test other versions.

## 4. Recommended workflow

```bat
python tests\run_tests.py                         :: unit tests (dependency-free runner; "pytest tests" also works)
python inspect_data.py --rgb_root "..." --sar_root "..."
python make_splits.py  --rgb_root "..." --sar_root "..."
python train.py --rgb_root "..." --sar_root "..." --epochs 1 --max_train 2000 --max_val 500 --exp_name smoke --no_final_test   :: smoke run
python train.py --rgb_root "..." --sar_root "..."                                                                              :: full run
python prepare_test_dataset.py --rgb_root "..." --sar_root "..."
python test.py
:: optional ablation (same settings, no attention) for the report:
python train.py --rgb_root "..." --sar_root "..." --exp_name no_attention --set model.attention=false
python test.py --checkpoint checkpoints\no_attention\best.pt --output reports\no_attention\test_results.json
python make_report.py --group "Group X: names"
python make_submission.py                         :: builds ..\cross_modal_retrieval_submission.zip
```

The smoke run prints seconds per epoch, which lets you estimate the full run time (I have not timed it on an RTX 4050).
If you see CUDA out-of-memory, use `--batch_size 32` (or `--set model.input_size=96`).

## 5. Directory structure

```
cross_modal_retrieval/
  README.md  requirements.txt
  train.py  test.py                      entry points (separate files, no notebooks)
  model.py  losses.py  dataset.py  retrieval.py  utils.py
  make_splits.py  inspect_data.py  prepare_test_dataset.py  make_report.py  make_submission.py
  configs/default.yaml                   all hyper-parameters
  splits/  train.csv val.csv test.csv classes.json
  checkpoints/best.pt (last.pt)          created by train.py
  plots/   loss_curves.png loss_components.png val_map.png history.json
  reports/ report.pdf train_summary.json test_results.json test_results_after_training.json
  test_dataset/RGB/<class>/*.jpg  test_dataset/SAR/<class>/*.tif
  tests/   test_retrieval.py test_data.py test_model_losses.py test_pipeline_smoke.py synth.py run_tests.py
```

## 6. Configuration

`configs/default.yaml`; override any entry with `--set section.key=value` (repeatable) plus the shortcuts
`--rgb_root --sar_root --epochs --batch_size --seed --device --exp_name`.

| Key | Default | Meaning |
|---|---|---|
| `seed` | 42 | split, initialisation of new layers, shuffling, augmentation |
| `data.pairing` | `instance` | `instance`: true (class, stem) pairs; `class`: random same-class SAR partner per RGB sample (use if you doubt the pairing) |
| `model.input_size` | 128 | 64 px patches upsampled; VGG map 512x4x4 |
| `model.attention` | true | `false` = ablation without SAR-derived attention |
| `loss.reduction`, `loss.w1..w4`, `loss.aux` | mean, 1.0 | loss weighting (see below) |
| `train.epochs / batch_size / lr` | 50 / 64 / 0.001 | paper: ~40-50 / 128 / 0.001 |
| `train.amp` | true | mixed precision on CUDA |
| `test.l2_normalize` | false | paper uses plain Euclidean distance (`test.py --l2norm` to try it) |

## 7. Evaluation protocol, mAP and P@10

* **Relevance:** a gallery item is relevant to a query iff it has the same class label (class-level, not instance-level).
* **Ranking:** ascending Euclidean distance between 128-d embeddings.
* **AP** of a query = mean of Precision@k over the ranks k of its relevant items in the *full* ranking; **mAP** = mean over queries.
  **P@K** = fraction of relevant items in the top K. All values are fractions in [0, 1]; chance is about 0.10 (10 balanced classes).
* **RGB->SAR:** RGB test images are queries, SAR test images the gallery; **SAR->RGB** the reverse. Test queries = gallery size = test pairs.
* **Standard vs strict:** in the headline cross-modal numbers the query's own paired counterpart is in the gallery (it is a legitimate
  same-class item). The `(strict)` rows remove it, so no instance-level correspondence can help.
* **Same-modality** retrieval (RGB->RGB, SAR->SAR) excludes the query itself. Queries without any relevant item are skipped and counted.
* A random-embedding row reports the chance level. Nothing is hard-coded; the paper's numbers appear only in the report, labelled as published.

## 8. Method: what follows the paper and what does not

**From the paper:** two ImageNet-pretrained VGG-16 conv encoders; attention = GAP of the last conv map of the attention-source encoder ->
FC -> sigmoid, multiplied onto the other modality's FC features; shared 128-d space; losses L1 (alignment), L2 (cross-entropy on both modalities),
L3 (norm), L4 (cross-decoders), L = L1+L2+L3+L4; SGD, lr 0.001; Euclidean kNN retrieval; mAP/P@K.

**Adaptations and assumptions (not stated in the paper):**

1. **Dataset / modalities:** RGB and SAR replace photo and sketch, following the assignment's dataset list. SAR plays the "attention source"
   role of the sketch by analogy only. Absolute mAP values are **not comparable** with the paper's (different task, classes, chance level).
2. **Input:** 64 px patches upsampled to 128 px (4x4x512 maps instead of 7x7x512). SAR (2 channels): first VGG conv rebuilt from ImageNet filters averaged over RGB.
3. **Unspecified details chosen by us:** FC head 512 units (+ReLU, dropout 0.3); shared 512-256-128 MLP for the embedding; `V_ab` realised as the two
   modalities' embeddings stacked along the batch with one shared linear classifier; `w_ab`, `w_ba` as linear layers.
4. **Inference without a pair:** the paper does not say how the photo is attended when no sketch is paired with it. Here each modality is embedded
   independently (no counterpart needed). To transfer the attention benefit to this path, training adds an **auxiliary term** (cross-entropy on the
   pair-free RGB path + distillation towards the attended embedding, stop-gradient). This is our design, not the paper's.
5. **Loss reduction:** squared errors are averaged over batch and dimensions by default (`loss.reduction: sum` gives the literal Frobenius norm).
   Taken literally, L1 and L3 have the trivial minimum V = 0, which only L2 resists; weights are configurable and were not tuned.
6. **Hardware-driven:** batch 64 (paper 128), mixed precision, 128 px input.
7. Augmentation (random flips/transpose, independent per modality) and SAR robust normalisation (train-split percentile clipping + standardisation, stored in the checkpoint) are our additions.

## 9. Reproducibility

Fixed `seed` for splits (`numpy.default_rng`), Python/NumPy/PyTorch RNGs, weight initialisation (new layers), shuffling and augmentation;
`cudnn.deterministic=True`, `use_deterministic_algorithms(warn_only=True)`; seed, config, class list, SAR statistics and environment versions
are stored in every checkpoint. Remaining non-determinism: mixed-precision accumulation, some CUDA kernels, different GPU/driver/torch
versions, and thread scheduling. Expect re-training to give close but not bit-identical mAP (the evaluator's E1 vs E2 comparison). On CPU with
the same torch build, results should match much more closely.

## 10. Verification status

| Component | Status |
|---|---|
| mAP / P@K / ranking / self-exclusion (hand-computed cases, brute-force cross-check) | **Executed, passing** (authoring sandbox) |
| Pair discovery, mismatch detection, deterministic stratified split, leakage check, loading, caching, SAR stats, scripts `make_splits` / `inspect_data` / `prepare_test_dataset` on synthetic EuroSAT-like data | **Executed, passing** |
| `make_report.py` (with mock numbers and with missing inputs) | **Executed** |
| Static undefined-name scan of all modules | Executed, clean |
| Model forward shapes, attention used, each loss term, gradients, checkpoint round-trip (`tests/test_model_losses.py`) | **Written, NOT executed** |
| `train.py` -> `test.py` end-to-end smoke test on synthetic data (`tests/test_pipeline_smoke.py`) | **Written, NOT executed** |
| Anything on real EuroSAT data, GPU behaviour, VRAM use, run time, achieved mAP | **Not run** |

Please run `python tests\run_tests.py` first; the two torch test files are the first thing that will reveal any bug in the PyTorch code.

## 11. Known limitations

Dataset differs from the paper (no claim of reproducing its numbers); several architectural details are assumptions; single seed per configuration (no
variance estimate); the geographic pairing is documented but pixel alignment / acquisition date are not verified; EuroSAT patches from nearby
locations may be similar across splits; loss weights untuned; the attention effect on the pair-free path is indirect (see 8.4).
