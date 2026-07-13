# OS_WORC

WORC-based radiomics pipeline for **Osteosarcoma treatment-response prediction**.

The project uses [WORC](https://github.com/MStarmans91/WORC) (Workflow for Optimal
Radiomics Classification) to predict chemotherapy response from MRI scans and
clinical features. The main outcome is the **Huvos** grade (histological response
to neo-adjuvant chemotherapy), with additional experiments on the **WIR** label
and on patient subgroups (age group, MRI modality combination, tumour subtype,
tumour location).

## Pipeline overview

```
raw data (DICOM / NIfTI + clinical SPSS)
   │  src/data_preparation, src/clinical
   ▼
experiment data  (image.nii.gz, mask.nii.gz, clinical_features*.csv, patient_splits.csv)
   │  src/exp
   ▼
WORC experiment  (src/exp/simple_worc.py → SimpleWORC.execute())
   │
   ▼
results  (performance_*.json, features_*.hdf5)
   │  src/analysis, src/visualization, src/statistics
   ▼
metrics, ROC curves, comparison tables, figures
```

## Repository layout

| Folder | Purpose |
| --- | --- |
| [src/data_preparation/](src/data_preparation/) | Download and stage image data (local + remote SHARK/Snellius servers), build WIR datasets, create dummy data. |
| [src/clinical/](src/clinical/) | Clean and convert clinical data (SPSS → CSV), build demographic / baseline tables, run simple ML baselines on clinical features. |
| [src/exp/](src/exp/) | The core experiments. `simple_worc.py` runs a WORC experiment; `helper.py` / `clinical_feature.py` build the per-experiment image/label/split inputs and the clinical feature files; `check_dup.py` checks for duplicates. |
| [src/analysis/](src/analysis/) | Aggregate and combine results: subject-level scoring from image-level predictions, combine performance across experiments, statistical comparison of experiments, custom estimator utilities. |
| [src/visualization/](src/visualization/) | Plots and figures: ROC curves, Mann–Whitney feature analysis, segmentation overlays. |
| [src/statistics/](src/statistics/) | `ClinicalFeatureAnalyzer` for statistical analysis of clinical features. |
| [src/user/](src/user/) | Segmentation quality tools: Dice overlap between segmentation versions, visualization helpers, segmentation-selection utilities. |
| [src/meta/](src/meta/) | Metadata bookkeeping: merge metadata sources, sanity-check experiment data. |
| [src/interactive/](src/interactive/) | Ad-hoc helpers for exploring patient records / history. |

## Running an experiment

The main entry point is [src/exp/simple_worc.py](src/exp/simple_worc.py):

```bash
python src/exp/simple_worc.py \
    --exp_name my_experiment \
    --modality T1W \
    --version v0 \
    --mode 1 \
    --label_type Huvos \
    --overfit 0
```

### Arguments

| Argument | Description |
| --- | --- |
| `--exp_name` | Name of the experiment (used for output folders). |
| `--modality` | MRI modality, e.g. `T1W`, `T2W_FS`, `T1W_FS_C`. |
| `--version` | Segmentation version (`v0`, `v1`, `v2`). |
| `--mode` | Feature configuration (see below). |
| `--label_type` | Outcome / cohort to predict (see below). |
| `--overfit` | `1` to build overfit splits (sanity check), `0` for normal splits. |

### `--mode` (feature groups, see `editconfig`)

| Mode | Features used |
| --- | --- |
| `0` | Radiomics only, no semantic (clinical) features |
| `1` | Radiomics + semantic features |
| `2` | Semantic + original features only |
| `3` | Histogram + semantic + original features |
| `4` | Semantic + wavelet + log features |
| `5` | Wavelet features only |

### `--label_type`

- `Huvos` — Huvos response grade (main outcome, `Huvosnew` label).
- `WIR` — WIR label.
- `Children` / `AYA` / `Older_adults` — age-subgroup cohorts (Huvos outcome).
- `T1W+T1W_FS_C`, `T1W+T2W_FS`, `T2W_FS+T1W_FS_C`, `T1W+T1W_FS_C+T2W_FS` — multi-modality cohorts.
- `Conventional_OS` — conventional osteosarcoma subtype cohort.
- `femur` — tumour-location (femur) cohort.

## Data layout expected by the experiment

Each experiment reads from a directory containing per-patient:

- `image.nii.gz` — the MRI image
- `mask.nii.gz` — the tumour segmentation
- `clinical_features.csv` — semantic features passed to WORC
- `clinical_features_with_<Huvos|WIR>.csv` — labels
- `patient_splits.csv` — fixed cross-validation splits

## Output

WORC writes results to its configured output mount under `WORC_<exp_name>/`:

- `performance_all_0.json` — aggregated performance statistics
- `Features/features_*.hdf5` — extracted radiomics features per patient

Use the scripts in [src/analysis/](src/analysis/) and
[src/visualization/](src/visualization/) to combine and plot these results.

## Notes

- Paths in the scripts are hard-coded for the Snellius / SHARK compute
  environment (e.g. `/projects/0/prjs1425/...`, `/scratch-shared/xwan/...`);
  adjust them for your own setup.
- WORC must be installed / on the path
  (see the `sys.path.append(...)` line in the experiment scripts).
