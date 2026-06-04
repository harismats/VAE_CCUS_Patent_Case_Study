# CCUS Patent VAE Study

This repository contains the Google Colab notebook used for the experiments in *Calibrated Term-Presence Modelling for CCUS Patent Abstracts: A Case Study with Variational Autoencoders and Retrieval Baselines*. The notebook trains and evaluates calibrated term-presence models for CCUS patent abstracts, including VAE variants and retrieval baselines, and generates the figures and tables used in the manuscript.

## Repository contents

* `revised_ccus_patent_vae_colab_deterministic_extra_figures.ipynb` — end-to-end Colab notebook for preprocessing, training, evaluation and output generation.
* `patents_cleaned_subset_10k.csv` — expected input data file. If the data file is stored elsewhere, update `PROJECT_FOLDER` in the notebook before running.
* Generated folders, created automatically by the notebook:

  * `preprocessed/` — split matrices, vocabulary, metadata, completion masks and cached embeddings.
  * `figuresForPaper2/` — generated manuscript figures.
  * `tablesForPaper2/` — result tables, LaTeX tables and result macros.
  * `modelsForPaper2/` — saved model checkpoints.

## Computational environment

The notebook is designed to run in Google Colab. The reported run was executed in Google Colab using an NVIDIA A100 GPU. Other CUDA-enabled GPUs should also work, but runtimes and small floating-point differences may vary across hardware and Colab environments.

The first notebook cell installs the required Python packages, including NumPy, pandas, scikit-learn, PyTorch-related dependencies, Matplotlib, `sentence-transformers`, `transformers` and `accelerate`. Internet access is required if the transformer baselines need to download pretrained models.

## Data requirements

By default, the notebook expects the input CSV at:

```text
/content/drive/MyDrive/Colab Notebooks/Patents_Adventure/patents_cleaned_subset_10k.csv
```

The CSV must contain:

* `biblio` — bibliographic metadata containing CPC classifications.
* `processed_abstract` or `abstract_text` — patent abstract text.

If a `lang` column is present, the notebook keeps only rows where `lang` is equal to `en`.

## Running the notebook in Colab

1. Upload or open `revised_ccus_patent_vae_colab_deterministic_extra_figures.ipynb` in Google Colab.
2. Select a GPU runtime from **Runtime > Change runtime type**. The reported run used an NVIDIA A100 GPU.
3. Place `patents_cleaned_subset_10k.csv` in the expected Google Drive folder, or edit `PROJECT_FOLDER` in the configuration cell.
4. Run all notebook cells from top to bottom.
5. Keep the default flags for the full reviewer-facing run:

```python
RUN_NBVAE_GRID = True
RUN_TRANSFORMER_BASELINE = True
RUN_OPTIONAL_PAECTER = True
CLEAN_PREVIOUS_OUTPUTS = True
```

`CLEAN_PREVIOUS_OUTPUTS = True` removes old generated figures and tables before a new run. Set it to `False` only if previous outputs should be preserved.

## Main outputs

After a successful run, the main outputs are written to:

```text
figuresForPaper2/
tablesForPaper2/
preprocessed/
modelsForPaper2/
```

The `figuresForPaper2/` and `tablesForPaper2/` folders contain the main files needed for the manuscript and supplementary/reviewer-facing outputs. The `preprocessed/` folder stores intermediate matrices, metadata, masks and cached transformer embeddings so that repeated runs can reuse expensive preprocessing steps where applicable.

## Reproducibility notes

The notebook fixes the main random seeds, requests deterministic PyTorch behaviour where supported, disables cuDNN benchmarking, uses fixed train/validation/test splits and evaluates all models on the same document-completion masks. The reported configuration uses seed `10`, an 80:10:10 split, validation mask seed `11`, and test mask seed `12`.

The neural models are trained for at most 150 epochs with early stopping, batch size 64, gradient clipping at 50.0 and a 30-epoch KL warm-up. Exact bitwise reproducibility is not guaranteed across all GPU backends, but the notebook records and fixes the relevant settings used for the reported run. For the strictest reproducibility, CPU execution can be enabled by setting:

```python
FORCE_CPU_FOR_REPRODUCIBILITY = True
```

CPU execution is expected to be substantially slower.
