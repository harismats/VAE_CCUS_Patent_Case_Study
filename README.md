# CCUS Patent VAE Study

This repository contains the reproducible Google Colab workflow for the paper *Calibrated Term-Presence Models for Carbon Capture, Utilisation and Storage Patent Abstracts*. The workflow prepares the patent corpus, trains the reported models, evaluates document completion, calibration and Cooperative Patent Classification prefix retrieval and runs the conditional posterior predictive sparsity study.

The notebook is the canonical executable source. The Python file is generated from the same notebook to make the code easier to search, compare and inspect.

## Main files

- `ccus_patent_vae_colab_with_conditional_ppc.ipynb` contains the complete Colab workflow.
- `ccus_patent_vae_colab_with_conditional_ppc.py` is a generated inspection copy of the notebook code.
- `patents_cleaned_subset_10k.zip` contains the study dataset. Extract `patents_cleaned_subset_10k.csv` before running the notebook.

Do not edit the notebook and Python copy independently. Regenerate the Python file after changing the notebook.

## Data and preprocessing

The default input path is:

```text
/content/drive/MyDrive/Colab Notebooks/Patents_Adventure/patents_cleaned_subset_10k.csv
```

The notebook extracts Cooperative Patent Classification symbols from `biblio`. It uses `processed_abstract` when that column is available and otherwise uses `abstract_text`. The reported run used `abstract_text`, as recorded in `run_manifest.json`. Records are restricted to English when `lang` is available.

The carbon capture, utilisation and storage corpus retains records with a classification symbol beginning with one of the following prefixes:

```text
Y02C
B01D53/04
B01D53/46
B01D53/62
```

The reported corpus contains 10,603 documents. It is divided into 8,481 training documents, 1,061 validation documents and 1,061 test documents. The vocabulary is fitted on the training split only and contains 4,412 terms.

## Running in Google Colab

1. Extract `patents_cleaned_subset_10k.csv` from the dataset archive.
2. Place the CSV in the default Google Drive folder or change `PROJECT_FOLDER` in the configuration cell.
3. Open `ccus_patent_vae_colab_with_conditional_ppc.ipynb` in Google Colab.
4. Select an NVIDIA A100 GPU runtime.
5. Run every cell from top to bottom.

Use the following settings for the full manuscript run:

```python
REQUIRE_CUDA = True
CLEAN_PREVIOUS_OUTPUTS = True
RUN_NBVAE_GRID = True
RUN_PATENTSBERTA = True
RUN_PAECTER = True
RUN_CONDITIONAL_PPC = True
SIMULATION_DRAWS = 1000
DOWNLOAD_RESULTS_BUNDLE = True
```

A smaller number of posterior predictive draws may be used for a development check, but those outputs should not be used for the reported analysis.

## Models and evaluation

The notebook evaluates:

- negative binomial to Bernoulli variational autoencoder
- direct Bernoulli variational autoencoder
- ProdLDA
- unigram prior
- term frequency inverse document frequency nearest neighbours
- PatentSBERTa nearest neighbours
- PaECTER nearest neighbours

Document completion uses fixed validation and test masks. A target masking rate of 50 percent is applied to the positive terms in each eligible document while at least one positive term remains observed. Calibration is evaluated on the unobserved vocabulary coordinates.

Cooperative Patent Classification prefix retrieval is evaluated for `B01D53/04` and `B01D53/62`. Each representation is passed to the same class balanced logistic regression procedure.

The conditional posterior predictive study uses 1,000 draws for each of the two variational autoencoders. It compares predicted and observed positive term counts at document and vocabulary levels. Its intervals condition on the fitted model weights, data split and fixed test mask. They are not confidence intervals.

## Generated outputs

The notebook creates:

- `preprocessed/` for split matrices, vocabulary, metadata, completion masks and validated embedding caches
- `figuresForPaper/` for manuscript and appendix figures
- `tablesForPaper/` for result tables, detailed CSV files and LaTeX outputs
- `simulationForPaper/` for posterior predictive outputs, compressed draw summaries, the protocol, requirements and run manifest

The final cell validates and downloads `ccus_revision_results_bundle.zip`. This archive contains the regenerated figures, tables and simulation files. It does not contain the source patent data, preprocessed caches or model checkpoints.

## Reproducibility

The workflow fixes the data split, completion mask, model and simulation seeds. It enables deterministic PyTorch behaviour where supported and keeps the held out test split separate from model selection.

Transformer caches are reused only when the model revision and input text hash match. The run manifest records the input data hash, vocabulary hash, selected neural configuration, model checkpoint hashes, transformer revisions, simulation signatures, package versions and hardware information.

The reported conditional posterior predictive run used an NVIDIA A100 GPU.

