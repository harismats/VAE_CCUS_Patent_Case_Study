# CCUS Patent VAE Study

This repository contains the Google Colab notebook used for the experiments in the paper *Calibrated Term Presence Modelling for CCUS Patent Abstracts: A Case Study with Variational Autoencoders and Retrieval Baselines*. The notebook processes the study dataset, trains the reported models, evaluates document completion, calibration and CPC prefix retrieval and generates the manuscript outputs.

## Files

`ccus_patent_vae_colab.ipynb`: Colab notebook for preprocessing, training, evaluation and output generation.

`patents_cleaned_subset_10k.zip`: Compressed study dataset used in the reported analysis. Please unzip this file to extract `patents_cleaned_subset_10k.csv` before running the notebook.

`preprocessed/`: Split matrices, vocabulary, metadata, fixed completion masks and cached embeddings.

`figuresForPaper2/`: Generated manuscript figures.

`tablesForPaper2/`: Generated result tables and LaTeX outputs.

`modelsForPaper2/`: Saved model checkpoints.

## Data and processing

The reported analysis reads:

```text
/content/drive/MyDrive/Colab Notebooks/Patents_Adventure/patents_cleaned_subset_10k.csv
```

The notebook extracts CPC symbols from `biblio` and uses abstract text from `processed_abstract`, with `abstract_text` included as a fallback for equivalent exports. English language records are filtered when `lang` is available. The CCUS corpus retains records with CPC symbols beginning with `Y02C`, `B01D53/04`, `B01D53/46` or `B01D53/62`.

The reported filtered corpus contains 10,603 documents, split into 8,481 training documents, 1,061 validation documents and 1,061 test documents. The final vocabulary contains 4,412 terms.

## Running in Colab

Step 1: Open the notebook in Google Colab.

Step 2: Select a GPU runtime. The reported run used an NVIDIA A100 GPU.

Step 3: Place `patents_cleaned_subset_10k.csv` in the Google Drive folder above, or update `PROJECT_FOLDER` in the configuration cell.

Step 4: Run the notebook cells from top to bottom.

Step 5: Keep the following flags enabled for the full manuscript run:

```python
RUN_NBVAE_GRID = True
RUN_TRANSFORMER_BASELINE = True
RUN_OPTIONAL_PAECTER = True
CLEAN_PREVIOUS_OUTPUTS = True
```

## Models and evaluation

The notebook evaluates the NB to Bernoulli VAE, direct Bernoulli VAE, ProdLDA, unigram prior, TF IDF nearest neighbours, PatentSBERTa nearest neighbours and PaECTER nearest neighbours. Document completion uses fixed validation and test masks. Calibration is evaluated on unobserved document term pairs. CPC prefix retrieval is evaluated for `B01D53/04` and `B01D53/62`.

## Reproducibility

The notebook fixes the main random seeds, uses deterministic PyTorch settings where supported, creates fixed train, validation and test splits, and evaluates all models on fixed document completion masks. Model selection uses the validation split. The held out test split is kept separate from model selection and is used for final reporting.
