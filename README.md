# PubMedBERT: Retrieval-Augmented Generation for Biomedical Question Answering

Large Language Models struggle with biomedical Q&A due to hallucinations and outdated knowledge. This research addresses these limitations by developing PubMedBERT, a domain-specific retriever model trained using Simple Contrastive Sentence Embeddings (SimCSE) on the PubMedQA labeled dataset.

## Key Results

- **64% accuracy** with Llama3-OpenBioLLM-8B
- **61% accuracy** with Llama-3.1-8B  
- Significant improvement over baseline models in biomedical question answering

## Installation

```bash
git clone https://github.com/HamsiniGupta/PubMedBERT
cd PubMedBERT/sentence_transformers
pip install -r requirements.txt
```

# Dataset Format
The PubMedQA dataset should be formatted for SimCSE training:
Columns: sent0, sent1, label
Labels: Positive pairs (1), Negative pairs (0)

# Preprocessing Data
```bash
# Creates negative and positive pairs for training, generates test, train, and validation .csv files
python preprocess.py
```

# Training
```bash
# Trains BERT model with training dataset and 5 epochs
python train.py --train_file data/pubmedqa_train_supervised.csv --epochs 5
```
# Evaluation
```bash
# Compare PubMedBERT and BERT and evaluate results
python compareAllEmbeddings.py
```

# Acknowledgements
This work was supported by the NSF grants #CNS-2349663 and #OAC-2528533. This work used Indiana JetStream2 GPU at Indiana University through allocation NAIRR250048 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by the NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed in this work are those of the author(s) and do not necessarily reflect the views of the NSF.
