# PubMedRAG: Retrieval-Augmented Generation for Biomedical Question Answering

Large Language Models struggle with biomedical Q&A due to hallucinations and outdated knowledge. This research addresses these limitations by developing PubMedRAG, a domain-specific retriever model trained using Simple Contrastive Sentence Embeddings (SimCSE) on the PubMedQA labeled dataset.

## Key Results

- **64% accuracy** with Llama3-OpenBioLLM-8B
- **61% accuracy** with Llama-3.1-8B  
- Significant improvement over baseline models in biomedical question answering

## Installation

```bash
git clone https://github.com/HamsiniGupta/PubMedRAG
cd PubMedRAG/PubMedRAG_Files
pip install -r requirements.txt
```

## Dataset Format
The PubMedQA dataset should be formatted for SimCSE training:
- Columns: sent0, sent1, label
- Labels: Positive pairs (1), Negative pairs (0)

## Preprocessing Data
```bash
# Creates negative and positive pairs for training, generates test, train, and validation .csv files
python preprocess.py
```

## Training
```bash
# Trains BERT model with training dataset and 5 epochs
python train.py --train_file data/pubmedqa_train_supervised.csv --epochs 5
```
## Evaluation
```bash
#Get results for PubMedRAG
python testPubMedRAG.py
# Compare PubMedRAG and BERT and evaluate results
python compareAllEmbeddings.py
```
The following table shows the results for different retrievers in the RAG pipeline evaluated on both LLMs.
<img width="792" height="397" alt="image" src="https://github.com/user-attachments/assets/9c7a05cd-4c15-4136-a39a-6d7a07ae4b38" />


## Acknowledgements
This work was supported by the NSF grants #CNS-2349663 and #OAC-2528533. This work used Indiana JetStream2 GPU at Indiana University through allocation NAIRR250048 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by the NSF grants #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed in this work are those of the author(s) and do not necessarily reflect the views of the NSF.
