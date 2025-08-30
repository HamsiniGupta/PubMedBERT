---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:8144
- loss:ContrastiveLoss
base_model: google-bert/bert-base-uncased
widget:
- source_sentence: Are sports medicine journals relevant and applicable to practitioners
    and athletes?
  sentences:
  - Studies on coronary risk factors in men and women are mainly based on mortality
    data and few compare results of both sexes with consistent study design and diagnostic
    criteria. This study assesses the major risk factors for coronary events in men
    and women from the Reykjavik Study.
  - There is a dearth of studies addressing diagnostic and treatment interventions
    in the sports medicine literature. The evidence base for sports medicine must
    continue to increase in terms of volume and quality.
  - Hypotheses for this discontent are presented. Physicians may be uninterested in
    helping caregivers; even if they were receptive to counseling caregivers, they
    could be poorly remunerated for the types of counseling sessions that are usual
    for caregivers; and being a professional caregiver to family caregivers is demanding
    in itself.
- source_sentence: Is Bare-Metal Stent Implantation Still Justifiable in High Bleeding
    Risk Patients Undergoing Percutaneous Coronary Intervention?
  sentences:
  - 'Compared with patients without, those with 1 or more HBR criteria had worse outcomes,
    owing to higher ischemic and bleeding risks. Among HBR patients, major adverse
    cardiovascular events occurred in 22.6% of the E-ZES and 29% of the BMS patients
    (hazard ratio: 0.75; 95% confidence interval: 0.57 to 0.98; p = 0.033), driven
    by lower myocardial infarction (3.5% vs. 10.4%; p<0.001) and target vessel revascularization
    (5.9% vs. 11.4%; p = 0.005) rates in the E-ZES arm. The composite of definite
    or probable stent thrombosis was significantly reduced in E-ZES recipients, whereas
    bleeding events did not differ between stent groups.'
  - MDA on its own was insufficient to control the prevalence of schistosomiasis,
    intensity of Schistosoma infection, or morbidity of the disease. Alternative control
    measures will be needed to complement the existing national MDA program.
  - A task-specific intervention designed to improve gait speed may potentially provide
    secondary benefits by positively impacting depression, mobility and social participation
    for people post stroke.
- source_sentence: 'In vivo visualization of pyloric mucosal hypertrophy in infants
    with hypertrophic pyloric stenosis: is there an etiologic role?'
  sentences:
  - 3D ultrasound validation of the postfiring needle position is an efficient adjunct
    to ultrasound-guided LCNB. The advantages of 3D ultrasound validation are likely
    to include a reduction in the number of core samples needed to achieve a reliable
    histological diagnosis (and a possible reduction in the risk of tumor cell displacement),
    reduced procedure time and lower costs.
  - We identified 102 consecutive infants with surgically confirmed IHPS and determined
    the thickness of the pyloric mucosa compared with the thickness of the surrounding
    hypertrophied muscle. Fifty-one infants who did not have pyloric stenosis served
    as controls.
  - Broad-based electronic health information exchange (HIE), in which patients' clinical
    data follow them between care delivery settings, is expected to produce large
    quality gains and cost savings. Although these benefits are assumed to result
    from reducing redundant care, there is limited supporting empirical evidence.
- source_sentence: Does pretreatment with statins improve clinical outcome after stroke?
  sentences:
  - There is no standard protocol for the evaluation of antiseptics used for skin
    and mucous membranes in the presence of interfering substances. Our objective
    was to suggest trial conditions adapted from the NF EN 13727 standard, for the
    evaluation of antiseptics used in gynecology and dermatology.
  - Post-operative version or percentage of DFV>15° did not significantly differ following
    IMN of diaphyseal femur fractures between surgeons with and without trauma fellowship
    training. However, prospective data that removes the inherent bias that the more
    complex cases are left for the traumatologists are required before a definitive
    comparison is made.
  - The statistical power of this case-referent study was such that only large beneficial
    effects of statins in acute stroke could be confirmed. However, the observed trend,
    together with experimental observations, is interesting enough to warrant a more
    detailed analysis of the relationship between statins and stroke outcome.
- source_sentence: The aim of this study was to describe the evolution and epidemiologic
    characteristics of shigellosis patients over a 25 year period in a large city.
  sentences:
  - There was no difference between the two groups in the number of emboli detected
    (p=0.49) and no significant correlation between number of emboli and dissection
    time (r=0.0008). However, there was a significantly higher number of emboli in
    the patient sub-group that were current smokers (p=0.034).
  - An increased trend was detected in men who had no history of food poisoning or
    travel to endemic areas. This increase points to a change in the pattern of shigellosis,
    becoming predominantly male and its main mechanism probably by sexual transmission.
  - There is a trend for progressively increasing mean intra-osseous length associated
    with increased flexion of the knee. The mean intra-osseous length for 70° flexion
    was 25.2 mm (20 mm to 32 mm), which was statistically significant when compared
    to mean intra-osseous lengths of 32.1 mm (22 mm to 45 mm) and 38.0 mm (34 mm to
    45 mm) in the 90° and 120° flexion groups, respectively (p<0.05). There were no
    significant differences among the groups with respect to distance to the LCL.
    There is a trend toward longer distances to the common peroneal nerve with increased
    flexion. There was a statistically significant dif - ference when comparing 120°
    versus 70° (p<0.05).
pipeline_tag: sentence-similarity
library_name: sentence-transformers
metrics:
- cosine_accuracy
- cosine_accuracy_threshold
- cosine_f1
- cosine_f1_threshold
- cosine_precision
- cosine_recall
- cosine_ap
- cosine_mcc
model-index:
- name: SentenceTransformer based on google-bert/bert-base-uncased
  results:
  - task:
      type: binary-classification
      name: Binary Classification
    dataset:
      name: validation
      type: validation
    metrics:
    - type: cosine_accuracy
      value: 0.9347826086956522
      name: Cosine Accuracy
    - type: cosine_accuracy_threshold
      value: 0.7773971557617188
      name: Cosine Accuracy Threshold
    - type: cosine_f1
      value: 0.9344
      name: Cosine F1
    - type: cosine_f1_threshold
      value: 0.6629912257194519
      name: Cosine F1 Threshold
    - type: cosine_precision
      value: 0.8957055214723927
      name: Cosine Precision
    - type: cosine_recall
      value: 0.9765886287625418
      name: Cosine Recall
    - type: cosine_ap
      value: 0.984640914650542
      name: Cosine Ap
    - type: cosine_mcc
      value: 0.8664159803630758
      name: Cosine Mcc
---

# SentenceTransformer based on google-bert/bert-base-uncased

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased). It maps sentences & paragraphs to a 768-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) <!-- at revision 86b5e0934494bd15c9632b12f734a8a67f723594 -->
- **Maximum Sequence Length:** 256 tokens
- **Output Dimensionality:** 768 dimensions
- **Similarity Function:** Cosine Similarity
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'The aim of this study was to describe the evolution and epidemiologic characteristics of shigellosis patients over a 25 year period in a large city.',
    'An increased trend was detected in men who had no history of food poisoning or travel to endemic areas. This increase points to a change in the pattern of shigellosis, becoming predominantly male and its main mechanism probably by sexual transmission.',
    'There is a trend for progressively increasing mean intra-osseous length associated with increased flexion of the knee. The mean intra-osseous length for 70° flexion was 25.2 mm (20 mm to 32 mm), which was statistically significant when compared to mean intra-osseous lengths of 32.1 mm (22 mm to 45 mm) and 38.0 mm (34 mm to 45 mm) in the 90° and 120° flexion groups, respectively (p<0.05). There were no significant differences among the groups with respect to distance to the LCL. There is a trend toward longer distances to the common peroneal nerve with increased flexion. There was a statistically significant dif - ference when comparing 120° versus 70° (p<0.05).',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 768]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Binary Classification

* Dataset: `validation`
* Evaluated with [<code>BinaryClassificationEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.BinaryClassificationEvaluator)

| Metric                    | Value      |
|:--------------------------|:-----------|
| cosine_accuracy           | 0.9348     |
| cosine_accuracy_threshold | 0.7774     |
| cosine_f1                 | 0.9344     |
| cosine_f1_threshold       | 0.663      |
| cosine_precision          | 0.8957     |
| cosine_recall             | 0.9766     |
| **cosine_ap**             | **0.9846** |
| cosine_mcc                | 0.8664     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 8,144 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence_0                                                                         | sentence_1                                                                          | label                                                          |
  |:--------|:-----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                             | string                                                                              | float                                                          |
  | details | <ul><li>min: 8 tokens</li><li>mean: 31.29 tokens</li><li>max: 229 tokens</li></ul> | <ul><li>min: 11 tokens</li><li>mean: 75.66 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.54</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                                                                                                                                                                                                                                                                                                                | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                           | label            |
  |:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Hepatorenal syndrome (HRS) is the functional renal failure associated with advanced cirrhosis and has also been described in fulminant hepatic failure. Without liver transplantation its prognosis is dismal. Our study included patients with type 1 HRS associated with cirrhosis, who were not liver transplant candidates.AIM: To identify variables associated with improved survival.</code> | <code>We report for the first time ESLD etiology as a prognostic factor for survival. The renal function (expressed as serum creatinine) and urinary Na (<5 mEq/l) at the time of diagnosis were found to be associated with survival, suggesting that early treatment might increase survival.</code>                                                                                                                                                               | <code>1.0</code> |
  | <code>Do patients with localized prostate cancer treatment really want more aggressive treatment?</code>                                                                                                                                                                                                                                                                                                  | <code>Examine whether patients with prostate cancer choose the more aggressive of two radiotherapeutic options, whether this choice is reasoned, and what the determinants of the choice are.</code>                                                                                                                                                                                                                                                                 | <code>0.0</code> |
  | <code>Is grandmultiparity an independent risk factor for adverse perinatal outcomes?</code>                                                                                                                                                                                                                                                                                                               | <code>A database of the vast majority of maternal and newborn hospital discharge records linked to birth/death certificates was queried to obtain information on all multiparous women with a singleton delivery in the state of California from January 1, 1997 through December 31, 1998. Maternal and neonatal pregnancy outcomes of grandmultiparous women were compared to multiparous women who were 30 years or older at the time of their last birth.</code> | <code>1.0</code> |
* Loss: [<code>ContrastiveLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#contrastiveloss) with these parameters:
  ```json
  {
      "distance_metric": "SiameseDistanceMetric.COSINE_DISTANCE",
      "margin": 0.5,
      "size_average": true
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `num_train_epochs`: 5
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 16
- `per_device_eval_batch_size`: 16
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 5e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.0
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: False
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 42
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: False
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: None
- `hub_always_push`: False
- `hub_revision`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `include_for_metrics`: []
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `eval_use_gather_object`: False
- `average_tokens_across_devices`: False
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin

</details>

### Training Logs
| Epoch  | Step | Training Loss | validation_cosine_ap |
|:------:|:----:|:-------------:|:--------------------:|
| 0.9823 | 500  | 0.0137        | 0.9816               |
| 1.0    | 509  | -             | 0.9816               |
| 1.9646 | 1000 | 0.0074        | 0.9844               |
| 2.0    | 1018 | -             | 0.9789               |
| 2.9470 | 1500 | 0.005         | 0.9827               |
| 3.0    | 1527 | -             | 0.9846               |


### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 4.1.0
- Transformers: 4.53.1
- PyTorch: 2.7.1+cu126
- Accelerate: 1.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### ContrastiveLoss
```bibtex
@inproceedings{hadsell2006dimensionality,
    author={Hadsell, R. and Chopra, S. and LeCun, Y.},
    booktitle={2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'06)},
    title={Dimensionality Reduction by Learning an Invariant Mapping},
    year={2006},
    volume={2},
    number={},
    pages={1735-1742},
    doi={10.1109/CVPR.2006.100}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->