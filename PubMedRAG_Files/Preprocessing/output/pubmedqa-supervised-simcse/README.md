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
- source_sentence: Does the aggressive use of polyvalent antivenin for rattlesnake
    bites result in serious acute side effects?
  sentences:
  - Current surgical techniques of tibial preparation may result in partial or total
    PCL damage. Tibial tuberosity is a useful anatomical landmark to locate the PCL
    footprint and to predict the probability of its detachment pre-, intra-, and postoperatively.
    This knowledge might be useful to predict and avoid instability, consecutive pain,
    and dissatisfaction after TKA related to PCL insufficiency.
  - Combined treatment with hyperbaric oxygen and GH increased the mean bursting pressure
    values in all of the groups, and a statistically significant increase was noted
    in the ischemic groups compared to the controls (p<0.05). This improvement was
    more evident in the ischemic and normal groups treated with combined therapy.
    In addition, a histopathological evaluation of anastomotic neovascularization
    and collagen deposition showed significant differences among the groups.
  - Measuring calprotectin may help to identify UC and colonic CD patients at higher
    risk of clinical relapse.
- source_sentence: 'Differentiation of nonalcoholic from alcoholic steatohepatitis:
    are routine laboratory markers useful?'
  sentences:
  - Results of 77 pairs of CT (thorax, abdomen, and pelvis) and BS in newly diagnosed
    patients with metastatic breast cancer (MBC) were compared prospectively for 12
    months. Both scans were blindly assessed by experienced radiologists and discussed
    at multidisciplinary team meetings regarding the diagnosis of bone metastases.
  - Our data suggest that UHR intake criteria predict transition over 6 months in
    the order of Trait alone<APS<BLIPS. The fact that BLIPS patients are at the highest
    risk of transition over the short term is consistent with the "early" versus "late"
    prodrome model. It also indicates that particular clinical attention may need
    to be paid to BLIPS patients, especially early in the course of treatment.
  - Observational study of patients with ischaemic heart disease attending an urban
    tertiary referral cardiology centre.
- source_sentence: Does a preoperative medically supervised weight loss program improve
    bariatric surgery outcomes?
  sentences:
  - In women with POP, the symptom of pelvic pain is associated with the presence
    of defecatory symptoms.
  - MSWM does not appear to confer additional benefit as compared to the standard
    preoperative bariatric surgery protocol in terms of weight loss and most behavioral
    outcomes after LAGB in our patient population.
  - Central anterior chamber depth was measured in 39 patients with clinically apparent
    unilateral pseudoexfoliation and elevated intraocular pressure. Patients were
    placed in a face-up position for 5 minutes, at which time anterior chamber depth
    and axial length were measured by A scan, and intraocular pressure was measured
    by Tonopen (Oculab, La Jolla, CA) in both eyes. The measurements were repeated
    on both eyes after 5 minutes in a face-down position.
- source_sentence: Women's experiences of childbirth may affect their future reproduction,
    and the model of care affects their experiences, suggesting that a causal link
    may exist between model of care and future reproduction. The study objective was
    to examine whether the birth center model of care during a woman's first pregnancy
    affects whether or not she has a second baby, and on the spacing to the next birth.
  sentences:
  - Measuring calprotectin may help to identify UC and colonic CD patients at higher
    risk of clinical relapse.
  - There is no indication of a rebound aggravation of symptoms 12 to 14 days after
    a 5-day treatment with lansoprazole 60 mg once daily in patients with reflux symptoms.
  - A woman's model of care, such as birth center care, during her first pregnancy
    does not seem to be a sufficiently important factor to affect subsequent reproduction
    in Sweden.
- source_sentence: 'Telemedicine and type 1 diabetes: is technology per se sufficient
    to improve glycaemic control?'
  sentences:
  - Pregnant women (n=35) and trainee midwives (n=36) were randomly presented with
    one of four PISs where the title and font of the PIS had been manipulated to create
    four experimental conditions (i.e., Double Fluent; Double Awkward; Fluent Title-Awkward
    Font; Awkward Title-Fluent Font). After reading the PIS, participants rated their
    perceptions of the intervention (i.e., Attractiveness, Complexity, Expected Risk,
    Required Effort) using five-point Likert scales.
  - Although initial infection control rate was substantially lower in the retention
    group than the removal group, final results were comparable at latest followup.
    We believe retention treatment can be selectively considered for non-S. aureus
    infection, and when applied in selected patients, polyethylene exchange should
    be performed.
  - The statistical power of this case-referent study was such that only large beneficial
    effects of statins in acute stroke could be confirmed. However, the observed trend,
    together with experimental observations, is interesting enough to warrant a more
    detailed analysis of the relationship between statins and stroke outcome.
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
      value: 0.9381270903010034
      name: Cosine Accuracy
    - type: cosine_accuracy_threshold
      value: 0.7200639247894287
      name: Cosine Accuracy Threshold
    - type: cosine_f1
      value: 0.9390444810543658
      name: Cosine F1
    - type: cosine_f1_threshold
      value: 0.7200639247894287
      name: Cosine F1 Threshold
    - type: cosine_precision
      value: 0.9253246753246753
      name: Cosine Precision
    - type: cosine_recall
      value: 0.9531772575250836
      name: Cosine Recall
    - type: cosine_ap
      value: 0.9847237869758949
      name: Cosine Ap
    - type: cosine_mcc
      value: 0.8766514068929248
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
    'Telemedicine and type 1 diabetes: is technology per se sufficient to improve glycaemic control?',
    'Although initial infection control rate was substantially lower in the retention group than the removal group, final results were comparable at latest followup. We believe retention treatment can be selectively considered for non-S. aureus infection, and when applied in selected patients, polyethylene exchange should be performed.',
    'The statistical power of this case-referent study was such that only large beneficial effects of statins in acute stroke could be confirmed. However, the observed trend, together with experimental observations, is interesting enough to warrant a more detailed analysis of the relationship between statins and stroke outcome.',
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
| cosine_accuracy           | 0.9381     |
| cosine_accuracy_threshold | 0.7201     |
| cosine_f1                 | 0.939      |
| cosine_f1_threshold       | 0.7201     |
| cosine_precision          | 0.9253     |
| cosine_recall             | 0.9532     |
| **cosine_ap**             | **0.9847** |
| cosine_mcc                | 0.8767     |

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
  | details | <ul><li>min: 9 tokens</li><li>mean: 32.44 tokens</li><li>max: 227 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 76.81 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.56</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                                                                                 | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | label            |
  |:-----------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Preoperative platelet count in esophageal squamous cell carcinoma: is it a prognostic factor?</code> | <code>The 18 patients could be divided on the basis of their performance into three groups: Three patients were demented and had impaired language function (group 1); two non-demented patients had an aphasic syndrome characterised by word finding difficulties and anomia (group 2). Major cognitive deficits were therefore found in five of the 18 patients (28%). The remaining 13 performed normally on the test battery apart from decreased verbal fluency (group 3).</code> | <code>0.0</code> |
  | <code>Does elective re-siting of intravenous cannulae decrease peripheral thrombophlebitis?</code>         | <code>PET/CT has a limited role in hepatic staging of LMCRC. Although PET-CT has higher sensitivity for the detection of extrahepatic disease in some anatomic locations, its results are hampered by its low PPV. PET/CT provided additional useful information in 8% of the cases but also incorrect and potentially harmful data in 9% of the staging. Our findings support a more selective use of PET/CT, basically in patients with high risk of local recurrence.</code>         | <code>0.0</code> |
  | <code>Quality of life in lung cancer patients: does socioeconomic status matter?</code>                    | <code>As demonstrated in this study, size reduction of the ascending aorta using aortoplasty with external reinforcement is a safe procedure with excellent long-term results. It is a therapeutic option in modern aortic surgery in patients with poststenotic dilatation of the aorta without impairment of the sinotubular junction of the aortic valve and root.</code>                                                                                                            | <code>0.0</code> |
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
| 0.9823 | 500  | 0.0136        | 0.9819               |
| 1.0    | 509  | -             | 0.9821               |
| 1.9646 | 1000 | 0.0074        | 0.9828               |
| 2.0    | 1018 | -             | 0.9834               |
| 2.9470 | 1500 | 0.0048        | 0.9841               |
| 3.0    | 1527 | -             | 0.9847               |


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