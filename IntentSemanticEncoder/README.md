# Intent Semantic Encoder

This is an AWS Applied Science Internship project from Yuwei Zhang (yuwzhan).

Project goal: During this internship, we mainly investigated (1) whether current encoders for intent can actually discriminate negation utterances (i.e. the customers express that they are not interested in the service). (2) whether these encoders can model implicature utterances (i.e. the customers imply their needs without literal mentions). (3) whether we can train a model leveraging different prompts to reduce the errors.

Project outcome:
* We build an evaluation toolkit to analyze embedding models on understanding intent semantics. The toolkit integrates two challenges: negation and implicature. It also combines 4 evaluation tasks: triplet task, binary classification, clustering and classification, in order to fully investigate model capabilities.
* Our evaluation results on current intent encoders show that these models indeed focus more on the surface form rather than the actual intent semantics. And larger models tend to perform better.
* We curate training data via a “zoo” of prompts to generate diverse and hard positive/negative utterances according to the original ones. We train the encoder with these triplets.
* Our evaluation results for trained models show that, LLM-generated data effectively improve results on triplet and binary classification tasks. On clustering and classification, the results are comparable or better.
* Further analysis shows that the performances on negation and classification/clustering are actually (almost) negatively correlated. This encourages more research to balance the two.

In order to improve reproducibility, I will record all steps to build from source. I also wrote down several challenging bugs I have encountered before `ISSUES.md`.

## Preparation

### Install
```bash
conda create -n intent_semantic python=3.8
conda activate intent_semantic
pip install -r requirements.txt
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
python -m spacy download en_core_web_sm
```
If you want to add more packages, please add all of them into `requirements.txt` and install through `pip install -r requirements.txt`. Otherwise, it will be pretty hard to keep track of the package installing history.

### Download datasets
```bash
cd datasets
bash download.sh
```
* If the original dataset has a split, please follow their splits. Otherwise, please provide the code to generate your split with a fixed seed.
* This code will generate everything for you, including both evaluation and pre-training data.

### LLM Usage
#### For OpenAI models,
* Set all temperature to 0. But it will not guarantee to be consistent though: [https://platform.openai.com/docs/guides/gpt/why-are-model-outputs-inconsistent](https://platform.openai.com/docs/guides/gpt/why-are-model-outputs-inconsistent)
* DO NOT POSTPROCESS OUTPUTS. Keep them as they were generated.
* Please add your own OpenAI key and org in `scripts/build_toolkit_negation.sh`, `scripts/brain_storm_scenarios.sh`, `scripts/generate_implicature.sh`, `scripts/intent_alignment.sh`.

#### For AWS endpoint models,
* Set all temperature to 0.
* Plsease add your credentials in `build_toolkit_with_endpoints/endpoints_tools.py`.

### IAE model
Make sure IAE model is saved in `models/iae_model`.

## Intent Semantic Toolkit

### Data Generation
```bash
bash scripts/build_toolkit_negation.sh
bash scripts/brain_storm_scenarios.sh
bash scripts/generate_implicature.sh
```
* After generation, the data will be saved into `build_toolkit/results`.
* Also, you can check out all the prompts in `build_toolkit/prompts`. There are some deprecated ones.
* For negation, the most recent prompt is `build_toolkit/prompts/negate_intent_all_v2.txt`.
* For scenario, the most recent prompt is `build_toolkit/prompts/brain_storm_v2.txt`.
* For implicature, the most recent prompt is `build_toolkit/prompts/in_context_implicature.txt`.

### Evaluation
```bash
bash scripts/triplet_task_v2.sh
bash scripts/binary_classification_v2.sh
bash scripts/clustering.sh
bash scripts/protonet.sh
bash scripts/measure_implicature_v2.sh
cd evaluate_results
python average_results.py
```
* The evaluation requires gpu access. After running, the results will be saved into `evaluate_results`.
* The final averaging step will output `averaged_logs.json` and `dataset_logs.json`. The former is the averaging scross 3 evaluation datasets, while the latter is the dataset-specific logs.
* You might optionally draw the graph for vocabulary overlap with `evaluate_results/draw_vocab_overlap_fig.py`.
* The models are also evaluated on dev&test splits. Make sure to manualy switch the split on the top.
```bash
bash scripts/triplet_task_split.sh
bash scripts/binary_classification_split.sh
bash scripts/clustering_split.sh
bash scripts/protonet_split.sh
```

### Intent Alignment
Intent Alignment is used to check multi-label cases in implicature set with ChatGPT. Run
```bash
bash scripts/intent_alignment.sh
cd build_toolkit
python calc_gpt4_acc.py
```
You will see `build_toolkit/intent_alignment_logs.json` generated. This is treated as an upper-bound to the classification performance.

## Model Training

### Intent Extraction
You might want to shuffle the input data before generation. It will take a long time and you probably want to terminate it early.
```bash
bash scripts/extract_goals.sh
```
You will see `build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3.json` generated.
```bash
bash scripts/parse_intents.sh
```
You will see `build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_parsed.json` generated.
```bash
bash scripts/summarize_intent.sh
```
You will see `build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5.json` generated.

### Hard Example Generation
```bash
bash scripts/generate_hard_examples.sh
```
You will see `build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive.json` generated.
```bash
bash scripts/generate_hard_negatives.sh
```
You will see `build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative.json` generated.
```bash
bash scripts/retrieve_hard_examples.sh
```
A gpu access is also needed. The results will be saved in `build_toolkit_with_endpoints/results/pretrain/extracted_intents_falcon-40b-instruct_extract_intent_v3_summarize_intent_v5_hard_positive_hard_negative_ret_${MODEL_NAME}.json`
Since I did LLM generation and retrieval parallelly, I need to later merge them into a single file.
```bash
cd build_toolkit_with_endpoints
python merge_data.py
```
Remember to modify `merge_data.py` to change the file names.

In order to diversify the pre-training data, I also use some heuristics to modify them.
```bash
bash scripts/post_process_hard_examples.sh
```
A file name with `_proc` at last will be generated.

Everything is completed now. You can check the data generated manually to make sure this is what you want. I would recommend to do a pilot round first (by providing it a small amount of data). And then you can make your decision.

### Training
* `pretrain_code/train.py` is the main training code. The training objective starts at line 100. The ablation starts at line 602.
* `pretrain_code/train_st.py` is the training code for sentence-transformers. The main difference is that there are no prompts.
* `pretrain_code/train_st_v2.py` is different from v1 because of the input format. Now it takes inputs from IAE training data.
* The training commands are saved in `scripts/train.sh`. Uncomment the one you want and make sure to check all the arguments before you start training.
* After training, the results will be saved in `pretrain/results`.

### Evaluate Trained Models
```bash
bash scripts/evaluate_ours_instructor_base.sh
bash scripts/evaluate_ours_instructor_large.sh
bash scripts/evaluate_ours_paraphrase.sh
```
The trained models are also evaluated on dev&test splits. Remember to switch the `split` at top manually.
```bash
bash scripts/evaluate_ours_instructor_base_split.sh
bash scripts/evaluate_ours_instructor_large_split.sh
bash scripts/evaluate_ours_paraphrase_split.sh
```
After that, in order to avoid copying data a lot, there is also a script to save averaged results in `.csv` format - `pretrain/get_result_table.py`. Check the code there. Also, please check this repo: [RankingNLPSystems](https://github.com/PierreColombo/RankingNLPSystems.git) to get the final ranking.

## Human Eval
```bash
bash scripts/human_eval_sample_data.sh
```
The data will be generated into `human_eval/results`. To read results
```bash
bash scripts/human_eval_read_results.sh
```

## Reference
https://github.com/xlang-ai/instructor-embedding.git

https://github.com/PierreColombo/RankingNLPSystems.git

https://github.com/amazon-science/intent-aware-encoder.git

https://quip-amazon.com/pqlxA124g8kY/Yuwei-Artifacts

https://quip-amazon.com/bIPIAJ0Pe1Wx/Experiment-Hub

## S3 URL
s3://lex-interns/yuwzhan/

## repo
https://code.amazon.com/packages/AWSContactLensIntentDetectionDataset/trees/mainline

## Contact
Yuwei Zhang (yuwzhan@amazon.com, zhangyuwei.work@gmail.com, yuz163@ucsd.edu)