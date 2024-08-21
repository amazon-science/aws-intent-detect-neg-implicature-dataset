## Structure

* Pretraining (5): multiwoz, sgd, top, topv2, dstc11-t2
* Evaluation (3): BANKING77, HWU64, CLINC150

In each evaluation dataset, you will find: `candidates`, `negated_candidates`, `explanations`, `domains`. Those are used for zero-shot learning and LLM prompts. Pre-training data are combined to a single file. TOPv2 might be a superset of TOP, so it is necessary to merge them and deduplicate.

## Download and Preprocess
```
bash download.sh
```