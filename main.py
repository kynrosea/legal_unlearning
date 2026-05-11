# reads like the project pipeline
# all the messy details in separate files
from config import *
from utils import data_utils as du, model_utils as mu, reproducibility as r, text_utils as tu
from training import train
from evaluation import metrics as m
from data import build_sets as bs
from results import *

# setting random seed for reproducibility
r.set_seed(SEED)

# loading tokenizer, model, and NER pipeline
tokenizer = mu.load_tokenizer(MODEL_NAME)
model = mu.load_model(MODEL_NAME)
ner_pipeline = du.load_NER(NER_MODEL)

contracts_data = du.load_contracts_data(DATASET, SUBSET, 'train', SAMPLE_SIZE, SEED)

name_counts, name_contexts, candidate_names = bs.extract_names(contracts_data, ner_pipeline)
validated_names = []
for name in candidate_names:
    if bs.validate_name(name):
        validated_names.append(name)

context_pairs = bs.build_pairs(validated_names, name_contexts)
forget_train, forget_test, split_report = bs.build_forget_set(context_pairs, EVAL_FORGET_PAIRS, MAX_FORGET_PAIRS, SEED)
retain_set = bs.build_retain_set(contracts_data, validated_names)

forget_metrics_before = m.compute_forget_metrics(model, tokenizer, forget_test)

lb_accuracy_before = m.run_legalbench_eval(model, tokenizer, LEGALBENCH_TASKS)

model = mu.attach_lora(model)
ref_model = mu.make_reference_model(model)

loss_history, ratio_history, retain_history, forget_history = train.run_npo_with_retain(model,
                                                                                        ref_model,
                                                                                        tokenizer,
                                                                                        forget_train,
                                                                                        retain_set,
                                                                                        NPO_STEPS,
                                                                                        NPO_LR,
                                                                                        NPO_BATCH_SIZE,
                                                                                        MAX_SEQ_LEN,
                                                                                        NPO_BETA,
                                                                                        RETAIN_LOSS_WEIGHT)


forget_metrics_after = m.compute_forget_metrics(model, tokenizer, forget_test)

lb_accuracy_after = m.run_legalbench_eval(model, tokenizer, LEGALBENCH_TASKS)

forget_delta, lb_delta, selectivity_gap = compute_selectivity_gap(forget_metrics_before, forget_metrics_after, lb_accuracy_before, lb_accuracy_after)

print_results(forget_metrics_before, forget_metrics_after, lb_accuracy_before, lb_accuracy_after)

plot_results(forget_metrics_before, forget_metrics_after, lb_accuracy_before, lb_accuracy_after,
             loss_history, ratio_history, retain_history, forget_history)

save_results(
    forget_train, retain_set, validated_names,
    forget_metrics_before, forget_metrics_after,
    lb_accuracy_before, lb_accuracy_after,
    lb_accuracy_after['__mean__'],
    lb_accuracy_before['__mean__'],
    loss_history, forget_history,
    ratio_history, retain_history
)
