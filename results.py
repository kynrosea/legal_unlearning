# file prints and saves results of the model run and evaluation
from evaluation import metrics as m
import matplotlib.pyplot as plt
import numpy as np
from config import *
import json

# outputting all results for easy user viewing
def print_results(forget_before, forget_after, lb_before, lb_after):

    forget_delta, lb_drop, selectivity_gap = m.compute_selectivity_gap(forget_before, forget_after, lb_before, lb_after)
    print('-----Evaluation Metrics (BEFORE)-----')
    print(f'Name Log Probability Mean: {forget_before['target_name_logprob_mean']:.3f}')
    print(f'Name Completion Accuracy: {forget_before['name_completion_full']:.3f}')
    print(f'Name Extraction Leakage Rate: {forget_before['name_extraction_leakage_full']:.3f}')
    print(f'LegalBench Balanced Accuracy: {lb_before['__mean__']:.3f}')

    print('-----Evaluation Metrics (AFTER)-----')
    print(f'Name Log Probability Mean: {forget_after['target_name_logprob_mean']:.3f}')
    print(f'Name Completion Accuracy: {forget_after['name_completion_full']:.3f}')
    print(f'Name Extraction Leakage Rate: {forget_after['name_extraction_leakage_full']:.3f}')
    print(f'LegalBench Balanced Accuracy: {lb_after['__mean__']:.3f}')
    print(f'Selectivity Gap: {selectivity_gap:.3f}')
    print(f'Forget Delta: {forget_delta:.3f}')
    print(f'LegalBench Accuracy Delta: {lb_drop:.3f}')

def plot_results(forget_before, forget_after, lb_before, lb_after, loss_history, ratio_history, retain_history, forget_history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'Targeted NPO and Retain on SaulLM-7B-Instruct', fontsize=12)

    # bar graph of unlearning effectiveness metrics
    ax = axes[0,0]
    fk = ['target_name_logprob_mean', 'name_completion_full', 'name_extraction_leakage_full']
    x = np.arange(len(fk)); w = 0.38
    ax.bar(x-w/2, [forget_before[k] for k in fk], w, label='Before', color='seagreen')
    ax.bar(x+w/2, [forget_after[k] for k in fk], w, label='After', color='orchid')
    ax.set_xticks(x); ax.set_xticklabels([k.replace('_', '\n') for k in fk], fontsize=8)
    ax.set_title('Unlearning Effectiveness Metrics'); ax.legend()

    # bar graph of legal bench task accuracies
    shared = [t for t in LEGALBENCH_TASKS if t in lb_before and t in lb_after]
    ax = axes[0,1]
    x = np.arange(len(shared)); w = 0.38
    ax.bar(x-w/2, [lb_before[t] for t in shared], w, label='Before', color='seagreen')
    ax.bar(x+w/2, [lb_before[t]  for t in shared], w, label='After',  color='orchid')
    ax.set_xticks(x); ax.set_xticklabels(shared, rotation=40, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_title('LegalBench Balanced Accuracy'); ax.legend()

    # line graph of npo loss during each step of unlearning
    ax = axes[1,0]
    ax.plot(loss_history, color='darkorange', label='total')
    ax.plot(forget_history, color='red', alpha=0.6, label='forget only')
    ax.set_title('Loss'); ax.set_xlabel('step'); ax.legend(); ax.grid(alpha=0.3)

    # line graph of target name log ratio and retain KL to see the difference between
    # the forget set's unlearning and the model's utility
    ax = axes[1,1]
    ax.plot(ratio_history, color='purple', label='log π_θ − log π_ref (forget)')
    ax.set_xlabel('step'); ax.set_ylabel('log ratio', color='purple')
    ax.tick_params(axis='y', labelcolor='purple')
    ax2 = ax.twinx()
    ax2.plot(retain_history, color='green', label='retain KL')
    ax2.set_ylabel('retain KL', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax.set_title('Forget log-ratio vs retain KL'); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_PREFIX}_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def save_results(train_pairs, retain_chunks, 
                 forget_persons, forget_before, 
                 forget_after, lb_before, lb_after,
                 mean_a, mean_b, loss_history, forget_history,
                 ratio_history, retain_history):
    
    # grabbing all information pertinent to the full project run
    output = {
        "model":   MODEL_NAME,
        "forget_dataset":      f"{DATASET}/{SUBSET}",
        "unlearning_method":   "NPO span-targeted + KL retain",
        "hyperparameters": {
            "npo_beta": NPO_BETA, "npo_lr": NPO_LR, "npo_steps": NPO_STEPS,
            "lambda_retain": RETAIN_LOSS_WEIGHT, "max_seq_len": MAX_SEQ_LEN,
            "forget_pairs": len(train_pairs), "retain_chunks": len(retain_chunks),
            "min_entity_freq": MIN_NAME_FREQ, "ner_score_threshold": NER_SCORE_THRESHOLD,
        },
        'forget_set_names': forget_persons[:30],
        "forget_metrics_before": forget_before,
        "forget_metrics_after":  forget_after,
        "forget_deltas": {k: forget_after[k] - forget_before[k] for k in forget_before},
        "legalbench_before": {k: float(v) for k, v in lb_before.items()},
        "legalbench_after":  {k: float(v) for k, v in lb_after.items()},
        "legalbench_mean_before": float(mean_b),
        "legalbench_mean_after":  float(mean_a),
        "legalbench_mean_delta":  float(mean_a - mean_b),
        "loss_history":   [float(x) for x in loss_history],
        "forget_loss_history":  [float(x) for x in forget_history],
        "ratio_history":  [float(x) for x in ratio_history],
        "retain_history": [float(x) for x in retain_history],
        "training_diagnostics": {
            "executed_steps": len(loss_history),
            "min_log_ratio": float(np.min(ratio_history)) if ratio_history else None,
            "final_log_ratio": float(ratio_history[-1]) if ratio_history else None,
            "max_retain_kl": float(np.max(retain_history)) if retain_history else None,
            "final_retain_kl": float(retain_history[-1]) if retain_history else None,
            "early_stop_enabled": EARLY_STOP_ENABLED,
            "retain_kl_shifted": RETAIN_KL_SHIFTED,
        },
    }

    # saving results to a designated json file for easy lookup and comparison
    with open(f"{OUTPUT_PREFIX}_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {OUTPUT_PREFIX}_results.json")

    print(f"\nForget logprob Δ      : {output['forget_deltas']['target_name_logprob_mean']:+.4f}")
    print(f"Completion (full) Δ   : {output['forget_deltas']['name_completion_full']:+.4f}")
    print(f"Leakage (full) Δ      : {output['forget_deltas']['name_extraction_leakage_full']:+.4f}")
    print(f"LegalBench mean Δ     : {output['legalbench_mean_delta']:+.4f}")
    
