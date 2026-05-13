from evaluation import metrics as m
import torch
from torch.utils.data import DataLoader, Dataset
from config import *
from torch.optim import AdamW
import torch.nn.functional as F
import csv



class ForgetSpanDataset(Dataset):

    # each sample is (input_ids, attention_mask, target_mask)
    # build the input_ids manually so target tokens are the name tokens
    def __init__(self, pairs, tokenizer, max_len):
        self.examples = []
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        for p in pairs:
            ids, tmask = m.encode_sequence(tokenizer, p["prefix"], p["name"], max_len)
            if ids is None:
                continue
            real_len = len(ids)
            # Right-pad to max_len so attention_mask + target_mask are aligned with input_ids
            pad_n = max_len - real_len
            ids   = ids   + [pad_id] * pad_n
            tmask = tmask + [0]      * pad_n
            attn  = [1] * real_len + [0] * pad_n
            if sum(tmask) == 0:
                continue
            self.examples.append({
                "input_ids":      torch.tensor(ids,   dtype=torch.long),
                "attention_mask": torch.tensor(attn,  dtype=torch.long),
                "target_mask":    torch.tensor(tmask, dtype=torch.long),
            })
        if len(self.examples) == 0:
            raise RuntimeError("ForgetSpanDataset is empty — every pair had degenerate target mask.")
    def __len__(self):  return len(self.examples)
    def __getitem__(self, i):  return self.examples[i]


class RetainDataset(Dataset):

    def __init__(self, texts, tokenizer, max_len):
        # use right padding for retain batches
        prev = tokenizer.padding_side
        tokenizer.padding_side = "right"
        try:
            self.enc = tokenizer(texts, padding="max_length", truncation=True,
                                  max_length=max_len, return_tensors="pt")
        finally:
            tokenizer.padding_side = prev
    def __len__(self): return self.enc["input_ids"].shape[0]
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()}

# calculating the log probability of only the target name
def masked_logprob_sum(model, input_ids, attention_mask, target_mask):
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask   = (attention_mask[:, 1:] * target_mask[:, 1:]).float()
    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_lp = logprobs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    return (token_lp * shift_mask).sum(dim=-1)

# calculating npo loss
def npo_span_loss(model, ref_model, batch, beta):
    ids  = batch["input_ids"].to(model.device)
    mask = batch["attention_mask"].to(model.device)
    tgt  = batch["target_mask"].to(model.device)
    lp_theta = masked_logprob_sum(model,     ids, mask, tgt)
    with torch.no_grad():
        lp_ref = masked_logprob_sum(ref_model, ids, mask, tgt)
    log_ratio = lp_theta - lp_ref
    # npo loss function
    loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio).mean()
    return loss, log_ratio.mean().item()

# KL compares model output with the ref model's output
# done on next-token predictions here
def retain_kl_loss(model, ref_model, batch):

    ids  = batch["input_ids"].to(model.device)
    mask = batch["attention_mask"].to(model.device)

    # grabbing logits from reference and current model given retain set input ids
    with torch.set_grad_enabled(model.training):
        logits_m = model(input_ids=ids, attention_mask=mask).logits
    with torch.no_grad():
        logits_r = ref_model(input_ids=ids, attention_mask=mask).logits

    # using shifted masks to avoid scoring pad or current token positions
    if RETAIN_KL_SHIFTED:
        logits_m = logits_m[:, :-1, :]
        logits_r = logits_r[:, :-1, :]
        tok_mask = mask[:, 1:].float()
    else:
        tok_mask = mask.float()

    log_m = F.log_softmax(logits_m, dim=-1)
    p_r   = F.softmax(logits_r, dim=-1)
    kl = (p_r * (torch.log(p_r + 1e-12) - log_m)).sum(dim=-1)
    kl = (kl * tok_mask).sum() / tok_mask.sum().clamp_min(1)

    return kl

# running the npo training here
def run_npo_with_retain(model, ref_model, tokenizer, train_pairs, retain_chunks,
                         steps, lr, batch_size, max_seq_len, beta, lambda_retain,
                         trace_csv_path=None):
    
    # preparing the forget and retain sets here for model input
    forget_ds = ForgetSpanDataset(train_pairs, tokenizer, max_seq_len)
    retain_ds = RetainDataset(retain_chunks, tokenizer, max_seq_len)
    forget_dl = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
    retain_dl = DataLoader(retain_ds, batch_size=RETAIN_BATCH_SIZE, shuffle=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=lr, weight_decay=0.0)
    f_iter, r_iter = iter(forget_dl), iter(retain_dl)

    # need these to see the changes in the model after every five steps
    loss_hist, ratio_hist, retain_hist, forget_hist = [], [], [], []
    model.train()
    print(f"NPO+retain: steps={steps} β={beta} λ_retain={lambda_retain} lr={lr}")
    print(f"forget pool={len(forget_ds)} retain pool={len(retain_ds)}")

    trace_rows = []
    for step in range(1, steps + 1):
        try: fb = next(f_iter)
        except StopIteration:
            f_iter = iter(forget_dl); fb = next(f_iter)
        try: rb = next(r_iter)
        except StopIteration:
            r_iter = iter(retain_dl); rb = next(r_iter)

        forget_loss, log_ratio = npo_span_loss(model, ref_model, fb, beta)
        ret_loss = retain_kl_loss(model, ref_model, rb)
        loss = forget_loss + lambda_retain * ret_loss

        # computing gradients and updating parameter weights
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), 1.0)
        optimizer.step()

        loss_hist.append(float(loss.item()))
        forget_hist.append(float(forget_loss.item()))
        ratio_hist.append(float(log_ratio))
        retain_hist.append(float(ret_loss.item()))
        trace_rows.append({
            "step": step,
            "total_loss":  float(loss.item()),
            "forget_loss": float(forget_loss.item()),
            "retain_kl":   float(ret_loss.item()),
            "log_ratio":   float(log_ratio),
        })
        if step % 5 == 0 or step == 1:
            print(f"  step {step:3d}  total={loss.item():8.4f}  forget={forget_loss.item():8.4f}"
                  f"  retain_kl={ret_loss.item():7.4f}  log_ratio={log_ratio:8.2f}")

        # stop training loop if model utility starts significantly degrading compared to ref model
        if EARLY_STOP_ENABLED and step >= EARLY_STOP_MIN_STEPS:
            if ret_loss.item() > EARLY_STOP_RETAIN_KL and log_ratio <= EARLY_STOP_LOG_RATIO_TARGET:
                print(f"Early stop at step {step}: retain_kl={ret_loss.item():.4f}, log_ratio={log_ratio:.2f}")
                break

    return loss_hist, ratio_hist, retain_hist, forget_hist