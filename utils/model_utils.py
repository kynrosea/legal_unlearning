# loads the model and prepares LoRA
from config import *
import torch
import copy
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

def load_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    return tokenizer

# source: TODO
def load_model(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map='auto'
    )

    model.eval()

    return model

# source: https://huggingface.co/docs/datasets/v1.11.0/dataset_streaming.html#:~:text=To%20shuffle%20your%20dataset%2C%20the,are%20replaced%20by%20new%20examples.
def load_contracts_data(dataset_name, subset_name, split, sample_size, seed):
    dataset = load_dataset(dataset_name, subset_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed)
    batch = list(dataset.take(sample_size))
    return batch

def load_NER(ner_model):
    tokenizer = AutoTokenizer.from_pretrained(ner_model)
    model = AutoModelForTokenClassification.from_pretrained(ner_model)

    # aggregation prevents returning weird fragments instead of full names
    nlp = pipeline('ner',
                   model=model,
                   tokenizer=tokenizer,
                   aggregation_strategy='simple',
                   device=0 if torch.cuda.is_available() else -1
                   )

    return nlp

def attach_lora(model):
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=['q_proj', 'k_proj', 'v_proj','o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    
    lora_model = get_peft_model(model, lora_config)

    #lora_model.print_trainable_parameters()

    return lora_model

def make_reference_model(model):
    # deep copy the model
    ref_model = copy.deepcopy(model)

    # set copied model to eval model
    ref_model.eval()

    # freeze all copied model params
    for param in ref_model.parameters():
        param.requires_grad_(False)

    # return copied reference model
    return ref_model

