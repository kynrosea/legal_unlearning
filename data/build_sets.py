import random
import tqdm
from utils import text_utils as tu
from config import *


# use NER to find candidate names
# source: https://huggingface.co/dslim/bert-base-NER
def extract_names(samples, nlp):
    name_counts = {}
    name_contexts = {}
    candidate_names = []

    for sample in tqdm(samples, desc='Name Extraction'):
        text = sample.get('text', '')
        if not text:
            continue
        chunks = tu.chunk_text(text)
        for chunk in chunks:
            try:
                ner_results = nlp(chunk)
            except Exception as e:
                continue
            
            # ensuring entity is a proper name candidate before validation
            for entity in ner_results:
                if entity['entity_group'] != 'PER':
                    continue
                if float(entity['score']) < NER_SCORE_THRESHOLD:
                    continue

                name = entity['word'].strip()

                if len(name.split()) < MIN_NAME_TOKENS:
                    continue

                if name.islower():
                    continue

                if any(c.isdigit() for c in name):
                    continue

                if len(name) < 4:
                    continue
                
                # need this information for building context pairs and redacting name when performing the extraction leakage test
                name_counts[name] = name_counts.get(name, 0) + 1
                name_contexts[name] = name_contexts.get(name, [])
                name_contexts[name].append({
                    'chunk': chunk,
                    'start': entity['start'],
                    'end': entity['end'],
                    'score': float(entity['score'])
                })

    for candidate in name_counts:
        if name_counts[candidate] >= MIN_NAME_FREQ:
            candidate_names.append(candidate)

    return name_counts, name_contexts, candidate_names

# validating name candidates to ensure that they are formatted properly for unlearning
def validate_name(name):
    name = name.strip()
    lower_copy = name.lower()
    tokens = name.split()

    if any(c == '#' for c in name):
        return False

    if len(name) < 4:
        return False

    if name.islower():
        return False

    if name.isupper():
        return False

    if len(tokens) > 4 or len(tokens) < MIN_NAME_TOKENS:
        return False

    if name in LOCATIONS:
        return False

    if name in LEGAL_TERMS:
        return False

    if any(suffix in lower_copy for suffix in ORG_SUFFIX):
        return False

    if any(word in lower_copy for word in COMMON_WORDS):
        return False

    upper_count = 0
    for token in tokens:
        if token[0].isupper():
            upper_count += 1

    if upper_count < 2:
        return False

    return True

# building context pairs that include the chunk of text the target name is from and the index of where the name starts and ends
# it also contains the text before the name appears as a prefix
def build_pairs(all_names, all_contexts):
    context_pairs = []

    for name in all_names:
        for context in all_contexts.get(name):
            chunk = context['chunk']
            start = context['start']
            end = context['end']

            if chunk is None or start is None or end is None:
                continue

            # check if name is actually in the given context/chunk
            if chunk[start:end] != name:
                start = chunk.find(name)
                if start == -1:
                    continue
                else:
                    end = start + len(name)

            # grabbing the text right before the name appears to use as prefix
            prefix = chunk[:start]
            if len(prefix) < 40:
                continue

            pair = {
                'name': name,
                'prefix': prefix,
                'chunk': chunk,
                'start': start,
                'end': end
            }

            context_pairs.append(pair)

    return context_pairs

# taking chunked text to identify the matching name, context pairs for set building
def build_pairs_from_chunks(chunks, all_pairs):
    chunkset = set(chunks)
    chunk_pairs = []

    for p in all_pairs:
        if p['chunk'] in chunkset:
            chunk_pairs.append(p)

    return chunk_pairs


def build_forget_set(all_pairs, eval_forget_pairs, max_forget_pairs, seed):
    unique_chunks = sorted({p['chunk'] for p in all_pairs})
    random.seed(seed)
    random.shuffle(unique_chunks)

    # ensuring the num of eval pairs is not more than 1/4 of the num of disjoint chunks
    while eval_forget_pairs > len(unique_chunks) // 4:
        eval_forget_pairs //= 4

    eval_set = []
    eval_chunk_count = max(1, eval_forget_pairs//2)
    max_eval_chunks = max(1, len(unique_chunks)//2) # should save at least half of chunks for train set
    eval_chunks = unique_chunks[:eval_chunk_count]
    eval_pairs = build_pairs_from_chunks(eval_chunks, all_pairs)
    # incrementally building test set 
    while len(eval_pairs) < eval_forget_pairs and eval_chunk_count < max_eval_chunks:
        eval_chunk_count += 5
        eval_chunks = unique_chunks[:eval_chunk_count]
        eval_pairs = build_pairs_from_chunks(eval_chunks, all_pairs)

    eval_set = eval_pairs[:eval_forget_pairs]


    train_chunks = unique_chunks[eval_chunk_count:]
    train_set = build_pairs_from_chunks(train_chunks, all_pairs)[:max_forget_pairs]

    # sets must be disjoint, otherwise the overlap will damage the training and testing process
    assert set(eval_chunks).isdisjoint(set(train_chunks)), 'sets are not disjoint!'
    assert eval_set, 'test set is empty!'
    assert train_set, 'train set is empty!'

    # sanity check
    split_report = {
        'unique_chunks': len(unique_chunks),
        'eval_chunks': len(eval_chunks),
        'train_chunks': len(train_chunks),
        'eval_set': len(eval_set),
        'train_set': len(train_set)
    }

    return train_set, eval_set, split_report

# need this for calculating retain KL loss
def build_retain_set(contracts, all_names):
    retain_chunks = []

    for row in contracts:
        text = row['text']
        chunks = tu.chunk_text(text)
        for chunk in chunks:
            # should not contain any target names in the retain chunks
            if any(name in chunk for name in all_names):
                continue
            if len(chunk) < 150:
                continue

            retain_chunks.append(chunk)

            if len(retain_chunks) >= MAX_RETAIN_CHUNKS:
                return retain_chunks

    return retain_chunks
