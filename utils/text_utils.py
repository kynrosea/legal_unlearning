# text cleaning and chunking functions
import string
# text cleaning and chunking functions

# chunk text
def chunk_text(text, chunk_size=1000):
    all_chunks = []
    start = 0
    while start < len(text):
        end = start+chunk_size
        if end < len(text):
            # ensures that chunks do not end in the middle of a sentence
            boundary = text.rfind('.', start, end)
            if boundary != -1 and boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        # don't add empty chunks to list
        if chunk:
            all_chunks.append(chunk)
        start = end

    return all_chunks

def normalize_name(text):
    # remove irrelevant punctuation
    name = text
    for c in text:
        if c in string.punctuation:
            if c == '-' or c == '\'':
                continue
            else:
                index = text.find(c)
                name = text[:index] + text[index+1:]

    # collapse multiple spaces; may not be necessary for right now

    # convert to lowercase; may not need this anymore
    #name = name.lower()

    # return normalized string
    return name

# checking if any target name is in chunk to help structure forget sets
def contains_target(chunk, target_names):
    for name in target_names:
        if name in chunk:
            return True

    return False