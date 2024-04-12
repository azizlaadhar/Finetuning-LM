import pickle

def save_binary(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_binary(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def isEnglish(sample):
    try:
        sample.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True
    
def lowerCase(sample):
    return {"text": sample["text"].lower()}

def replaceRare(sample, rare_tokens, unk_token):
    text = sample["text"]
    modified_tokens = [(token if token not in rare_tokens else unk_token)
                       for token in text.split()]
    return {"text": " ".join(modified_tokens)}

def isUnkSeq(sample, unk_token, unk_thred=0.1):
    sample_tokens = sample["text"].split()
    if sample_tokens.count(unk_token)/len(sample_tokens) > unk_thred:
        return True
    else:
        return False


