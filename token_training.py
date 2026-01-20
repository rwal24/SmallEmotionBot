import re
import json

def encoder(text):
    # encode textual input into utf-8 encoding, becomes a list of integers
    text2 = re.sub(r'\"', '', text)
    return list(map(int, re.sub(r' {2,}', ' ', text2).encode("utf-8")))

    #NO MORE .lower(), may add back later, for now, no
    

def stats(encoding):
    counts = {}
    for pair in zip(encoding, encoding[1:]):
        if not pair in counts:
            counts[pair] = 1
        else:
            counts[pair] += 1
    

    #sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    pair = max(counts, key=counts.get)
    if counts[pair] == 1:
        return pair, 1

    return pair, 0


def merge(old_ids, pair, new_id):
    # from encoded_ids, if a pair matches, replace it with the new_id
    newids = []

    i = 0
    n = len(old_ids)
    while i < n:
        # if pair is found, add new id too string from that
        if i < n - 1 and old_ids[i] == pair[0] and old_ids[i + 1] == pair[1]:
            newids.append(new_id)
            i += 2
        else:
            newids.append(old_ids[i])
            i += 1
    
    return newids



def bpe_training(text):
    # utf-8 encoding of the original text
    current = encoder(text)
    #old_ids = list(current)

    with open("lookup_table.json", "r") as f:
        data = json.load(f)
        table_size = data["table_size"]
        current_pairs = data["LookupTable"]
    
    merges = {}
    for i in range(21):
        if len(current) <= 3:
            break

        pair, stat = stats(current)
        if stat == 1:
            if pair not in current_pairs:
                break

        pair_key = f"{pair[0]},{pair[1]}"
        

        # check if pair has already been accounted for previously
        if pair_key in current_pairs:
            current = merge(current, pair, current_pairs[pair_key])
            continue
            

        idx = 256 + table_size
        table_size += 1

        # merge using the index calculated by adding table size too idx
        current = merge(current, pair, idx)
        merges[pair_key] = idx

    # add keys to existing pairs
    for key in merges:
        current_pairs[key] = merges[key]

    # create new json format for lookup table
    newData = {
        "LookupTable":current_pairs,
        "table_size":table_size
    }

    with open("lookup_table.json", "w") as f:
        json.dump(newData, f, indent=4)
    




with open("tokenizer_phrases.json", "r") as f:
    data = json.load(f)
    phrases = data["phrases"] 

for sentence in phrases:
    bpe_training(sentence)





"""
JSON emotion index guide:

0: happy
1: sad
2: angry
30 confused
40 neutral
5: fear
6: suprised
7: disgusted
8: anxious
9: shame
10: excited

"""


def break_up_text_for_tokenizer(text_file_names):
    for file in text_file_names:
        with open(file, "r") as f:
            previous = ""
            for line in f:
                clean_line = line.strip()

                if len(clean_line) == 0:
                    continue
                
                if previous:
                    current_line = previous + " " + clean_line
                else:
                    current_line = clean_line

                if len(current_line) <= 225:
                    previous = current_line
                    continue

                else:
                    bpe_training(current_line)
                    previous = ""

        
'''
# requires a path to a text file;./ :
text_files = []

break_up_text_for_tokenizer(text_files)


'''