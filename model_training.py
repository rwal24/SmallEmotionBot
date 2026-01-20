import re
import json
import cpython_calculations as c 

def encoder(text):
    # encode textual input into utf-8 encoding, becomes a list of integers
    return list(map(int, re.sub(r'[ \n]+', ' ', text).encode("utf-8")))

    #NO MORE .lower(), may add back later, for now, no
    
# encoding function
def stats(encoding):
    counts = {}
    for pair in zip(encoding, encoding[1:]):
        if not pair in counts:
            counts[pair] = 1
        else:
            counts[pair] += 1
    

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    print(sorted_counts)
    
    return sorted_counts


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



def tokenize(text):
    current_encoding = encoder(text)
    with open("lookup_table.json", "r") as f:
        data = json.load(f)
        lookup_table = data["LookupTable"]
        # retrieve the lookup table

    
    i = 0
    while len(current_encoding) > 350:
        current_stats = stats(current_encoding)
        n = len(current_stats)

        most_common_id = current_stats[i][0]

        print(f"\n\n\n\n{most_common_id}\n\n\n\n")

        most_common_id_string = f"{most_common_id[0]},{most_common_id[1]}"
        print(f"\n\n\n\n{most_common_id_string}\n\n\n\n")

        # if the most common pair is found in the lookup table, replace all occurances of this pair with
        # the index of the pair found from the lookup table
        if most_common_id_string in lookup_table:
            current_encoding = merge(current_encoding, most_common_id, lookup_table[most_common_id_string])
            i = 0

        # if i is the size of the lookup table it means that not a single pair from 
        # the lookup table is present
        if i == n - 1:
            break

        # add 1 to i if the current most common pair is not found in the lookup table, and look for the second most common pair
        else:
            i += 1
    
    return current_encoding




def train_weight_vector(text_input, emotion_int, weight_file="weights.json", size = 500, num_of_emotions = 11):
    """
        JSON emotion index guide:

        0: happy
        1: sad
        2: angry
        3: confused
        4: frustrated
        5: fear
        6: suprised
        7: disgusted
        8: anxious
        9: shame
        10: excited
    """
    tokenized_text = tokenize(text_input)

    with open(weight_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    

    # all_data["emotions"][str(i)] is the current array representation of the weight vector
    new_weights = c.get_new_weight_vec(tokenized_text, all_data["emotions"], emotion_int, size, num_of_emotions)
    all_data["emotions"] = new_weights
    

    # all data now contains the updated weight vectors
    with open(weight_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=4)



# function to test the emotion analysis capabilities of this model
def test_emotion_analysis(text_input, weight_file="weights.json", size = 500, num_of_emotions = 11):
    """
        JSON emotion index guide:

        0: happy
        1: sad
        2: angry
        3: confused
        4: frustrated
        5: fear
        6: suprised
        7: disgusted
        8: anxious
        9: shame
        10: excited
    """
    tokenized_text = tokenize(text_input)

    with open(weight_file, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    

    # all_data["emotions"][str(i)] is the current array representation of the weight vector
    probabilities = c.get_predicted_emotions(tokenized_text, all_data["emotions"], size, num_of_emotions)

    return probabilities
    
    

def main(text, emotion_int, what_to_do):
    # what to do: 1 means train, 2 means test
    if len(text) <= 5:
        print(f"Inputs of length {len(text)} is too small to process")
        return 1
    
    if not 0 <= emotion_int <= 10:
        print(f"Emotion class {emotion_int} is not a valid emotion")
        return 1
    
    if what_to_do == 1:
        train_weight_vector(text, emotion_int)

    if what_to_do == 2:
        return test_emotion_analysis(text)
    
    return 0


