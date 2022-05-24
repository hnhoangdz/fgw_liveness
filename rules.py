import random
from itertools import permutations

def make_rules():
    data = {
        'emotions': ['happy', 'angry', 'sad'],
        'eye': ['left_blink', 'right_blink', 'entire_blink'],
        'orientation': ['left_side', 'right_side']
    }
    
    base_rules = []
    for k in data.keys():
        rand_idx = random.randint(0, len(data[k])-1)
        base_rules.append(data[k][rand_idx])
        
    perm_rules = list(permutations(base_rules))
    rand_choice = random.randint(0, len(perm_rules)-1)
    my_rules = perm_rules[rand_choice]
    return my_rules
        
make_rules()