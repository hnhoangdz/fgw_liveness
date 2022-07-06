import random
from itertools import permutations

def make_rules():
    data = {
        'emotions': ['happy', 'angry', 'sad'],
        'eye_blink': ['blink left eye', 'blink right eye', 'blink entire eyes'],
        'orientation': ['turn left', 'turn right']
    }
    
    base_rules = []
    
    for k in data.keys():
        rand_idx = random.randint(0, len(data[k])-1)
        base_rules.append([k,data[k][rand_idx]])
    perm_rules = list(permutations(base_rules))
    rand_choice = random.randint(0, len(perm_rules)-1)
    my_rules = perm_rules[rand_choice]
    
    return my_rules

def solve_rules(question, blink_up=False):
    if question == 'happy':
        result = 'pass'
    elif question == 'angry':
        result = 'pass'
    elif question == 'sad':
        result = 'pass'
    elif question == 'blink left eye':
        if blink_up == True:
            result = 'pass'
        else:
            result = 'fail'
    elif question == 'blink right eye':
        if blink_up == True:
            result = 'pass'
        else:
            result = 'fail'
    elif question == 'blink entire eyes':
        if blink_up == True:
            result = 'pass'
        else:
            result = 'fail'
    elif question == 'turn left':
        result = 'pass'
    elif question == 'turn right':
        result = 'pass'
    return result
    