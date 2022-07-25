import random as rd
from itertools import permutations, combinations

def make_rules():
    rules = ['smile', 'eye_blink', 'side_face']
    rules = list(permutations(rules, 2))
    rules = rules[rd.randint(0, len(rules) - 1)]
    return rules

def solve_rules(question, blink_up=False):
    pass

if __name__ == '__main__':
    make_rules()