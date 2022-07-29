import random as rd
from itertools import permutations, combinations

def make_rules():
    rules = ['side_face_left', 'side_face_right', 'eye_blink', 'smile_face']
    return rd.sample(rules, 3) #, signs

def solve_rule(question, label):
    challenge = ''
    if question == 'eye_blink':
        if label == True:
            challenge = 'pass'
        else:
            challenge = 'fail'
            
    elif question == 'smile_face':
        if label == 'smile':
            challenge = 'pass'
        else:
            challenge = 'fail'
    
    elif question == 'side_face_left':
        if label == True:
            challenge = 'pass'
        else:
            challenge = 'fail'
    
    elif question == 'side_face_right':
        if label == True:
            challenge = 'pass'
        else:
            challenge = 'fail'
            
    return challenge

def convert_rule2require(rule):
    require = ''
    if rule == 'eye_blink':
        require = 'Please blinking your eyes!'
        
    elif rule == 'smile_face':
        require = 'Please smiling!'
        
    elif rule == 'side_face_left':
        require = 'Please turning left face!'
        
    elif rule == 'side_face_right':
        require = 'Please turning right face!'

    return require
if __name__ == '__main__':
    # c = solve_rule('eye_blink', True)
    # print(c)
    x = make_rules()
    print(x)