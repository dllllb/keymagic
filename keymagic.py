import numpy as np
import math
import random
import re

def arc(beg, end, details=100.):
    xb, yb = beg
    xe, ye = end
    dx = xe - xb
    dy = ye - yb
    dist = (dx**2 + dy**2)**.5

    arc_r = dist/2. + 300.

    dist_center_x = xb + dx/2.
    dist_center_y = yb + dy/2.

    arc_x = dist_center_x - math.sqrt(arc_r**2-(dist/2.)**2)*dy/dist
    arc_y = dist_center_y + math.sqrt(arc_r**2-(dist/2.)**2)*dx/dist

    n_steps = 4+int(dist/details)

    beg_angle = math.atan2(arc_y-yb, arc_x-xb)
    end_angle = math.atan2(arc_y-ye, arc_x-xe)

    line = []
    for step in range(n_steps+1):
        dist = abs(end_angle - beg_angle)
        if dist > math.pi:
            dist = 2*math.pi-dist
        step_angle = beg_angle + dist*step/(n_steps)
        
        step_point_x = arc_x - arc_r*math.cos(step_angle)
        step_point_y = arc_y - arc_r*math.sin(step_angle)

        step_point = (step_point_x, step_point_y)
        line.append(step_point)
    return line

def keyboardIOS7():
    button_sz = (50, 75)

    buttons = []

    nex_pos_dist = 64

    for i in range(7, nex_pos_dist*10, nex_pos_dist):
        buttons.append((i, 25))

    for i in range(38, nex_pos_dist*9, nex_pos_dist):
        buttons.append((i, 133))

    for i in range(103, nex_pos_dist*8, nex_pos_dist):
        buttons.append((i, 240))

    letters = 'qwertyuiopasdfghjklzxcvbnm'
    return dict(zip(letters, zip(buttons, [button_sz for i in range(len(letters))])))

def word_to_line(kbrd, word, randomness=0, details=10., max_points=1000):
    word = re.sub("[^a-zA-Z]", '', word).lower()
    
    fixed_word = []
    prev = ''
    for l in word:
        if l != prev:
            prev = l
            fixed_word.append(l)

    line = []
    rnd = zip(np.random.randn(len(fixed_word)), np.random.randn(len(fixed_word)))
    for letter, shift in zip(fixed_word, rnd):
        pos, sz = kbrd[letter]
        center = (pos[0] + sz[0]/2. + shift[0]*randomness, pos[1] + sz[0]/2. + shift[1]*randomness)
        line.append(center)
    
    line_full = []
    for beg, end in zip(line[:-1], line[1:]):
        line_full.extend(arc(beg, end, details))
    
    if max_points < len(line_full):
        line_full = [line_full[i] for i in sorted(random.sample(range(len(line_full)), max_points))]
        
    return line_full


def generate_input(word, n_points):
    details = 0.05*len(word)
    l = word_to_line(kbrd, word, randomness=10, details=details, max_points=n_points)
    letters = list()
    for x, y in l:
        if x <= km.shape[0] and y <= km.shape[1]:
            letters.append(km[x, y])
        else:
            letters.append(-1)
    lx, ly = zip(*l)
    return np.array(zip(lx, ly, letters), dtype=np.short)


def generate_dataset_words(input_text, char_ind, n_points=100, max_letters=20):
    words = [
        w for w in input_text.split()
        if len(w.translate(None, "'-")) > 1
        and len(set(w)) > 1
        and len(w) <= max_letters
    ]
    
    input_len = sum([len(w) for w in words]) + len(words)

    X = np.zeros((len(words), n_points, 3), dtype=np.short)
    y = np.zeros((len(words), max_letters, n_chars), dtype=np.bool)

    shift = 0
    for word, w_pos in zip(words, range(len(words))):
        try:
            line = generate_input(word, n_points)
            X[w_pos] = line
            out = word + ' '*(max_letters-len(word))
            for char, c_pos in zip(out, range(len(out))):
                y[w_pos, c_pos, char_ind[char]] = 1
        except Exception:
            print(word)
            raise
        
    return X, y