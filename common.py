import numpy as np
import math
import random
import re
from PIL import Image, ImageDraw


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
    keyboard_img = 'ios-7-1-keyboard.jpg'
    
    with Image.open(keyboard_img) as im:
        im_size = im.size
        
    button_sz = (50, 75)

    nex_pos_dist = 64

    buttons = []
    for i in range(7, nex_pos_dist*10, nex_pos_dist):
        buttons.append((i, 25))

    for i in range(38, nex_pos_dist*9, nex_pos_dist):
        buttons.append((i, 133))

    for i in range(103, nex_pos_dist*8, nex_pos_dist):
        buttons.append((i, 240))

    letters = 'qwertyuiopasdfghjklzxcvbnm'
    
    btn = dict(
        zip(letters, zip(buttons, [button_sz for i in range(len(letters))]))
    )
    
    char_indices = dict((c, i) for i, c in enumerate(letters+' '))
    indices_char = dict((i, c) for i, c in enumerate(letters+' '))
    km = key_matrix(btn, im_size)
    
    return {
        'img': keyboard_img,
        'size': im_size,
        'buttons': btn,
        'char_indices': char_indices,
        'indices_char': indices_char,
        'key_matrix': km,
    }


def word_to_line(buttons, word, randomness=0, details=10., max_points=1000):
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
        pos, sz = buttons[letter]
        center = (pos[0] + sz[0]/2. + shift[0]*randomness, pos[1] + sz[0]/2. + shift[1]*randomness)
        line.append(center)
    
    line_full = []
    for beg, end in zip(line[:-1], line[1:]):
        line_full.extend(arc(beg, end, details))
    
    if max_points < len(line_full):
        line_full = [line_full[i] for i in sorted(random.sample(range(len(line_full)), max_points))]
        
    line_full = [(int(np.round(x)), int(np.round(y))) for x, y in line_full]
    return line_full


def generate_input(word, n_points, kbrd):
    buttons = kbrd['buttons']
    km = kbrd['key_matrix']
    
    details = 0.05*len(word)
    l = word_to_line(buttons, word, randomness=10, details=details, max_points=n_points)
    letters = list()
    for x, y in l:
        if x <= km.shape[0] and y <= km.shape[1]:
            letters.append(km[x, y])
        else:
            letters.append(-1)
    lx, ly = zip(*l)
    return np.array(list(zip(lx, ly, letters)), dtype=np.short)


def predict(model, kbrd, curve):
    p = model.predict(np.array([curve]))
    id2char = kbrd['indices_char']
    return ''.join([id2char[ix] for ix in np.argmax(p[0], axis=1)]).strip()


def generate_dataset_words(props, kbrd, char_limit=None):
    char_ind = kbrd['char_indices']
    n_chars = props['n_chars']
    
    with open(props['train_data']) as f:
        if char_limit is not None:
            text = f.read(char_limit)
        else:
            text = f.read()

    input_text = re.sub("[^a-zA-Z'-]", ' ', text).lower()
    
    n_points=props['n_points']
    max_letters=props['max_letters']
    
    trans_table = {ord(c): None for c in "'-"}
    
    words = [
        w.translate(trans_table) for w in input_text.split()
        if len(w) > 1
        and len(set(w)) > 1
        and len(w) <= max_letters
    ]
    
    input_len = sum([len(w) for w in words]) + len(words)

    X = np.zeros((len(words), n_points, 3), dtype=np.short)
    y = np.zeros((len(words), max_letters, n_chars), dtype=np.bool)

    shift = 0
    for word, w_pos in zip(words, range(len(words))):
        try:
            line = generate_input(word, n_points, kbrd)
            X[w_pos] = line
            out = word + ' '*(max_letters-len(word))
            for char, c_pos in zip(out, range(len(out))):
                y[w_pos, c_pos, char_ind[char]] = 1
        except Exception:
            print(word)
            raise
        
    return X, y


def draw_keyboard(kbrd):
    im = Image.open(kbrd['img'])

    draw = ImageDraw.Draw(im)

    for pos, sz in kbrd['buttons'].values():
        bl = (pos[0], pos[1] + sz[1])
        br = (pos[0] + sz[0], pos[1] + sz[1])
        ur = (pos[0] + sz[0], pos[1])
        draw.line([pos, ur, br, bl, pos], (220, 100, 0), width=3)
        
    return im


def draw_word_line(word_line, img_path):
    im = Image.open(img_path)

    draw = ImageDraw.Draw(im)

    draw.line(word_line, (220, 100, 0), width=3)
        
    return im


def key_matrix(buttons, size):
    char_indices = dict((c, i) for i, c in enumerate(buttons.keys()))
        
    key_matrix = np.zeros(size, dtype=np.byte)
    key_matrix.fill(-1)
    
    for k, (pos, sz) in buttons.items():
        key_matrix[pos[0]:pos[0]+sz[0], pos[1]:pos[1]+sz[1]] = char_indices[k]
        
    return key_matrix


def dump_dataset(path, X, y):
    import h5py

    with h5py.File(path, 'w') as f:
        f.create_dataset('X', data=X, compression="gzip")
        f.create_dataset('y', data=y, compression="gzip")


def read_dataset(path):
    with h5py.File(path) as f:
        X = f['X'][:]
        y = f['y'][:]
        return X, y
