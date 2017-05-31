import numpy as np
import io
import lasagne as L


def load_squad_train(path, with_chars=False):
    return np.load(path + 'train.pkl'), np.load(path + 'train_char.pkl')

def get_squad_train_voc(path, pad_value=-1):
    w_to_i = {}
    i_to_w = []
    
    idx = 0
    with io.open(path + 'train_wordlist.txt', 'r', encoding='utf-8') as f:
        for line in f:
            w = line[:-1]
            w_to_i[w] = idx
            i_to_w.append(w)
            idx += 1
            
    w_to_i['<pad_value>'] = pad_value
    i_to_w.append('<pad_value>')

    return i_to_w, w_to_i, len(w_to_i)

def get_squad_train_chars(path, pad_value=-1):
    c_to_i = {}
    i_to_c = []
    
    idx = 0
    with io.open(path + 'train_charlist.txt', 'r', encoding='utf-8') as f:
        for line in f:
            c = line[:-1]
            c_to_i[c] = idx
            i_to_c.append(c)
            idx += 1
            
    c_to_i['<pad_value>'] = pad_value
    i_to_c.append('<pad_value>')

    return i_to_c, c_to_i, len(c_to_i)

def get_glove_train_embs(path, glove_path):
    glove_words = []
    with io.open(glove_path + 'glove.6B.wordlist.txt', 'r', encoding='utf-8') as f:
        for line in f:
            glove_words.append(line[:-1])
            
    i_to_w, _, voc_size = get_squad_train_voc(path)
            
    glove_vec = np.load(glove_path + 'glove.6B.300d.npy')
    
    glove_w_to_i = {v:k for (k,v) in enumerate(glove_words)}

    embs = np.zeros((voc_size, 300), dtype=np.float32)
    
    known_inds = [i for i in xrange(voc_size) if i_to_w[i] in glove_w_to_i]
    s = set(known_inds)
    unknown_inds = [i for i in xrange(voc_size) if i not in s]

    embs[known_inds] = glove_vec[[glove_w_to_i[i_to_w[i]] for i in known_inds]]
    embs[unknown_inds] = L.init.Normal()((len(unknown_inds), 300))
    
    return embs