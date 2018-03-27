import sys

sys.path.append('../scripts/')

from squad_tools import load_glove
from AnswerBot import AnswerBot

glove_path = '/home/maciek/Desktop/masters_thesis/DATA/word vectors/glove.6B/glove.6B.300d.txt'
squad_base_path = '/home/maciek/Desktop/masters_thesis/DATA/squad/'

glove_words, glove_embs = load_glove(glove_path)

abot = AnswerBot(
    '../models/best/model.ep09.npz',
    glove_embs,
    glove_words,
    train_unk=True,
    negative=False,
    conv='valid')

abot_neg = AnswerBot(
    '../models/best_neg/model.ep07.npz',
    glove_embs,
    glove_words,
    train_unk=True,
    negative=True,
    conv='full')

def pr(s):
    return u' '.join(s)

def ans(qs, x):
    for q in qs:
        print q
        a, p = abot.get_answers([q], [x])[0]
        print pr(a), p, '\n'
        
def ans_neg(qs, x):
    for q in qs:
        print q
        a, p = abot_neg.get_answers([q], [x])[0]
        print pr(a), p, '\n'