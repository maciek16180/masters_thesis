import numpy as np
from nltk import word_tokenize
from QANet import QANet

'''
A question answering class.
params: 
    glove_path - path to directory with glove emeddings
    model_file - path fo npz file with a model
    train_inds - technical paratemer, a list of indices of words 
                 that were not fixed during training, has to match the model
                 
get_answers args:
    question - question string
    contexts - list of context strings
    beam - beam size for beam search (1 is fine in vast majority of cases)
    
    question and contexts are supposed to not be processed, just plain text
    
    returns: a list of pairs ([answer tokens], score), one pair for each context
'''

class AnswerBot:
    
    def __init__(self, glove_path='/pio/data/data/glove_vec/6B/', model_file='charemb_all_fixed_ep3.npz',
                 train_inds=[]):
    
        self.glove_path = glove_path
        self.model_file = model_file
    
        self.words = np.load(self.glove_path + 'glove.6B.wordlist.pkl')
        self.w_to_i = {v:k for (k,v) in enumerate(self.words)}
    
        self.glove_embs = np.load(self.glove_path + 'glove.6B.300d.npy')
        self.voc_size = self.glove_embs.shape[0]
        self.alphabet_size = 128
        self.emb_size = 300
        self.emb_char_size = 20
        self.num_emb_char_filters = 200
        self.rec_size = 300
        self.train_inds = train_inds
        
        self.chars = [unichr(i) for i in xrange(128)]
        self.c_to_i = {v:k for (k,v) in list(enumerate(self.chars))}
        
        self.qa_net = QANet(self.voc_size, self.alphabet_size, self.emb_size, self.emb_char_size, 
                            self.num_emb_char_filters, self.rec_size, self.train_inds, self.glove_embs,
                            skip_train_fn=True)
        
        self.qa_net.load_params(self.model_file, self.glove_embs)
        
        
    def get_answers(self, question, contexts, beam=1):
        
        num_contexts = len(contexts)
        
        def make_words(sample):
            q, xs = sample
            q_num = [self.w_to_i.get(w, 0) for w in q]    
            xs_num = [[self.w_to_i.get(w, 0) for w in x] for x in xs]
            return [[[], q_num, x_num] for x_num in xs_num]
        
        def make_chars(sample):
            q, xs = sample
            q_char = [[1] + [self.c_to_i.get(c, 0) for c in w] + [2] for w in q]
            xs_char = [[[1] + [self.c_to_i.get(c, 0) for c in w] + [2] for w in x] for x in xs]
            return [[q_char, x_char] for x_char in xs_char]
        
        def make_bin_feats(sample):
            q, xs = sample
            qset = set(q)
            return [[w in qset for w in x] for x in xs]
        
        def tokenize(s):
            return word_tokenize(s.lower())
        
        sample = [tokenize(question), map(tokenize, contexts)]
        data = make_words(sample), make_chars(sample), make_bin_feats(sample)
        
        l, r, scr = self._predict_span(data, beam, batch_size=num_contexts)
        
        answers = []
        for i in xrange(num_contexts):
            answers.append((sample[1][i][l[i]:r[i]+1], scr[i]))
            
        return answers
        
        
    def _predict_span(self, data, beam, batch_size=1, premade_bin_feats=True):
        num_examples = len(data[0])

        start_probs = self.qa_net.get_start_probs(data, batch_size, premade_bin_feats=premade_bin_feats)
        best_starts = start_probs.argpartition(-beam, axis=1)[:, -beam:].astype(np.int32)

        scores = start_probs[np.arange(num_examples)[:, np.newaxis], best_starts]
        scores = np.tile(scores[:, np.newaxis], (beam, 1)).transpose(0, 2, 1)

        best_ends_all = []
        for i in xrange(beam):
            end_probs = self.qa_net.get_end_probs(data, best_starts[:, i], batch_size,
                                                  premade_bin_feats=premade_bin_feats)
            best_ends = end_probs.argpartition(-beam, axis=1)[:, -beam:]
            scores[:, i, :] *= end_probs[np.arange(num_examples)[:, np.newaxis], best_ends]
            best_ends_all.append(best_ends)

        best_ends_all = np.hstack(best_ends_all)

        scores = scores.reshape(num_examples, beam**2)
        best_spans = scores.argmax(axis=1)
        starts = [i / beam for i in best_spans]

        starts = best_starts[np.arange(num_examples), starts]
        ends = best_ends_all[np.arange(num_examples), best_spans]

        return starts, ends, scores[np.arange(num_examples), best_spans]