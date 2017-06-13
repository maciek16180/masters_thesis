# for use with HRED

import numpy as np
import sys
sys.path.append('../rnn_ex/')

from mt_load import load_mt, get_mt_voc, get_w2v_embs


mt_path = "/pio/data/data/mtriples/"
idx_to_w, w_to_idx, voc_size, _ = get_mt_voc(path=mt_path, train_len=0)

pad_value = -1

def softmax(x):
    assert len(x.shape) == 2
    x = np.exp(x)
    return x / x.sum(axis=1)[:, np.newaxis]

def diverse_beam_search(beam, gs, dec_init, voc_size, hred_net, init_seq=np.array([[1]]), rank_penalty=0,
                        group_diversity_penalty=1, seq_diversity_penalty=1, verbose_log=False, unk_penalty=100,
                        sample=False, sharpen_probs=None, only_last_groups=False):
    assert not beam % gs
    num_groups = beam / gs
    
    seq = np.repeat(init_seq.astype(np.int32), beam, axis=0)
        
    scores = np.zeros(beam)
    
    finished = []
    
    while seq.shape[1] < 50:
        all_probs, all_dec_init = hred_net.get_probs_and_new_dec_init_fn(seq[:,-1:], dec_init)
        
        new_seq = np.zeros((0, seq.shape[1] + 1), dtype=np.int32)
        new_dec_inits = []
        next_scores = []
        
        for g in xrange(num_groups):
            g_idx = slice(gs * g, gs * (g + 1))
            log_probs = np.log(all_probs[g_idx])
            
            if unk_penalty is not None:
                log_probs[:, w_to_idx['<unk>']] -= unk_penalty
                log_probs[:, w_to_idx['<number>']] -= unk_penalty
                log_probs[:, w_to_idx['<person>']] -= unk_penalty
                log_probs[:, w_to_idx['<continued_utterance>']] -= unk_penalty
                
            dec_init = all_dec_init[g_idx]
            
            # here we add the dissimilarity as described in https://arxiv.org/pdf/1610.02424.pdf            
            # simple Hamming diversity
            log_probs[:, new_seq[:, -1]] -= group_diversity_penalty
            
            # penalize repeating words in the same sequence
            log_probs[np.indices((gs, seq.shape[1]))[0], seq[g_idx]] -= seq_diversity_penalty
            
            words = np.arange(voc_size)[np.newaxis].repeat(gs, axis=0).astype(np.int32)
                
            next_word_scores = log_probs[np.indices((gs, voc_size))[0], words]

            new_scores = next_word_scores + scores[g_idx, np.newaxis]
            
            # this line is for implementing rank penalty: https://arxiv.org/abs/1611.08562
            # it doesn't work correctly at the moment
            new_scores = (new_scores - (new_scores.argsort(axis=1) + 1) * rank_penalty).ravel()
    
            new_scores = new_scores.ravel()
        
            if not sample:
                if seq.shape[1] == 1:
                    order = np.array(-new_scores[:voc_size]).argsort().astype(np.int32)
                else:
                    order = (-new_scores).argsort().astype(np.int32)
            else:
                count = new_scores.size
                scr = new_scores if sharpen_probs is None else -(-new_scores**sharpen_probs)
                order = np.random.choice(count, size=2*gs**2, replace=False, p=softmax(scr[np.newaxis])[0])

            #print "###############"
            #print new_seq.shape, order.size, new_scores.shape, next_word_scores.shape, scores.shape
            
            for idx in order:
                if new_seq.shape[0] == (g + 1) * gs:
                    break

                i,j = divmod(idx, voc_size)

                #print seq.shape, gs * g + i, i, j
                #print words
                
                extended_seq = np.concatenate([seq[gs * g + i], np.array([words[i,j]])])
                if extended_seq[-1] == w_to_idx['</s>']:
                    if not only_last_groups or g >= num_groups / 2:
                        finished.append((extended_seq, new_scores[idx]))
                else:
                    new_seq = np.vstack([new_seq, extended_seq])
                    new_dec_inits.append(dec_init[i])
                    next_scores.append(new_scores[idx])
                    
            if new_seq.shape[0] < (g + 1) * gs:
                pad_len = (g + 1) * gs - new_seq.shape[0]
                new_seq = np.vstack([new_seq, np.ones((pad_len, new_seq.shape[1]))]).astype(np.int32)
                new_dec_inits.append(np.zeros((pad_len, dec_init.shape[1]), dtype=np.float32))
                next_scores += [-np.inf] * pad_len

        if not new_seq.size:
            print 'Ending...'
            break
                
        seq = new_seq
        scores = np.array(next_scores)
        dec_init = np.vstack(new_dec_inits)
    
        if verbose_log:
            print 'Length ', seq.shape[1], '\n'
            for utt, s in zip(seq, scores):
                print '{:.4f} {}'.format(s, print_utt(utt))
                print ''
            print '#############\n'
            

#     final_scores = np.array(map(lambda x: x[1], finished))
#     finished = map(lambda x: x[0], finished)
    
    return finished#[final_scores.argmax()]
