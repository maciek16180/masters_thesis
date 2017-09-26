import time, sys
sys.path.append('../../')
sys.path.append('../../../')
from HRED import HRED

from data_load.redditv3_load import get_redditv3_voc, load_pairs
from data_load.subtle_fixed_load import load_subtle_fixed

subtle_fixed_path = "/pio/data/data/subtle/"
redditv3_path = "/pio/data/data/reddit_sample/v3/"

subtle_data = load_subtle_fixed(path=subtle_fixed_path, trim=200)
train, _ = load_pairs(redditv3_path, threeD=False)
_, _, voc_size, freqs = get_redditv3_voc(path=redditv3_path, train_len=len(train))


net = HRED(voc_size=voc_size,
           emb_size=300,
           lv1_rec_size=300,
           lv2_rec_size=300,
           out_emb_size=300,
           num_sampled=200,
           ssoft_probs=freqs,
           skip_gen=True)

num_epochs = 10

model_filename = '../trained_models/test2/pretrained_subtle_fixed_GaussInit_300_300_300_300_ssoft200unigr_bs30_cut200'

t0 = time.time()
for epoch in xrange(num_epochs):
    print '\n\nStarting epoch {}...\n'.format(epoch)
    train_error = net.train_one_epoch(train_data=subtle_data, batch_size=30, log_interval=1000)
    print '\nTraining loss:   {}'.format(train_error)
    net.save_params(model_filename + '_ep' + str(epoch + 1) + '.npz')

print '\n\nTotal training time: {:.2f}s'.format(time.time() - t0)
print 'Model saved as ' + model_filename
