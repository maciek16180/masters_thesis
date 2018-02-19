from __future__ import print_function

import os
import time
import numpy as np


def pretrain(net, output_path, num_epochs, train, valid, bs, log_interval):
    model_filename = os.path.join(output_path, 'model')

    t0 = time.time()
    for epoch in range(1, num_epochs + 1):
        print('\n\nStarting epoch {}...\n'.format(epoch))
        train_error = net.train_one_epoch(
            train_data=train,
            batch_size=bs,
            log_interval=log_interval)
        val_error = net.validate(
            val_data=valid,
            batch_size=bs)
        print('\nTraining loss:   {}'.format(train_error))
        print('MT validation loss: {}'.format(val_error))
        net.save_params(model_filename + '.ep{:02d}'.format(epoch))

    print('\n\nTotal training time: {:.2f}s'.format(time.time() - t0))


def train(net, output_path, train, valid, test, bs, log_interval,
          patience=5, tol=0.00001):
    last_scores = [np.inf]
    epoch = 1
    best_epoch = None

    model_filename = os.path.join(output_path, 'model')

    t0 = time.time()
    while len(last_scores) <= patience or \
            last_scores[0] > min(last_scores) + tol:
        print('\n\nStarting epoch {}...\n'.format(epoch))
        train_error = net.train_one_epoch(
            train_data=train,
            batch_size=bs,
            log_interval=log_interval)
        val_error = net.validate(
            val_data=valid,
            batch_size=bs)
        print('\nTraining loss:   {}'.format(train_error))
        print('Validation loss: {}'.format(val_error))

        if val_error < min(last_scores):
            net.save_params(model_filename + '.ep{:02d}'.format(epoch))
            best_epoch = epoch
            print('\nBest score so far, model saved.')

        last_scores.append(val_error)

        if len(last_scores) > patience + 1:
            del last_scores[0]
        epoch += 1

    test_error = net.validate(
        val_data=test,
        batch_size=bs)

    print('\n\nTotal training time: {:.2f}s'.format(time.time() - t0))
    print('Best model after {} epochs with loss {}'.format(
        best_epoch, min(last_scores)))
    print('Validation set perplexity: {}'.format(np.exp(min(last_scores))))
    print('Model saved as ' + model_filename)

    print('\nTest loss: {}'.format(test_error))
    print('Test set perplexity: {}'.format(np.exp(test_error)))
