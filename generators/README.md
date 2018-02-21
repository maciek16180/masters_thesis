I used Theano 0.9 with device=cuda (there were some major problems with new backend), cuda 8.0, cudnn 5.1, and lasagne 0.2.

**Training**

To train the models you need MovieTriples and SubTle data sets, which aren't publicly available. See these papers for details:

* [1] Magarreiro et al. - [Using subtitles to deal with Out-of-Domain interactions](http://www.inesc-id.pt/publications/10328/pdf)
* [2] Serban et al. - [Building End-To-End Dialogue Systems Using Generative Hierarchical Neural Networks](https://arxiv.org/pdf/1507.04808.pdf)

Training procedure HRED and SimpleRNNLM is the same. This example is for HRED. Run

```
cd HRED/scripts
python -u pretrain.py -mt <mt_path>
```

and after it finishes

```
python -u train.py -mt <mt_path> -p <model> --fix_emb
```

This trains the sampled softmax model using SubTle bootstrapping described in [2]. To use full softmax instead, add `-m full` to script calls. Adjust batch size if needed: `-bs NUMBER`. Note that the model might perform differently. See the respective scripts for the detailed description of other parameters.

`<mt_path>` should be a directory containing preprocessed MovieTriples and SubTle data sets:

```
<mt_path>/
    Training.triples.pkl
    Validation.triples.pkl
    Test.triples.pkl
    Training.dict.pkl
    Word2Vec_WordEmb.pkl
    Subtle_Dataset.triples.pkl
```

After every epoch of pre-training, a model is saved to `output.pretrain/model.epXX.npz`. Choose the pre-trained model with the lowest MT validation loss (`output.pretrain/log`) and pass it to training as `<model>`. During training, after each epoch the model is saved if validation error is at its lowest. The latest model is the best and should be used in a demo.

**Demo**

This example is for HRED. SimpleRNNLM demo has the same interface.

To run dialogue demo, you need [interactive Python](https://ipython.org/). Run

```
cd HRED
ipython
%run demo -mt <mt_path> -m <model> -md <mode>
```

`<mode>` should be 'ssoft' or 'full', and has to match the model. After the network is constructed, run

```
talk()
```

This allows you to talk to a bot. To reset a conversation, press ctrl+c and run the `talk()` function again. All conversations are saved to `log` directory. Look up `demo.py` for the parameters' descriptions.