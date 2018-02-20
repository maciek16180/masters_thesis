I used Theano 0.9 with device=cuda, cuda 8.0, cudnn 5.1, and lasagne 0.2.

**Training**

To train the model from scratch, run from this directory

```
cd scripts
./train.sh <glove_path> <squad_path>
```

Edit BATH_SIZE inside a script if needed. `<squad_path>` should be a directory containing `dev-v1.1.json` and `train-v1.1.json` from https://rajpurkar.github.io/SQuAD-explorer/. `<glove_path>` should point to `glove.6B.300d.txt` from http://nlp.stanford.edu/data/glove.6B.zip. It's needed to create vocabulary.

Best model is saved to `output/6B.best.npz`. For validation use `verification.ipynb` notebook.

**Negative training**

To recreate negative answers experiment, download negative data sets from the data pack. The directory scructure should be as follows:

```
<squad_path>/
    dev-v1.1.json
    train-v1.1.json
    negative_samples/
        dev.wiki.pos.json
        dev.wiki.neg.json
        dev.squad.random.json
        train.wiki.pos.json
        train.wiki.neg.json
        train.squad.random.json
```

Run

```
cd scripts
python -u prep_squad.py --glove=<glove_path> --squad=<squad_path>
python -u prep_squad_neg.py --glove=<glove_path> --squad=<squad_path>
python -u train.py --glove=<glove_path> --squad=<squad_path> --negative <list_of_negative_data_sets>
```

Add `--batch_size=NUMBER` to the last line to change batch size. `<list_of_negative_data_sets>` should be space-separated list of names of negative data sets to use. It has to be a subset of

```
squad_neg_cut
squad_neg_rng
wiki_neg
wiki_pos
```

Data sets are described in my master's thesis (Polish). Unprocessed files are in `negative_samples/`.

Model is saved as `output/6B.best.neg.npz`.

To test a negative model, run:

```
cd scripts
python -u test_neg.py --glove=<glove_path> --squad=<squad_path> --model=<path_to_model.npz>
```

Result is saved in `output/neg_test`.
