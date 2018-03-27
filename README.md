Source code for my master's thesis *Neural Networks in Dialogue Systems* at University of Wrocław, Poland.

`SQuAD/` - neural question answerer

`generators/` - dialogue generation

`thesis/` - thesis (in Polish), with latex source

Run on Python 2.7.13. Install `requirements.txt`. You might not be able to install development version of Lasagne with pip, use
`pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip` instead. Pre-trained models are provided. If you want to train the models yourself, you will need to set up your GPU for Theano (I used CUDA). I had problems with gradient optimizations in generators on Theano 1.0.1 and had to use 0.9 to train them.
