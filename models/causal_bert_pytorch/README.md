# This version

This code comes originally from https://github.com/rpryzant/causal-bert-pytorch
At the time of writing the original code contained an error
(https://github.com/rpryzant/causal-bert-pytorch/issues/6)
that made using one of the methods (.ATE()) unadvisable.
This version of code fixes this error.

I also fixed minor errors in the original README.md content

Below starts the original README.md file:

# Causal Bert -- in Pytorch!
Pytorch implementation of ["Adapting Text Embeddings for Causal Inference" by Victor Veitch, Dhanya Sridhar, and David M. Blei](https://arxiv.org/pdf/1905.12741.pdf). 

# Quickstart

```
pip install -r requirements.txt
python CausalBert.py
```

This will train a system on some test data and calculate an average treatment effect (ATE). 

# Description

As input this system expects data where each row consists of:
* Freeform **text**
* A categorical variable (numerically coded) representing a **confound**
* A binary **treatment variable**
* A binary **outcome variable**

Then the system will give the text to BERT, and use the BERT embeddings + confound to predict
1) _P(T | C, text)_ 
2) _P(Y | T = 1, C, text)_
3) _P(Y | T = 0, C, text)_
4) The original masked language modeling objective of BERT. 

Once trained the resulting BERT embeddings will be sufficient for some causal inferences. 

# Example

```
df = pd.read_csv('testdata.csv')            
cb = CausalBertWrapper(batch_size=2,                       # init a model wrapper
    g_weight=0.1, Q_weight=0.1, mlm_weight=1)
cb.train(df['text'], df['C'], df['T'], df['Y'], epochs=1)  # train the model
print(cb.ATE(df['C'], df['text'], platt_scaling=True))     # use the model to get an average treatment effect
```


# Usage

**Initialize** the model wrapper (handles training and inference):

```    
cb = CausalBertWrapper(
  batch_size=2,   # batch size for training
  g_weight=1.0,   # loss weight for P(T | C, text) prediction head
  Q_weight=0.1,   # loss weight for P(Y | T, C, text) prediction heads
  mlm_weight=1)   # loss weight for original MLM objective
```

Then **train**
```
cb.train(
  df['text'],    # list of texts
  df['C'],       # list of confounds
  df['T'],       # list of treatments
  df['Y'],       # list of outcomes
  epochs=1)      # training epochs
```

Perform **inference**

```
( ( P(Y=1|T=1), P(Y=0|T=1)), ( P(Y=1|T=0), P(Y=0|T=0) ), ... =  cb.inference(
  df['text'],   # list of texts
  df['C'])      # list of confounds
```

Or estimate an **average treatment effect**

```
ATE = cb.ATE(
  df['text'],   # list of texts
  df['C'],      # list of confounds
  platt_scailing=False)    # https://en.wikipedia.org/wiki/Platt_scaling
```


