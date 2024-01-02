

# Causal Inference and Discovery in Python

<a href="https://www.packtpub.com/product/causal-inference-and-discovery-in-python/9781804612989"><img src="https://content.packt.com/B18993/cover_image_small.jpg" alt="Causal Inference and Discovery in Python" height="256px" align="right"></a>

This is the code repository for [Causal Inference and Discovery in Python](https://www.packtpub.com/product/causal-inference-and-discovery-in-python/9781804612989), published by Packt.

**Unlock the secrets of modern causal machine learning with DoWhy, EconML, PyTorch and more**

## What is this book about?

Causal methods present unique challenges compared to traditional machine learning and statistics. Learning causality can be challenging, but it offers distinct advantages that elude a purely statistical mindset. Causal Inference and Discovery in Python helps you unlock the potential of causality.

You’ll start with basic motivations behind causal thinking and a comprehensive introduction to Pearlian causal concepts, such as structural causal models, interventions, counterfactuals, and more. Each concept is accompanied by a theoretical explanation and a set of practical exercises with Python code.

Next, you’ll dive into the world of causal effect estimation, consistently progressing towards modern machine learning methods. Step-by-step, you’ll discover Python causal ecosystem and harness the power of cutting-edge algorithms. You’ll further explore the mechanics of how “causes leave traces” and compare the main families of causal discovery algorithms.

The final chapter gives you a broad outlook into the future of causal AI where we examine challenges and opportunities and provide you with a comprehensive list of resources to learn more.

This book covers the following exciting features: 
* Master the fundamental concepts of causal inference
* Decipher the mysteries of structural causal models
* Unleash the power of the 4-step causal inference process in Python
* Explore advanced uplift modeling techniques
* Unlock the secrets of modern causal discovery using Python
* Use causal inference for social impact and community benefit

If you feel this book is for you, get your [copy](https://www.amazon.com/Causal-Inference-Discovery-Python-learning/dp/1804612987/ref=sr_1_1?keywords=Causal+Inference+and+Discovery+in+Python&s=books&sr=1-1) today!


## Instructions and Navigations
All of the code is organized into folders.

The code will look like the following:
```
preds = causal_bert.inference(
    texts=df['text'],
    confounds=df['has_photo'],
)[0]
```

**Following is what you need for this book:**

This book is for machine learning engineers, data scientists, and machine learning researchers looking to extend their data science toolkit and explore causal machine learning. It will also help developers familiar with causality who have worked in another technology and want to switch to Python, and data scientists with a history of working with traditional causality who want to learn causal machine learning. It’s also a must-read for tech-savvy entrepreneurs looking to build a competitive edge for their products and go beyond the limitations of traditional machine learning.

With the following software and hardware list you can run all code files present in the book (Chapter 1-15).

### Software and Hardware List

| Chapter  | Software required                                                                    | OS required                        |
| -------- | -------------------------------------------------------------------------------------| -----------------------------------|
|  	1-15	   |  Python 3.9 | Windows macOS, or Linux |
|  	1-15	   | DoWhy 0.8 | Windows, macOS, or Linux |
|  	1-15	   | EconML 0.12.0 | Windows, macOS, or Linux |
|  	1-15	   | CATENets 0.2.3 | Windows, macOS, or Linux | 
|  	1-15	   | gCastle 1.0.3 | Windows, macOS, or Linux |
|  	1-15	   | Causica 0.2.0 | Windows, macOS, or Linux |
|  	1-15	   | Causal-learn 0.1.3.3 | Windows, macOS, or Linux |
|  	1-15	   | Transformers 4.24.0 | Windows, macOS, or Linux |


## Join our Discord server <img alt="Coding" height="25" width="32"  src="https://cliply.co/wp-content/uploads/2021/08/372108630_DISCORD_LOGO_400.gif">

Join our Discord community to meet like-minded people and learn alongside more than 2000 members at [Discord](https://packt.link/infer) <img alt="Coding" height="15" width="35"  src="https://media.tenor.com/ex_HDD_k5P8AAAAi/habbo-habbohotel.gif">


### Related products <Other books you may enjoy>
* Hands-On Graph Neural Networks Using Python  [[Packt]](https://www.packtpub.com/product/hands-on-graph-neural-networks-using-python/9781804617526) [[Amazon]](https://www.amazon.com/Hands-Graph-Neural-Networks-Python/dp/1804617520/ref=sr_1_1?keywords=Hands-On+Graph+Neural+Networks+Using+Python&s=books&sr=1-1)
  
* Applying Math with Python - Second Edition  [[Packt]](https://www.packtpub.com/product/applying-math-with-python-second-edition/9781804618370) [[Amazon]](https://www.amazon.com/Applying-Math-Python-real-world-computational/dp/1804618373/ref=sr_1_1?keywords=Applying+Math+with+Python+-+Second+Edition&s=books&sr=1-1)
  
## Get to Know the Author
[**Aleksander Molak**](https://www.linkedin.com/in/aleksandermolak/) is a Machine Learning Researcher and Consultant who gained experience working with Fortune 100, Fortune 500, and Inc. 5000 companies across Europe, the USA, and Israel, designing and building large-scale machine learning systems. On a mission to democratize causality for businesses and machine learning practitioners, Aleksander is a prolific writer, creator, and international speaker. As a co-founder of Lespire, an innovative provider of AI and machine learning training for corporate teams, Aleksander is committed to empowering businesses to harness the full potential of cutting-edge technologies that allow them to stay ahead of the curve.
He's the host of the Causal AI-centered [Causal Bandits Podcast](https://causalbanditspodcast.com/).




# Note from the Author:

## Environment installation
1. See the section **Using `graphviz` and GPU** below

2. To install the basic environment run: `conda env create -f causal_book_py39_cuda117.yml`

3. To install the environment for notebook `Chapter_11.2.ipynb` run: `conda create -f causal-pymc.yml`

## Selecting the kernel
After a successful installation of the environment, open your notebook and select the kernel `causal_book_py39_cuda117`

For notebook `Chapter_11.2.ipynb` change kernel to `causal-pymc`

## Using `graphviz` and GPU

**Note**: Depending on your system settings, you might need to install `graphviz` manually in order to recreate the graph plots in the code. 
Check https://pypi.org/project/graphviz/ for instructions 
specific to your operating system.

**Note 2**: To use GPU you'll need to install CUDA 11.7 drivers.
This can be done here: https://developer.nvidia.com/cuda-11-7-0-download-archive

## Citation

### BibTeX
```{bibtex}
@book{Molak2023,
    title={Causal Inference and Discovery in Python: Unlock the secrets of modern causal machine learning with DoWhy, EconML, PyTorch and more},
    author={Molak, Aleksander},
    publisher={Packt Publishing},
    address={Birmingham},
    edition={1.},
    year={2023},
    isbn={1804612987},
    note={\url{https://amzn.to/3RebWzn}}
}
```

### APA
```
Molak, A. (2023). Causal Inference and Discovery in Python: Unlock the secrets of modern causal machine learning with DoWhy, EconML, PyTorch and more. Packt Publishing.
```

## ‼️ Known mistakes // errata
For known errors and corrections check:

* [Books purchased before ~12:00 PM on June 13, 2023](https://github.com/PacktPublishing/Causal-Inference-and-Discovery-in-Python/blob/main/errata/Errata%20-%20Early%20Print%20(ordered%20before%20June%2013%202023).ipynb)

* [Books purchased after ~12:00 PM on June 13, 2023](https://github.com/PacktPublishing/Causal-Inference-and-Discovery-in-Python/blob/main/errata/Errata%20-%20Non-Early%20Print%20(ordered%20after%20June%2013%202023).ipynb)

If you spotted a mistake, let us know at book(at)causalpython.io or just open an **issue** in this repo. Thank you 🙏🏼
