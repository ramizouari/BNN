# BNN

## 1. Introduction

Binary neural network constitute a class of very lightweight neural networks, which can speed up the model inference speed by $32 \le k < 64$ folds, and can reduce its memory consumption by $32$ folds.

As an example GoogleNet 

## 2. Rationale

As the power of computation increases, more and more capable AI models are published. But, the sole major problem is their complexity.

Those algorithms may be deployed on a powerful computer with a GPU (or TPU) for fast interference calculation. But the energy consumption of this approach is very discouraging as AI models are beginning to use more energy than modest usines.



Also these models generally can never be equipped on a small device such as a mobile phone, smartwatch, and especially embedded devices.



So BNN can be seen as a form of model compression that helps to alleviate these major problems, and a bridge between AI and the embedded world.



## 3. Definitions

### 3.1 Binarisation

For a vector space $E$ over $\mathbb{R}$ with $\dim E=n,$ a binarisation of a variable $x\in E$ is a mapping $\Psi:E\rightarrow B^n$  where $B$ is a set containing only two values

The two most known binarisations classes are:

- $B=\{\pm 1\}$
- $B=\{0,1\}$

In our approach, we will focus on the first class. Also, we will use the most straightforward binarisation:
$$
\DeclareMathOperator{\sign}{sign}
\Psi (x)=\sign x
$$
We call $\Psi(x)$ a binarisation of $x,$ or a binarised version of $x$

### 3.2 Binarised Layer 

A binarised layer is a layer whose parameters and nodes are binarised. By default, we will automatically use $\Psi=\sign$

### 3.3 Binary Neural Network

A binary neural network is a neural network that is composed of binarised layers.



## 4. Advantages of BNN

To be done