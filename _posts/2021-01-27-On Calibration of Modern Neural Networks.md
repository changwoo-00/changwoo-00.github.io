---
layout: post
title: "Paper Summary"
use_math: true
---
# On Calibration of Modern Neural Networks

Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger

(정리 : 21-01-27)

## 1. Introduction

In real-world decision making systems, **classification networks must not only be accurate, but also should indicate when they are likely to be incorrect**. As an example, consider a self-driving car that uses a neural network to detect pedestrians and other obstructions (Bojarski et al., 2016). **If the detection network is not able to confidently predict the presence or absence of immediate obstructions**, the car should rely more on the output of other sensors for braking. Alternatively, **in automated health care, control should be passed on to human doctors when the confidence of a disease diagnosis network is low** (Jiang et al., 2012). Specifically, a network should provide a calibrated confidence measure in addition to its prediction. In other words, the probability associated with the predicted class label should reflect its ground truth correctness likelihood.

Calibrated confidence의 장점

provide a valuable extra bit of information to establish trustworthiness with the user

Further, good probability estimates can be used to incorporate neural networks into other probabilistic models.

While neural networks today are undoubtedly more accurate than they were a decade ago, we discover with great surprise that **modern neural networks are no longer well-calibrated.**

Surprisingly, we find that a single-parameter variant of Platt scaling (Platt et al., 1999) – which we refer to as **temperature scaling – is often the most effective method at obtaining calibrated probabilities.**

## 2. Definitions

$\hat{Y}$ : class prediction

$\hat{P}$ : associated confidence, i.e. probability of correctness

We would like the confidence estimate $\hat{P}$ to be calibrated, which intuitively means that $\hat{P}$ represents a true probability.

For example, given 100 predictions, each with confidence of  0.8, we expect that 80 should be correctly classified. More formally, we define *perfect calibration* as

<center>
    <figure>
        <img src="D:\Experimental\githubio\hcw-00.github.io\assets\2021-01-27-On Calibration of Modern Neural Networks/Untitled.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 1</figcaption>
    </figure>
</center>

 (위) 110-layer ResNet이 5-layer LeNet에 비해 크게 개선되었지만 average confidence와 accuracy의 차이가 크다. (아래) accuracy as a function of confidence.

### Reliability Diagrams

If the model is perfectly calibrated – i.e. if (1) holds – then the diagram should plot the identity function.

To estimate the expected accuracy from finite samples, we group predictions into $M$ interval bins and calculate the accuracy of each bin.

Let $B_m$ **be the set of indices of samples whose prediction confidence falls into
the interval** $I_{m}=\left(\frac{m-1}{M}, \frac{m}{M}\right]$.  The accuracy of $B_m$ is 

$$\operatorname{acc}\left(B_{m}\right)=\frac{1}{\left|B_{m}\right|} \sum_{i \in B_{m}} \mathbf{1}\left(\hat{y}_{i}=y_{i}\right)$$

where $\hat{y}_i$ and $y_i$ are the predicted and true class labels for sample $i$. 

Basic probability tells us that $\text{acc}(B_m)$ is an unbiased and consistent estimator of $\mathbb{P}(\hat{Y} = Y \| \hat{P} \in I_m)$ .

We define the average confidence within bin $B_m$ as 

$$\operatorname{conf}\left(B_{m}\right)=\frac{1}{\left|B_{m}\right|} \sum_{i \in B_{m}} \hat{p}_{i}$$

where $\hat{p}_i$ is the confidence for sample $i$.  ($\hat{p}_i$ : predicted probability of $y_i = 1$, $\hat{p}_i = \sigma(z_i)$)

$\text{acc}(B_m)$ and $\text{conf}(B_m)$ approximate the left-hand and right-hand sides of (1) respectively for bin $B_m$.

a perfectly calibrated model will have $\text{acc}(B_m) = \text{conf}(B_m)$ for all
$m \in\{1, \ldots, M\}$.

### Expected Calibration Error (ECE)

One notion of miscalibration is the difference in expectation between confidence and accuracy, i.e.

$$\underset{\hat{P}}{\mathbb{E}}[|\mathbb{P}(\hat{Y}=Y \mid \hat{P}=p)-p|]$$

Expected Calibration Error or ECE (partitioning predictions into $M$ equally-spaced bins)

$$\mathrm{ECE}=\sum_{m=1}^{M} \frac{\left|B_{m}\right|}{n}\left|\operatorname{acc}\left(B_{m}\right)-\operatorname{conf}\left(B_{m}\right)\right|$$

where $n$ is the number of samples.

The difference between $\text{acc}$ and $\text{conf}$ for a given bin represents the calibration **gap**
(red bars in reliability diagrams – e.g. Figure 1).

We use ECE as the primary empirical metric to measure calibration.

### Maximum Calibration Error (MCE)

In high-risk applications ... we may wish to minimize the worst-case deviation between confidence and accuracy:

$$\max _{p \in[0,1]}|\mathbb{P}(\hat{Y}=Y \mid \hat{P}=p)-p|$$

The Maximum Calibration Error (Naeini et al., 2015) or MCE

$$\mathrm{MCE}=\max {m \in\{1, \ldots, M\}}\left|\operatorname{acc}\left(B{m}\right)-\operatorname{conf}\left(B_{m}\right)\right|$$

MCE is the largest calibration gap (red bars) across all bins, whereas ECE is a weighted average of all gaps.

### Negative log likelihood

$$\mathcal{L}=-\sum_{i=1}^{n} \log \left(\hat{\pi}\left(y_{i} \mid \mathbf{x}_{i}\right)\right)$$

## 3. Observing Miscalibration

<center>
    <figure>
        <img src="D:\Experimental\githubio\hcw-00.github.io\assets\2021-01-27-On Calibration of Modern Neural Networks/Untitled201.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 2</figcaption>
    </figure>
</center>


### Model capacity

During training, after the model is able to correctly classify (almost) all training samples, NLL can be further minimized by increasing the confidence of predictions. Increased model capacity will lower training NLL, and thus the model will be more (over)confident on average.

### Batch Normalization

we do observe that models trained with Batch Normalization tend to be more miscalibrated.

We find that this result holds regardless of the hyperparameters used on the Batch Normalization model (i.e. low or high learning rate, etc.).

### Weight decay

However, due to the apparent regularization effects of Batch Normalization, recent research seems to suggest that models with less L2 regularization tend to generalize better (Ioffe & Szegedy, 2015). As a result, it is now common to train models with little weight decay, if any at all.

We find that training with less weight decay has a negative impact on calibration.

### NLL

...

## 4. Calibration Methods

...

...

### Temperature scaling

the simplest extension of Platt scaling, uses a single scalar parameter $T > 0$ for all classes.

$$\hat{q}_{i}=\max_{k} \sigma_{\mathrm{SM}}\left(\mathbf{z}_{i} / T\right)^{(k)}$$

$T$ is called the temperature, and it "softens" the softmax (i.e. raises the output entropy) with $T > 1$. As $T → \infty$, the probability $\hat{q}_i$ approaches $1/K$, which represents maximum uncertainty. With $T = 1$, we recover the original probability $\hat{p}_i$. As $T → 0$, the probability collapses to a point mass (i.e. $\hat{q}_i = 1$).

$T$ is optimized with respect to NLL on the validation set.

## 5. Results

### Calibration Results

<center>
    <figure>
        <img src="D:\Experimental\githubio\hcw-00.github.io\assets\2021-01-27-On Calibration of Modern Neural Networks/Untitled202.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 3</figcaption>
    </figure>
</center>

<!-- ![On%20Calibration%20of%20Modern%20Neural%20Networks%208398f8ee2acb4b8d8c16753c977d444d/Untitled%202.png](On%20Calibration%20of%20Modern%20Neural%20Networks%208398f8ee2acb4b8d8c16753c977d444d/Untitled%202.png) -->

### Reliability diagrams

<center>
    <figure>
        <img src="D:\Experimental\githubio\hcw-00.github.io\assets\2021-01-27-On Calibration of Modern Neural Networks/Untitled203.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 4</figcaption>
    </figure>
</center>

<!-- ![On%20Calibration%20of%20Modern%20Neural%20Networks%208398f8ee2acb4b8d8c16753c977d444d/Untitled%203.png](On%20Calibration%20of%20Modern%20Neural%20Networks%208398f8ee2acb4b8d8c16753c977d444d/Untitled%203.png) -->

### Computation time

All methods scale linearly with the number of validation set samples. Temperature scaling is by far the fastest method, as it amounts to a one-dimensional convex optimization problem.

### Ease of implementation

While all other methods are relatively easy to implement, temperature scaling may arguably be the most straightforward to incorporate into a neural network pipeline.

---

keywords : temperature scaling, over confidence,