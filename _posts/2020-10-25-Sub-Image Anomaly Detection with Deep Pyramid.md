---
layout: post
title: "Sub-Image Anomaly Detection with Deep Pyramid Correspondences"
use_math: true
---

# Sub-Image Anomaly Detection with Deep Pyramid Correspondences - _arXiv 20_

<!-- 작성일자 : 201025 -->

*Niv Cohen, Yedid Hoshen*

[*https://arxiv.org/abs/2005.02357*](https://arxiv.org/abs/2005.02357)

# Introduction

pretrained 된 모델로부터 M개의 feature를 추출하여 train set에서 k nearest neighbor 와의 비교를 통해 anomaly를 검출하는 방식을 제안하였다.

pre-trained 모델을 직접 사용하기 때문에 따로 학습이 필요가 없으며 성능 또한 매우 우수함을 보였다.

# Correspondence-based Sub-Image Anomaly Detection

## Feature Extraction

Feature Extractor로 pre-trained ResNet(Wide-ResNet50x2)를 사용하였다.

\begin{equation}
	f_{i}=F\left(x_{i}\right)
\end{equation}

$F$ : global feature extractor, $x_i$ : image, $f_i$ : extracted feature 를 의미한다.

우선 모든 training image 에 대해 feature를 추출하여 저장해둔다.

inference 시에는 test image의 feature 만 추출한다.

## K Nearest Neighbor Normal Image Retrieval

첫번째 과정으로 test image가 anomaly를 포함하고 있는지를 판별한다.

test image $y$에 대해 training set 에서 $K$ nearest  image, $N_k(f_y)$ 를 추출한다.

distance는 image(feature)-level Euclidean metric을 사용하였다.

\begin{equation}
	d(y)=\frac{1}{K} \sum_{f \in N_{K}\left(f_{y}\right)}\left\|f-f_{y}\right\|^{2}
\end{equation}

여기서 image가 정상인지 비정상인지 threshold $\tau$를 기준으로 판정한다.

## Sub-image Anomaly Detection via Image Alignment

image 가 비정상인지 판정하고 난 이후에 해야 할 일은 정확히 어느 부분이 비정상인지 (pixel 단위에서) 찾는 일이다. (localization, segmentation)

image 단위의 계산과 유사하다. 다만 여기서는 $K$ nearest neighbor image 각각의 pixel $p \in P$에 대한 feature gallery $G = \\{F(x\_{1}, p) \mid p \in P \\}\cup\\{F(x\_{2}, p) \mid p \in P \\}.. \cup \\{F(x\_{K}, p) \mid p \in P \\}$를 만들고 여기서 target pixel의 feature $F(y,p)$에 대한 $\kappa$개의 nearest features를 추출하여 average distance를 구한다. 따라서 pixel $p$에서의 anomaly score는 다음과 같다.

\begin{equation}
	d(y, p)=\frac{1}{\kappa} \sum_{f \in N_{\kappa}(F(y, p))}\|f-F(y, p)\|^{2}
\end{equation}

모든 $K$ nearest normal image에서 threshold $\theta$에 대해 $d(y,p) > \theta$ 를 만족하는 경우, 즉 가까운 pixel을 찾지 못하는 경우에 해당 pixel을 anomalous로 판정한다.

# Implementation Details

Dataset으로 MVTec AD, STC를 256 으로 resize하여 사용하였다.

feature extractor로는 ImageNet 으로 pre-trained 된 Wide-ResNet50x2를 사용하였다.

feature는 end of the first block (56 x 56), second block (28 x 28), third block(14 x 14)를 같은 중요도로 두고 사용하였다. 

Nearest neighbour 갯수($K, \kappa$)는 MVtec 의 경우 $K=50, \kappa = 1$, STC의 경우 $K=1, \kappa = 1$ 로 설정하였다.

pixel-wise anomaly score의 경우 계산 후 Gaussian filter($\sigma = 4$)로 smoothing 하였다.

# Experimental results

## MVTec AD

<center>
    <figure>
          <img src="/assets/2020-10-25-Sub-Image Anomaly Detection with Deep Pyramid/Untitled.png" alt="Untitled" style="width:80%">
          <figcaption></figcaption>
    </figure>
</center>
<center>
    <figure>
          <img src="/assets/2020-10-25-Sub-Image Anomaly Detection with Deep Pyramid/Untitled%201.png" alt="Untitled" style="width:80%">
          <figcaption></figcaption>
    </figure>
</center>


## Shanghai Tech Campus Dataset

<center>
    <figure>
          <img src="/assets/2020-10-25-Sub-Image Anomaly Detection with Deep Pyramid/Untitled%202.png" alt="Untitled" style="width:80%">
          <figcaption></figcaption>
    </figure>
</center>
<center>
    <figure>
          <img src="/assets/2020-10-25-Sub-Image Anomaly Detection with Deep Pyramid/Untitled%203.png" alt="Untitled" style="width:80%">
          <figcaption></figcaption>
    </figure>
</center>


# Ablation Study

Table 6에서 feature pyramid의 각 feature의 resolution에 따른 결과를 확인 할 수 있다. 전체적으로 이들을 함께 사용했을 때 좋은 성능을 보임을 확인 할 수 있다.

<center>
    <figure>
          <img src="/assets/2020-10-25-Sub-Image Anomaly Detection with Deep Pyramid/Untitled%204.png" alt="Untitled" style="width:80%">
          <figcaption></figcaption>
    </figure>
</center>

Table 7에서 stage 1에서 K개의 random image를 사용하는 것과 K nearest neighbor를 사용하는 것의 차이를 보여준다. 특히 image 사이의 variation이 큰 "Grid" class의 경우 효과가 컸다고 한다.
<center>
    <figure>
          <img src="/assets/2020-10-25-Sub-Image Anomaly Detection with Deep Pyramid/Untitled%205.png" alt="Untitled" style="width:80%">
          <figcaption></figcaption>
    </figure>
</center>

---
<!--
# Discussion

## Anomaly detection via alignment

## The role of context for anomaly detection

## Optimizing runtime performance

speedup techniques (e.g. KDTrees) for large scale datasets.

image-level kNN → pixel-level kNN reduces computation time.

## Pre-trained vs. learned features

# Conclusion

Our method consists of two stages, which are designed to achieve high accuracy and reasonable computational complexity.

-->
