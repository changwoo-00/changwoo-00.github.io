---
layout: post
title: "Paper Summary"
use_math: true
---
# PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization | arXiv' Nov.2020

[https://arxiv.org/pdf/2011.08785.pdf](https://arxiv.org/pdf/2011.08785.pdf)

(정리 : 21.01.11)

### Introduction

Pretrained CNN을 사용하여 embedding 추출

각 patch position은 multivariate gaussian distribution으로 표현됨

PaDiM takes into account the correlations between different semantic levels of a pretrained CNN.

MVTec AD, STC dataset에서 sota 달성.

추가적으로 실제상황과 유사한 non-aligned dataset에서 평가

### Embedding extraction

PaDiM은 SPADE [5]와 유사하다.

image를 $(i, j) \in[1, W] \times[1, H]$ 의 grid로 분할하여 각 grid에 해당하는 feature embedding 추출.

embedding의 크기를 줄이기 위해 PCA를 사용. 성능을 유지한 상태로 크기를 줄일 수 있음을 확인.

$(i, j)$ 위치의 N개의 학습 데이터셋에 대한 feature embedding $X_{i j}=\lbrace x_{i j}^{k}, k \in [ 1, N ]\rbrace$ 이 가우시안 분포  $\mathcal{N}(\mu_{i j}, \mathbf{\Sigma}\_{i j})$ 로 부터 생성되었다고 가정한다. ($\mu_{i j}$, $\Sigma_{i j}$ 은 각각 sample mean, sample covariance)

$$\Sigma_{i j}=\frac{1}{N-1} \sum_{k=1}^{N}\left(\mathbf{x}_{\mathbf{i j}}^{\mathbf{k}}-\mu_{\mathbf{i j}}\right)\left(\mathbf{x}_{\mathbf{i j}}^{\mathbf{k}}-\mu_{\mathbf{i j}}\right)^{\mathrm{T}}+\boldsymbol{\epsilon} I$$

**Inference : computation of anomaly map**

Mahalanobis distance 사용

$$M\left(x_{i j}\right)=\sqrt{\left(x_{i j}-\mu_{i j}\right)^{T} \Sigma_{i j}^{-1}\left(x_{i j}-\mu_{i j}\right)}$$

Mahalanobis distance matrix $M=\left(M\left(x_{i j}\right)\right)_{1<i<W, 1<j<H}$ 이 anomaly map이 됨.

위치의 값이 클 수록 anomalous 한 편.

이미지의 최종 anomaly score은 anomaly map $M$의 maximum 값.

K-NN과 같은 scalability issue가 존재하지 않음.


<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 1</figcaption>
    </figure>
</center>

<!-- ![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled.png) -->

### Metrics

**AUROC** 사용

**per-region-overlap score(PRO-score)**(AUROC가 anomaly의 크기에 따른 편향이 있기 때문)

### Datasets

MVTec AD, Shanghai Tech Campus (STC) Dataset

자체적으로 random rotation과 random crop을 적용한 Rd-MVTec AD을 별도로 테스트.

### Backbone networks

ResNet18, Wide ResNet-50-2, EfficientNet-B5 사용. (ImageNet pretrained)

### Results

<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%201.png" alt="Untitled" style="width:80%">
    </figure>
</center>
<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%202.png" alt="Untitled" style="width:80%">
    </figure>
</center>
<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%203.png" alt="Untitled" style="width:80%">
    </figure>
</center>
<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%204.png" alt="Untitled" style="width:80%">
    </figure>
</center>
<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%205.png" alt="Untitled" style="width:80%">
    </figure>
</center>
<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%206.png" alt="Untitled" style="width:80%">
    </figure>
</center>
<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%207.png" alt="Untitled" style="width:80%">
    </figure>
</center>
<center>
    <figure>
        <img src="\assets\2021-01-11-PaDiM a Patch Distribution Modeling Framework for Anomaly Detection and Localization/Untitled%208.png" alt="Untitled" style="width:80%">
    </figure>
</center>

<!-- ![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%201.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%201.png)

![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%202.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%202.png)

![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%203.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%203.png)

![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%204.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%204.png)

![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%205.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%205.png)

![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%206.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%206.png)

![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%207.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%207.png)

![PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%208.png](PaDiM%20a%20Patch%20Distribution%20Modeling%20Framework%20for%20%209917507cc7b24effb80025928f14322d/Untitled%208.png) -->