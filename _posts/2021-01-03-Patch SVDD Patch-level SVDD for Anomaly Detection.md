---
layout: post
title: "Paper Summary"
use_math: true
---
# Patch SVDD: Patch-level SVDD for Anomaly Detection and Segmentation | ACCV' 20

*Jihun Yi and Sungroh Yoon*

[*https://arxiv.org/abs/2006.16067v2*](https://arxiv.org/abs/2006.16067v2)

# Introduction

One-class support vector machine (OC-SVM) 과 함께 support vector data description (SVDD)는 one-class classification에 사용되는 전통적인 알고리즘이다.

Ruff et al. 은 SVDD에 deep learning 을 접목시킨 Deep SVDD를 제안하였다.

본 논문에서는 Patch-wise detection method를 이용해 Deep SVDD를 발전시킨 Patch SVDD 를 제안한다.

# Patch-wise Deep SVDD

DeepSVDD의 컨셉을 patchwise로 적용하고, self-supervised learning의 효과를 살린 논문이다.

Anomaly size가 다양하므로 추가적으로 Hierarchical encoding 방법을 적용하였다.

DeepSVDD → patchwise loss

\begin{equation}
    \mathcal{L}\_{\mathrm{SVDD\text{'}}}=\sum\_{i . i^{\prime}}\left\|f\_{\theta}\left(\mathbf{p}\_{i}\right)-f\_{\theta}\left(\mathbf{p}\_{i^{\prime}}\right)\right\|\_{2}
\end{equation}


self-supervised learning loss

\begin{equation}
    \mathcal{L}\_{\mathrm{SSL}}=\text { Cross-entropy }\left(y, C\_{\phi}\left(f\_{\theta}\left(\mathbf{p}\_{1}\right), f\_{\theta}\left(\mathbf{p}\_{2}\right)\right)\right)
\end{equation}


Total loss

\begin{equation}
    \mathcal{L}\_{\mathrm{Patch} \mathrm{SVDD}}=\lambda \mathcal{L}\_{\mathrm{SVDD}\text{'}}+\mathcal{L}\_{\mathrm{SSL}}
\end{equation}


Hierarchical encoding

<center>
    <figure>
        <img src="/assets/2021-01-03-Patch SVDD Patch-level SVDD for Anomaly Detection/Untitled.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 1</figcaption>
    </figure>
</center>


Inference (Anomaly map 생성)

<center>
    <figure>
        <img src="/assets/2021-01-03-Patch SVDD Patch-level SVDD for Anomaly Detection/Untitled%201.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 2</figcaption>
    </figure>
</center>

## References
"Unsupervised Visual Representation Learning by Context Prediction"
[https://arxiv.org/pdf/1505.05192.pdf](https://arxiv.org/pdf/1505.05192.pdf)

---
<!--
self-supervised learning ref.

"Unsupervised Visual Representation Learning by Context Prediction"

[https://arxiv.org/pdf/1505.05192.pdf](https://arxiv.org/pdf/1505.05192.pdf)
-->
