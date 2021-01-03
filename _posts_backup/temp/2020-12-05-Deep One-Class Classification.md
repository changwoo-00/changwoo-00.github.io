---
layout: post
title: "Paper Summary"
use_math: true
---

# Deep One-Class Classification

*Lukas Ruff, Robert A. Vandermeulen, Nico Gornitz, Lucas Deecke, Shoaib A. Siddiqui, Alexander Binder, Emmanuel Muller, Marius Kloft*

[*http://data.bit.uni-bonn.de/publications/ICML2018.pdf*](https://arxiv.org/abs/1911.08616v4)

<!--작성일자 : 201205 -->

# Introduction



Deep Support Vector Data Description (Deep SVDD)

hypershere의 volume을 줄이도록 neural network를 학습시키므로서 공통된 요소가 추출되도록 유도한다.



# Related Work

## Kernel-based One-Class Classification

One-Class SVM, Support Vector Data Description (SVDD)

Kernel-based method 의 단점.

느리다. 최소 data갯수 n에 대해 O(n^2).

메모리를 많이 차지한다.

Deep SVDD에서는 이러면에서 더 뛰어나다.



## Deep Approaches to Anomaly Detection





# Deep SVDD





## T2

### T3



\begin{equation}
    L\_{final} = \omega\_r L + \omega\_{adv} L\_{adv} + \omega\_{ae} L\_{ae}
\end{equation}






<center>
<figure>
<img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled%202.png" alt="Untitled" style="width:80%">
<figcaption></figcaption>
</figure>
</center>



## References

[4] Baur, C., Wiestler, B., Albarqouni, S., Navab, N.: Deep autoencoding models for unsupervised anomaly segmentation in brain mr images. In: International MICCAI Brainlesion Workshop. pp. 161–169. Springer (2018)


---
<!--
Our method, Deep Support Vector Data Description (Deep SVDD), trains a neural network while minimizing the volume of a hypersphere that encloses the network representations of the data (see Figure 1)



However, with Deep SVDD we learn useful feature representations of the data together with the one-class classification objective.



Given some training data $\mathcal{D}_{n}=\left\{\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{n}\right\}$ , we define the soft-boundary Deep SVDD objective as









-->