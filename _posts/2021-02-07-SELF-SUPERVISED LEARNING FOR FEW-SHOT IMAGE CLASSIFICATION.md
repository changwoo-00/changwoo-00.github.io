---
layout: post
title: "Paper Summary"
use_math: true
---
# SELF-SUPERVISED LEARNING FOR FEW-SHOT IMAGE CLASSIFICATION

*Da Chen et. al. 2020*

[https://arxiv.org/pdf/1911.06045.pdf](https://arxiv.org/pdf/1911.06045.pdf)

(정리 : 21.02.07)


<center>
    <figure>
        <img src="/assets/2021-02-07-SELF-SUPERVISED LEARNING FOR FEW-SHOT IMAGE CLASSIFICATION/Untitled.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 1</figcaption>
    </figure>
</center>


## 3. Method

few-shot learning classification 문제의 한가지 인기있는 해결책은 pre-trained embedding network위에 meta learning을 적용하는 것이다. 최근의  대부분의 연구들은 두번째 과정, 즉, meta learning stage에 집중하였다. 우리는 이 two stage 패러다임을 따르지만 강력한 embedding network를 학습시키기 위해 self-supervised learning을 활용한다.

### 3.1. Self-supervised learning stage

우리는 Augmented Multiscale Deep InfoMax(AMDIM)을 self-supervised model로써 사용하였다.

pretext task는 context로부터 뽑힌 여러 feature들 간의 mutual information을 최대화 하도록 설계되었다.

joint와 marginal의 곱 사이의 KullbackLeibler (KL) divergence로 정의된 mutual information(MI)은 random variable X와 Y 사이의 shared information을 측정한다.
\begin{equation}
	I(X, Y) =D\_{K L}(p(x, y) \| p(x) p(y)) =\sum \sum p(x, y) \log \frac{p(x \mid y)}{p(x)}
\end{equation}

우리는 단지 sample을 가지고 있고 직접적으로 underlying distribution에 접근할 수 없으므로 MI를 추정하는 일은 어려운 일이다.

[22]는 negative sampling에 대한 Noise Contrastive Estimation (NCE) loss를 최소화 함으로써 MI의 lower bound를 최대화 할 수 있음을 증명하였다.

AMDIM의 핵심 concept는 두 view $(x\_a, x\_b)$ 사이의 global feature와 local feature사이의 MI를 최대화 하는 것이다.

정확히 $<f\_g(x\_a), f\_5(x\_b)>$, $<f\_g(x\_a), f\_7(x\_b)>$ 그리고 $<f\_5(x\_a), f\_5(x\_b)>$의 MI를 최대화 시킨다.

여기서 $f\_g$는 global feature, $f\_5$는 encoder의 $5\times5$ local feature map, $f\_7$은 encoder의 $7\times7$  feature map이다.

예를 들어, $f\_g(x\_a)$와 $f\_5(x\_b)$사이의 NCE loss는 다음과 같다.
	
\begin{equation}
	\mathcal{L}\_{amdim}(f\_{g}(x\_{a}), f\_{5}(x\_{b})) = -\log \frac{\exp \phi(f\_{g}(x\_{a}), f\_{5}(x\_{b})))}{\sum_{\widetilde{x\_{b}} \in \mathcal{N}\_{x} \cup x\_{b}} \exp (\phi(f\_{g}(x\_{a}), f\_{5}(\widetilde{x\_{b}})))}
\end{equation}


$\mathcal{N}\_x$는 $x$의 negative sample이며 $\phi$는 distance metric function이다.

$x\_a$와 $x\_b$ 사이의 overall loss는 다음과 같다.

\begin{equation}
	\mathcal{L}\_{amdim}\left(x\_{a}, x\_{b}\right)=\mathcal{L}\_{ amdim}\left(f\_{g}\left(x\_{a}\right), f\_{5}\left(x\_{b}\right)\right)+\mathcal{L}\_{amdim }\left(f\_{g}\left(x\_{a}\right), f\_{7}\left(x\_{b}\right)\right)+\mathcal{L}\_{amdim }\left(f\_{5}\left(x\_{a}\right), f\_{5}\left(x\_{b}\right)\right)
\end{equation}


### 3.2. Meta-learning stage

K-way C-shot  
D : entire training dataset  
V : class labels  
S : support set, $S = \left{(x\_i, y\_i)|i=1,...,m\right}$, where $m = C\times K$  
Q : query set, $Q = \{(x\_j,y\_j)|j = 1,...,n\}$  

Snell et. al. 과 같은 최근의 인기있는 framework는 모든 input sample을 mean vector c에 mapping 시킬 수 있는 embedding function을 학습 시킬 수 있다.

class k에 대해서 training sample들의 embedding feature의 centroid는 다음과 같이 얻을 수 있다.

\begin{equation}
	c\_{k}=\frac{1}{|S|} \sum\_{\left(x\_{i}, y\_{i}\right) \in S} f\_{g}\left(x\_{i}\right)
\end{equation}

query sample q에 대해 모든 class에 대한 distribution을 다음과 같이 얻는다.

\begin{equation}
	p(y=k \mid q)=\frac{\exp \left(-d\left(f\_{g}(q), c\_{k}\right)\right)}{\sum\_{k^{\prime}} \exp \left(-d\left(f\_{g}(q), c\_{k^{\prime}}\right)\right)}
\end{equation}


우리는 distance function d로 Euclidean distance를 채택하였다.

식에서 확인 할 수 있듯이 distribution은 sample의 embedding과 class의 reconstructed features 사이의 distance에 대한 softmax로 표현된다. meta learning stage에서 loss는 다음과 같다.

\begin{equation}
	\mathcal{L}\_{meta}=d\left(f\_{g}(q), c\_{k}\right)+\log \sum\_{k^{\prime}} d\left(f\_{g}(q), c\_{k^{\prime}}\right)
\end{equation}

## Experimental results

### 4.1. Datasets

MiniImageNet dataset. 100 class중 64 class를 학습, 16 class를 validation, 20 class를 test에 사용하였다.

Caltech-UCSD Birds-200-2011(CUB-200-2011) dataset. a dataset for fine-grained classification. 200 class중 100 class를 학습에, 50 class를 validation, 50 class를 test에 사용하였다.

### 4.2. Training Details

### 4.3. Quantitative comparison

<center>
    <figure>
        <img src="/assets/2021-02-07-SELF-SUPERVISED LEARNING FOR FEW-SHOT IMAGE CLASSIFICATION/Untitled%201.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 1</figcaption>
    </figure>
</center>

<center>
    <figure>
        <img src="/assets/2021-02-07-SELF-SUPERVISED LEARNING FOR FEW-SHOT IMAGE CLASSIFICATION/Untitled%202.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 1</figcaption>
    </figure>
</center>


### 4.4. Ablation Study

하나의 우려는 제안된 network가 주는 성능 향상이 단순히 network의 capacity 증가 때문이 아닌지에 대한 것이다. 제안한 방법의 효과를 검증하기 위해 embedding network를 labeled data (Mini80-SL, CUB150_SL)로 학습시켰다. Table 2와 3에서 볼 수 있듯이 제한적인 data로 큰 network를 학습시키는 것은 overfit문제를 발생시키고 test 단계에서 새로운 unseen class에 대해 adjust 할 수 없으므로 단순한 4 Conv block network보다 나쁜 성능을 보였다. 그러나  SSL에 기반한 pre-training은 성능을 훨씬 개선시켰다. 또한 meta learning에 기반한 fine-tuning을 추가함으로써 성능을 대폭 향상 시킴을 보였다.

[28]에 의하면 현재의 few-shot learning method들은 domain간에 효율적으로 transfer 하는 것이 힘들다고 하였다.  그러나 이 논문에서는 ImageNet으로 학습하여 CUB dataset을 테스트하는 transferability test도 진행하였다. Table3에서 볼 수 있듯이 효과적으로 transfer하여 각 task에서 15%, 8%의 성능이 향상됨을 보였다.

---
<!--
EGNN (kakaobrain blog) → few shot learning (papers with code) → this paper
-->