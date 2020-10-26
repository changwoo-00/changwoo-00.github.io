---
layout: post
title: "Paper Summary"
use_math: true
---

# Attention Guided Anomaly Localization in Images

*Shashanka Venkataramanan, Kuan-Chuan Peng, Rajat Vikram Singh, Abhijit Mahalanobis*

[*https://arxiv.org/abs/1911.08616v4*](https://arxiv.org/abs/1911.08616v4)

<!--작성일자 : 201026 -->

# Introduction

이 논문에서는 unsupervised setting에서 attention expansion loss ($L\_{ae}$)를 weakly supervised setting에서 complementary guided attention loss ($L\_{cga}$)를 도입하여 MVTAD, mSTC dataset 등에서 anomaly detection 및 localization 성능이 향상됨을 보이고 있다.

# Proposed Approach: $\text{CAVGA}$

## Unsupervised Approach: $\text{CAVGA}\_{u}$

### Convolutional latent variable

기본적으로 Variational Autoencoder 모델을 사용하였다.

일반적인 1-d latent variable 대신 input과 latent variable 사이의 spatial relation을 보존할 수 있는 convolutional latent variable을 사용하였다. [4]

### Attention expansion loss $L\_{ae}$

이 논문에서는 anomaly를 localize 하는 방식으로 attention map을 사용하였다. attention map(A)을 얻기위해 Grad-CAM[49] 방식을 사용하였다.

Defect에 대한 정보를 사전에 알 수 없는 Unsupervised learning을 가정했을 때 합리적으로 생각할 수 있는 방법은 학습시 정상 이미지 전체에 집중하도록 하는 것이다. (사람이 defect을 찾는 방식도 마찬가지이다.)

이런 아이디어를 바탕으로 학습시 정상 이미지의 모든 feature representation에 집중하도록 하기 위해 attention expansion loss를 제안하였다.

\begin{equation}
    L\_{a e, 1}=\frac{1}{|A|} \sum\_{i, j}\left(1-A\_{i, j}\right)
\end{equation}


여기서 $\|A\|$는 전체 요소의 갯수이며(feature size), $A\_{i,j}$는 $A$의 $(i, j)$ 위치의 요소이며, $A\_{i,j}\in [0,1]$ 이다. 최종적으로는 $N$ 개의 이미지에 대한 평균을 사용한다.

Attention expansion loss를 추가함으로 해서 attention map이 전체 이미지에 집중하도록 유도하였고  Fig. 1에서 효과를 확인 할 수 있다.

전체 objective function은 다음과 같다.

\begin{equation}
    L\_{final} = \omega\_r L + \omega\_{adv} L\_{adv} + \omega\_{ae} L\_{ae}
\end{equation}


여기서 $\omega\_r, \omega\_{adv}, \omega\_{ae}$ 는 각각 1, 1, 0,01로 설정했다고 한다.

input image $x\_{test}$ 와 resconstructed image $\hat{x}\_{test}$사이의 nomalized pixel-wise difference를 anomalous score $s\_a$ 로 정하고 threshold 0.5를 기준으로 anomaly를 판별하였다. 

$z$로 부터 attention map $A\_{test}$ 을 구하고 $(\mathbf{1} - A\_{test})$를 anomalous attention map으로 사용하였다. localization 또한 threshold를 0.5로 설정하여 performance를 측정하였다.


<center>
    <figure>
        <img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled.png" alt="Untitled" style="width:80%">
        <figcaption>Fig. 1</figcaption>
    </figure>
</center>

<center>
    <figure>
        <img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled%201.png" alt="Untitled" style="width:80%">
        <figcaption></figcaption>
    </figure>
</center>


## Weakly Supervised Approach: $\text{CAVGA}\_w$

몇몇의 localize label 데이터가 존재할때 classifier와 loss를 추가하여 weakly supervised $\text{CAVGA}$를 만들 수 있다.

Fig. 2. (b)에서 $\text{CAVGA}\_w$ 의 형태를 확인 할 수 있다. latent variable $z$를 1차원으로 펼친 후 fully connected layer로 정상/비정상의 binary-class classifier($C$)를 만들고 binary cross entropy loss $L\_{bce}$ 를 통해 학습 시킨다. 

input image $x$, ground truth label $y$, 가 주어졌을 때 $p\in \\{c\_a, c\_n\\}$ 을 $C$의 prediction이라 한다. 여기서 $c\_a, c\_n$ 은 anomalous, normal class를 의미한다.

$x$가 정상 이미지($y = c\_n$)인 경우 $p$ 로부터 Grad-CAM을 통해 비정상, 정상 class에 대한 attention map $A\_x^{c\_a}, A\_x^{c\_n}$ 을 구한다. $x$가 정상 이미지이기 때문에 각각 minimize, maximize 해야 한다. 

정상으로 분류된 정상이미지에 대해서만 다음과 같이 complementary guided attention loss를 정의 한다.

\begin{equation}
    L\_{c g a, 1}=\frac{\mathbb{1}\left(p=y=c\_{n}\right)}{\left|A\_{x}^{c\_{n}}\right|} \sum\_{i, j}\left(1-\left(A\_{x}^{c\_{n}}\right)\_{i, j}+\left(A\_{x}^{c\_{a}}\right)\_{i, j}\right)
\end{equation}


$L\_{cga}$는 $L\_{cga,1}$을 $N$개의 이미지에 대해 평균한 값이다. 

최종적인 objective function $L\_{final}$은 다음과 같다.

\begin{equation}
    L\_{\text {final}}=w\_{r} L+w\_{a d v} L\_{a d v}+w\_{c} L\_{b c e}+w\_{c g a} L\_{c g a}
\end{equation}

여기서 $\omega\_r, \omega\_{adv}, \omega\_c, \omega\_{cga}$는 각각 $1, 1, 0.001, 0.01$로 설정하였다.

# Experimental Setup

### Benchmark datasets

anomaly detection에 MVTAD, mSTC, LAG, MNIST, CIFAR-10, Fashion-MNIST 데이터셋을, 

anomaly localization에 MVTAD , mSTC, LAG 데이터셋을 사용하였다.

### Baseline methods

anomaly detection

LAG dataset은 CAM, GBP, Smooth-Grad, Patho-GAN와 비교하였다.

MNIST, CIFAR-10, Fashion-MNIST dataset은 LSA, OCGAN, ULSLM, CapsNet PP-based, CapsNet RE-based, AnoGAN, ADGAN, beta-VAE과 비교하였다.

anomaly localization

AVID, AE_L2, AE_SSIM, AnoGAN, CNN feature dictionary, texture inspection(TI), gamma-VAE grad, LSA, ADVAE, variation model(VM)과 비교하였다.

### Architecture details

encoder로 ImageNet dataset으로 pre-trained 된 ResNet-18을 finetuning 하여 사용하였다. [9]를 참고 및 수정하여 residual decoder로 사용하였다. 이 모델을 $\text{CAVGA-R}$이라 부른다. 

한편, based line과 Architecture에 대한 공정한 비교를 위해 Celeb-A로 pre-trained 된 DC-GAN의 discriminator와 generator를 각각 encoder와 decoder로 사용였고, 이를 $\text{CAVGA-D}$라 부른다. 

discriminator는 두 모델 모두 Celeb-A로 pre-trained된 DC-GAN을 사용하였다.

unsupervised, weakly supervised 에 따라 각각 $u$, $w$의 아래 첨자가 붙는다.

### Training and evaluation

Anomaly localization에서는 AuROC, Intersection-over-Union(IoU)을 사용하여 성능을 확인하였다.

# Experimental Results

### Performance on anomaly localization

Table 3에서 MVTAD dataset에 대해 $\text{CAVGA}$와 기타 baseline model들의 category 별 IoU, mean IoU, mean AuROC 성능을 볼 수 있다. 대체적으로 $\text{CAVGA}$ 모델이 더 나은 성능을 보임을 확인 할 수 있으며 특히 $\text{CAVGA-R}\_w$ 모델이 가장 높은 성능을 보였다.

mSTC dataset에 대한 localization 결과는 Table 4에서 확인 할 수 있다.

LAG dataset에 대한 localization 결과는 Table 5에서 확인 할 수 있다.


<center>
<figure>
<img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled%202.png" alt="Untitled" style="width:80%">
<figcaption></figcaption>
</figure>
</center>

<center>
<figure>
<img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled%203.png" alt="Untitled" style="width:80%">
<figcaption></figcaption>
</figure>
</center>


### Performance on anomaly detection

MVTAD, LAG dataset 에 대한 anomaly detection 결과를 각각 Table 6, Table 5에서 확인 할 수 있다.

<center>
<figure>
<img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled%204.png" alt="Untitled" style="width:80%">
<figcaption></figcaption>
</figure>
</center>


<center>
<figure>
<img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled%205.png" alt="Untitled" style="width:80%">
<figcaption></figcaption>
</figure>
</center>


# Ablation Study

Table 8에서 latent variable $z$, expansion loss $L\_{ae}$, $L\_{cga}$에 대한 ablation test 결과를 볼 수 있다. Method 이름 뒤에 *가 붙은 경우는 1-d(flattened) latent variable $z$ 를 사용했음을 의미한다.

<center>
<figure>
<img src="/assets/2020-10-26-Attention Guided Anomaly Localization in Images/Untitled%206.png" alt="Untitled" style="width:80%">
<figcaption></figcaption>
</figure>
</center>


## References

[4] Baur, C., Wiestler, B., Albarqouni, S., Navab, N.: Deep autoencoding models for unsupervised anomaly segmentation in brain mr images. In: International MICCAI Brainlesion Workshop. pp. 161–169. Springer (2018)

[9] Brock, A., Donahue, J., Simonyan, K.: Large scale GAN training for high fidelity natural image synthesis. In: International Conference on Learning Representations (2019)

[49] Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D.: Grad-cam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 618–626 (2017)

---
<!--
Unlike traditional autoencoders[6, 18] where the latent variable is flattened, inspired from [4], we use a convolutional latent variable to preserve the spatial relation between the input and the latent variable
-->
