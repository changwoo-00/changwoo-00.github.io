---
layout: post
title: "Paper Summary"
use_math: true
---

# Adversarial Discriminative Attention for Robust Anomaly Detection - _WACV' 20_

_Daiki Kimura, Subhajit Chaudhury, Minori Narita, Asim Munawar, Ryuki Tachibana_

<!--(정리 20.10.1~2)-->

<br/>

## 목적

anomaly detection의 기존 방법들은 global level pixel comparison에 의존하고 있는데 이는 real world application에서는 robust한 방법이 아니다.

이 논문에서는 robust anomaly detection을 위한 self-supervised masking method를 제안한다.

더불어, 몇개의 anomaly sample을 구할수 있는 상황에서 semi-supervised learning을 통해 정확도를 향상 시키는 방법을 제시한다.

masking method

- discriminator's class activation map
- mask suppresses spurious signals from the background
<br/>

## Baseline method

AnoVAEGAN
<center>
    <figure>
          <img src="/assets/2020-10-18-Adversarial Discriminative Attention for Robust Anomaly Detection/Untitled.png" alt="Untitled" style="width:60%">
          <figcaption></figcaption>
    </figure>
</center>


VAE를 기본으로 사용하되 VAE의 blur한 이미지 문제를 해결하기 위해 adversarial training을 진행함. (Jensen-Shannon divergence 사용)

### Objective function

<br/>

\begin{equation}  
	\nonumber 
    \begin{aligned} 
        \nonumber 
        \mathcal{L}\_{GAN} =  &\mathbb{E}\_{x\sim p\_{data}}\log{[\mathcal{D} (x)]} \\\  &+ \mathbb{E}\_{z\_{\mu},z\_{\sigma}\sim p\_{data}}\log{[1 - \mathcal{D} (\mathcal{G} \_{dec}(z\_{\mu},z\_{\sigma}))]}\\\ &+ \mathbb{E}\_{x\sim p\_{data}}\log{[1 - \mathcal{D} (\mathcal{G}\_{dec}(\mathcal{G}\_{enc}(x)))]} 
    \end{aligned} 
\end{equation}
<br/>
\begin{equation}
	\nonumber
		\min_{\mathcal{G}\_{enc},\mathcal{G}\_{dec}} \max\_{\mathcal{D}}
		(\mathcal{L}\_{GAN} + \mathcal{L}\_{prior} + \mathcal{L}\_{image})
\end{equation}

<br/>
In Semi-suprevised setting, adversarial loss is modified as follows,

\begin{equation}
	\nonumber
	\mathcal{L}\_{GAN}^{ano} = \mathcal{L}\_{GAN} + \mathbb{E}\_{x \sim p\_{ano}} (\log[1 - \mathcal{D}(x)])
\end{equation}

where the $p\_{ano}$ is the known anomaly distribution.

<br/>

<center>
    <figure>
          <img src="/assets/2020-10-18-Adversarial Discriminative Attention for Robust Anomaly Detection/Untitled%204.png" alt="Untitled" style="width:100%">
          <figcaption></figcaption>
    </figure>
</center>


비교모델

VAE(AnoVAE), GAN(AnoGAN), VAEGAN(AnoVAEGAN).

모든 network의 structure는 VAEGAN(Larsen et al.)을 참고했다고 한다.

<br/>

## Anomaly detection with attention

학습이 진행될 수록 discriminator가 중요한 영역을 집중해서 보기 시작한다.

threshold와 anomaly score를 사용해서 "normal", "anomaly"를 대한 판정한다.

하지만 논문에서는 평가를 ROC curve로 하기 때문에 threshold 대한 discussion은 따로 하지 않는다.

### anomaly score

**self-supervised setting**

class activation map과 error(difference)의 곱

더 좋은 성능을 위해 input $x$와 생성된 $\hat{x}$ 로 부터의 attention map을 모두 사용한다.

attention value에 대한 normalization 작업을 거친 후, 마지막으로 probability of the normal distribution from the discriminator를 빼준다.
<br/>
<br/>
\begin{equation}
	\nonumber
	\begin{aligned}
	c &= CAM\_{\mathcal{D}}(x) + CAM\_{\mathcal{D}}(\hat{x}), \\\ score &= \frac{||c*(x - \hat{x})||^2\_2}{||c||^2\_2} - \beta \mathcal{D}(x)
	\end{aligned}
\end{equation}
<br/>

**semi-supervised setting**

ensemble of discriminator를 사용하는게 하나를 사용하는 것 보다 더 좋은 결과가 나왔다고 한다.

<br/>

\begin{equation}
	\nonumber
	c = CAM\_{\mathcal{D}\_{\mathcal{G}}}(x)   + CAM\_{\mathcal{D}\_{\mathcal{G}}}(\hat{x}) + CAM\_{\mathcal{D}\_{ano}}(x) + CAM\_{\mathcal{D}\_{ano}}(\hat{x}), \\\ score = \frac{||c*(x - \hat{x})||^2\_2}{||c||^2\_2} - \beta (\mathcal{D}\_{\mathcal{G}}(x) + \mathcal{D}\_{ano}(x))
\end{equation}

<br/>
## Datasets

MNIST with noise, Hand gesture, Pigeon gesture, Venus images
<br/>
## Quantitative evaluation

anomaly score에 대한 AUROC curve

$\beta$ 값 : MNIST와 pigeon의 경우  0.2, hand gesture의 경우 5, Venus의 경우 0.05를 적용했다.
<br/>
## Results

**self-supervised learning**

<center>
    <figure>
          <img src="/assets/2020-10-18-Adversarial Discriminative Attention for Robust Anomaly Detection/Untitled%207.png" alt="Untitled" style="width:100%">
          <figcaption></figcaption>
    </figure>
</center>

<center>
    <figure>
          <img src="/assets/2020-10-18-Adversarial Discriminative Attention for Robust Anomaly Detection/Untitled%208.png" alt="Untitled" style="width:100%">
          <figcaption></figcaption>
    </figure>
</center>


Table 1.&Table 2. 각 digit에 해당하는 column은 해당 digit만 normal이고 나머지 digit은 anomaly로 봤을 때 실험한 결과. 각각에 column에 대해 random seed 로 5번 실험 후 평균을 냈다. 다른 model 과 비교하여  더 좋은 성능을 보이는 것을 볼 수 있다.

**semi-supervised learning**
 <center>
    <figure>
          <img src="/assets/2020-10-18-Adversarial Discriminative Attention for Robust Anomaly Detection/Untitled%209.png" alt="Untitled" style="width:100%">
          <figcaption></figcaption>
    </figure>
</center>

 <center>
    <figure>
          <img src="/assets/2020-10-18-Adversarial Discriminative Attention for Robust Anomaly Detection/Untitled%2010.png" alt="Untitled" style="width:100%">
          <figcaption></figcaption>
    </figure>
</center>


Table 3. & Table 4. 학습시 하나의 anomaly mode를 알게 했을 때의 결과. 각 column에 해당하는 evaluation결과는 하나의 mode에 대한 결과인듯(?). Table 1., Table 2. 와 비교해 봤을 때 큰 폭의 성능향상을 볼 수 있다.
<br/>
## Discussion

이 논문의 핵심 아이디어는 anomaly를 검출하기 위해 adversarial learning setting에서 discriminator의 CAM를 사용한다는 것이다. CAM을 통해 중요한 feature에 더 집중하므로써 더욱 강건한 모델을 만들 수 있다.

실험을 통해 이러한 방법이 self-supervised와 semi-supervised 모든 경우에 더 좋은 성능을 내는 것을 보였다. 

이 방식의 한계점은 anomaly가 배경에 위치하는 경우이다. 이 경우 score를 계산할때 mask가 anomaly를 제외시켜 버려서 원하는 성능이 나오지 않을 것이다. 하지만 배경에서 anomaly를 찾아야 하는 사례는 드물것이다.

---

<!--

### note

We argue that, for discerning normal and anomaly class distribution, the discriminators attention map would align with the location of the primary object in the image and hence would mask unwanted noisy perturbations outside the main discriminative area.

In this paper, we compare our method with the
state-of-the-art methods on four different kinds of anomaly
detection problems which have a wide variety of samples.

four different datasets

small training samples

- **Related work**

    Schlegl et al. (AnoGAN)

    Zenati et al. 

    Akcay et al.

    Sabokrou et al.

    Baur et al. (AnoVAEGAN)

    **(medical)**

    Hauskrecht et al.

    Prastawa et al.

    **(other fields)**

    anomaly detection in crowded scenes

    defect detection

    Sultani et al. (in videos)

    Brown et al. and Wu et al. (RNN based)

    **(attention from class activation maps)**

    Selvaraju et al. (Grad-CAM)

The calculation time for producing an attention map is around 10 milliseconds per image on Nvidia Titan X

The typical problem setting involves samples from the normal class distribution, and the goal is to detect whether or not and input test sample belongs to the normal class.

This setting is also referred to as one-class classification or out-distribution detection

### 개인적 코멘트

어떤 지난 연구결과의 특정 모델과 비교하는 것이 아니라 단순한 모델 형태을(AE, VAE, GAN) 비교대상으로 삼고 network의 디테일은 저자가 기준에 맞게 설계하는 방법으로 결과를 도출하고 논문을 쓸 수 있다는 것을 확인할 수 있었다.

semi-supervised 의 경우 결국 classification과 거의 비슷한 상황이 되어버린 상황인데 이런 상황에서의 성능향상이 얼마나 의미를 지니는지 모르겠다.

Localization에 대한 내용이 없어 아쉽다.

anomaly score와 threshold, 그리고 AUROC의 관계에 대해 생각해 보기 좋은 논문이다. (localization이 아닌경우에는 처음 제대로 이해한 것 같다.)

CAM을 계산하는 대략적인 시간이 궁금했는데 이를 알려주는 논문을 처음 찾은 것 같아서 도움이 많이 될 듯 하다.

-->