---
layout: post
title: "Paper Review"
use_math: true
---

# Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows

Marco Rudolph, Bastian Wandt, Bodo Rosenhahn

(정리 20.10.2)

### 목적

기존의 anomaly detection 방법들은 subtle difference를 잡아내지 못했다.  이 논문에서는 cnn에서 추출한 image feature로 부터 정확한 density 함수를 추출해내 이 문제를 해결하고자 한다.

### Introduction

이를 위해 저자는 anomaly detection 논문에서 주로 사용되어온 variational autoencoder나 GAN이 아니라 feature space와 latent space 사이의 bijective map이 가능한 normalizing flow 방식을 사용한다.

(Differnet)

...

high&low likelihood → score

likelihood of several transformations

feature extractor와 제안한 image transformation 덕분에 적은 수의 학습 데이터로도 sota의 성능을 뛰어 넘는다.

defect detection 뿐만 아니라 scoring function으로 부터의 gradient를 계산하여 defect의 localization도 가능하다.

contributions

- multi-scale image feature에 적용한 normalizing flow로 anomaly를 detect 함.
- training label 없이 anomaly localization을 함.
- 적은 양의 학습 데이터셋을 가지고 있는 경우에 적용가능.
- MVTec AD와 Magnetic Tile Defects dataset에 대해 State-of-the-art 성능을 달성함.

Normalizing Flows

Normalizing Flow(NF)는 data distribution 과 well-defined density 사이의 변환을 학습하는 neural network이다.

이 모델의 특징은 양방향(bijective) mapping이 가능하다는 것이다.

...자세한 내용은 다른 페이지에서..

Normalizing flow의 특성을 anomaly detection에 사용하는 것에 대해서 아직 많은 관심을 끌지 못했다. 몇몇 Real-NVP와 MADE를 사용한 연구가[24,27,7] 진행되었으나 visual data로 진행된 사례는 아직 없는 듯 하다.

Germain et al. (MADE)

Kingma et al. (Inverse autoregressive flows)

Dinh et al. (Real-NVP)

Ardizzone et al. (Invertible Neural Networks)

### Baseline method

![Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled.png](Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled.png)

overview of pipeline

anomaly free training images : $x \in X$

image features : $y \in Y$

latent vectors : $z$ (with a well-defined distribution $p_Z(z)$ (normal gaussian))

**feature extractor**

$f_{ex} : X → Y$

Imagenet dataset으로 pretrained 된 AlexNet을 사용함.

3가지 input image size의 feature 사용 (448x448, 224x224, 112x112)

사용된 총 feature의 개수는 3*256=768.

**normalizing flow**

$f_{NF} : Y → Z$

Real-NVP의 coupling block 8개로 구성

internal function s, t는 fully-connected network 사용

![Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%201.png](Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%201.png)

Architecture of one block inside the normalizing flow

### Dataset

MVTec AD, Magnetic Tile Defects(MTD) dataset을 사용하였다.

### Training

문제의 목적은 extracted features $y\in Y$ 가 주어졌을 때 likelihood를 최대화 하는 parameter를 찾는 것다. change-of-variables formula로 부터 objective loss를 다음과 같이 정의 할 수 있다.

$$p_Y(y) = p_Z(z)\left|\det{\frac{\partial z}{\partial y}}\right|. \\ \ \\ \log p_Y(y) = \log p_Z(z) + \log\left|\det{\frac{\partial z}{\partial y}}\right| \\ \ \\ \mathcal{L}(y) = \frac{||z||^2_2}{2} - \log \left| \det{\frac{\partial z}{\partial y}}\right|.$$

두번째 식에서 $p_Z(z)$를 standard normal distribution으로 설정하여 최종적인 식을 도출 하였다.

### Scoring Function

scoring function으로 negative log-likelihood를 사용하였다.

robust한 결과를 얻기위해 input image 에 여러 transformation 함수를 적용하여 평균을 냈다.

$$\tau (x) = \mathbb{E}_{T_i \in \mathcal{T}} [-\log p_Z(f_{NF}(f_{ex}(T_i(x))))]$$

where $T_i (x) \in \mathcal{T}$.

anomaly score $\tau(x)$ 와 threshold $\theta$ 를 이용하여 최종 판정

$$ \mathcal{A}(x)= 
\begin{cases}
    1,& \text{for } \tau(x)\geq \theta\\
    0,              & \text{for } \tau(x)< \theta
\end{cases}$$

where $\mathcal{A}(x) = 1$ indicates an anomaly.

### Localization

Localization에 대해 최적화 하지는 않았지만 gradient를 계산하여 localization map을 얻을 수는 있다.

### Results

Detection

one-class SVM, 1-NN, GeoTrans, GANomaly, 그리고 DSEBM에 대해 AUROC를 통해 성능을 비교하였다.

![Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%202.png](Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%202.png)

![Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%203.png](Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%203.png)

![Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%204.png](Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%204.png)

Training Sample 갯수에 따른 성능 변화

Localization

![Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%205.png](Same%20Same%20But%20DifferNet%20Semi-Supervised%20Defect%20Det%209fe9f60c335346cebfbe024468063c3d/Untitled%205.png)

MVTec AD 에 대한 Localization 결과

### Discussion

normalizing flow-based density estimation 을 통한 defect detection model(DifferNet)을 제안하였다.

비교적 적은 양의 training sample을 가지고도 좋은 성능을 보이는 것을 확인 할 수 있다.

---

### note

중요

**Note that we focus on works that deal with image anomaly detection rather than anomaly localization  to keep the focus on our main problem**

Generative models

Anomaly detection approaches using these models are based on the idea that the anomalies cannot be generated since they do not exist in the training set.

Autoencoder-based approaches

Bergmann et al. (SSIM-loss model)

In many cases autoencoder-based methods fail because they generalize too strongly, i.e. anomalies can be reconstructed as good as normal samples.

Gong et al. tackle the generalization problem by employing memory modules.

Zhai et al. connect regularized autoencoders with energy-based models to model the data distribution and classify samples with hight energy as an anomaly.

GAN-based approaches

assume that only positive samples can be generated.

Schlegl et al. propose a two stage training method

Akcay et al. make use of adversarial training by letting an autoencoder directly act as generating part of the GAN. This enforces the property of the decoder to only generate normal-like samples which can be measured by the difference between the embedding of the original and the reconstructed data.

We argue that generative models are appropriate for a wide range of defect detection scenarios since they strongly depend on the anomaly type. ...

### 개인적 코멘트

### 코드분석

[https://github.com/marco-rudolph/differnet/blob/master/train.py](https://github.com/marco-rudolph/differnet/blob/master/train.py)

- [train.py](http://train.py) -evaluate

    ```python
    def train(train_loader, test_loader):
    		# ...
    		for epoch in range(c.meta_epochs):
    				# train some epochs
    				# ...
    				
    				# evaluate
    				model.eval()
    				if c.verbose:
    				    print('\nCompute loss and scores on test set:')
    				test_loss = list()
    				test_z = list()
    				test_labels = list()
    				with torch.no_grad():
    				    for i, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
    				        inputs, labels = preprocess_batch(data)
    				        z = model(inputs)
    				        loss = get_loss(z, model.nf.jacobian(run_forward=False)) # !
    				        test_z.append(z)
    				        test_loss.append(t2np(loss))
    				        test_labels.append(t2np(labels))
    				
    				test_loss = np.mean(np.array(test_loss))
    				if c.verbose:
    				    print('Epoch: {:d} \t test_loss: {:.4f}'.format(epoch, test_loss))
    				
    				test_labels = np.concatenate(test_labels)
    				is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
    				
    				z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    				anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1))) # !
    				score_obs.update(roc_auc_score(is_anomaly, anomaly_score), epoch,
    				                 print_score=c.verbose or epoch == c.meta_epochs - 1)
    ```

    ```python
    def get_loss(z, jac):
        '''check equation 4 of the paper why this makes sense - oh and just ignore the scaling here'''
        return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]

    def t2np(tensor):
        '''pytorch tensor -> numpy array'''
        return tensor.cpu().data.numpy() if tensor is not None else None
    ```
