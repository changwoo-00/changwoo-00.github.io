# Sub-Image Anomaly Detection with Deep Pyramid Correspondences | arXiv 20

<!— 작성일자 : 201025 —>

*Niv Cohen, Yedid Hoshen*

[*https://arxiv.org/abs/2005.02357*](https://arxiv.org/abs/2005.02357)

# Introduction

pretrained 된 모델로부터 M개의 feature를 추출하여 train set에서 k nearest neighbor 와의 비교를 통해 anomaly를 검출하는 방식을 제안하였다.

pre-trained 모델을 직접 사용하기 때문에 따로 학습이 필요가 없으며 성능 또한 매우 우수함을 보였다.

# Correspondence-based Sub-Image Anomaly Detection

## Feature Extraction

Feature Extractor로 pre-trained ResNet(Wide-ResNet50x2)를 사용하였다.

$$f_{i}=F\left(x_{i}\right)$$

$F$ : global feature extractor, $x_i$ : image, $f_i$ : extracted feature 를 의미한다.

우선 모든 training image 에 대해 feature를 추출하여 저장해둔다.

inference 시에는 test image의 feature 만 추출한다.

## K Nearest Neighbor Normal Image Retrieval

첫번째 과정으로 test image가 anomaly를 포함하고 있는지를 판별한다.

test image $y$에 대해 training set 에서 $K$ nearest  image, $N_k(f_y)$ 를 추출한다.

distance는 image(feature)-level Euclidean metric을 사용하였다.

$$d(y)=\frac{1}{K} \sum_{f \in N_{K}\left(f_{y}\right)}\left\|f-f_{y}\right\|^{2}$$

여기서 image가 정상인지 비정상인지 threshold $\tau$를 기준으로 판정한다.

## Sub-image Anomaly Detection via Image Alignment

image 가 비정상인지 판정하고 난 이후에 해야 할 일은 정확히 어느 부분이 비정상인지 (pixel 단위에서) 찾는 일이다. (localization, segmentation)

image 단위의 계산과 유사하다. 다만 여기서는 pixel 단위의 feature extractor $F(x_i,p)$를 통해 모든 pixel location $p \in P$ 에 대한 $\kappa$ nearest gallery of features, $G=\left\{F\left(x_{1}, p\right) \mid p \in P\}\left.\cup\left\{F\left(x_{2}, p\right) \mid p \in P\right\}\right\} . . \cup\left\{F\left(x_{K}, p\right) \mid p \in P\right\}\right\}$ 를 사용한다. 

$$d(y, p)=\frac{1}{\kappa} \sum_{f \in N_{\kappa}(F(y, p))}\|f-F(y, p)\|^{2}$$

모든 $K$ nearest normal image에서 threshold $\theta$에 대해 $d(y,p) > \theta$ 를 만족하는 경우, 즉 가까운 pixel을 찾지 못하는 경우에 해당 pixel을 anomalous로 판정한다.

# Implementation Details

Dataset으로 MVTec AD, STC를 256 으로 resize하여 사용하였다.

feature extractor로는 ImageNet 으로 pre-trained 된 Wide-ResNet50x2를 사용하였다.

feature는 end of the first block (56 x 56), second block (28 x 28), third block(14 x 14)를 같은 중요도로 두고 사용하였다. 

Nearest neighbour 갯수($K, \kappa$)는 MVtec 의 경우 $K=50, \kappa = 1$, STC의 경우 $K=1, \kappa = 1$ 로 설정하였다.

pixel-wise anomaly score의 경우 계산 후 Gaussian filter($\sigma = 4$)로 smoothing 하였다.

# Experimental results

## MVTec AD

![Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled.png](Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled.png)

![Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%201.png](Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%201.png)

## Shanghai Tech Campus Dataset

![Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%202.png](Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%202.png)

![Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%203.png](Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%203.png)

# Ablation Study

Table 6에서 feature pyramid의 각 feature의 resolution에 따른 결과를 확인 할 수 있다. 전체적으로 이들을 함께 사용했을 때 좋은 성능을 보임을 확인 할 수 있다.

![Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%204.png](Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%204.png)

Table 7에서 stage 1에서 K개의 random image를 사용하는 것과 K nearest neighbor를 사용하는 것의 차이를 보여준다. 특히 image 사이의 variation이 큰 "Grid" class의 경우 효과가 컸다고 한다.

![Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%205.png](Sub-Image%20Anomaly%20Detection%20with%20Deep%20Pyramid%20Corr%2099ec570eb7ca4a50baaf5f958ab867cf/Untitled%205.png)

---

# Discussion

## Anomaly detection via alignment

## The role of context for anomaly detection

## Optimizing runtime performance

speedup techniques (e.g. KDTrees) for large scale datasets.

image-level kNN → pixel-level kNN reduces computation time.

## Pre-trained vs. learned features

# Conclusion

Our method consists of two stages, which are designed to achieve high accuracy and reasonable computational complexity.