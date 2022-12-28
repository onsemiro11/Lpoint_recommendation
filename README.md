# Matrix Factorization을 활용한 상품 추천 시스템

## EDA & SGD Matrix Factorization

## 주제선정 배경 및 소개

공모전 주제인 [고객 구매 데이터에 기반한 추천 모델 개발 및 개인화 마케팅 전략 제안]에서 벗어나지 않고
고객과 기업에게 유익한 정보와 시스템을 제공할 수 있는 초점을 두고 대회에 임하였다.
모델을 통해 선정된 상품을 고객에게 추천함으로써, 고객 이탈을 최소화 해 줄 수 있다. 
또한, 구매를 예측하여 광고 및 인센티브 부여 결정에 도움을 줄 수 있다.
최종적으로 구매에 보다 빠르게 도달 할 수 있도록 도와, 기업과 서비스를 이용하는 고객 모두에게 이점을 부여하는 것이 목표다.

## 모델 설명 - Matrix Factorization

상품과 고객간의 잠재요인을 이용하여 모델을 구현하기 위해, 희소행렬을 분해 할 수 있는 SGD(확률적 경사 하강법)방식을 이용해 MF를 수행한다.

SGD를 이용한 행렬 분해(MF)는 User Latent(P) 와 Item Latent(Q) matrix로 계산된 예측 R matrix값이 실제 R matrix 값과 최소한의 오류를 가질 수 있도록,
하나의 데이터를 추출하여 반복적으로 비용함수를 최적화함으로써 적합한 User Latent와 Item Latent를 유추한다.

![image](https://user-images.githubusercontent.com/49609175/209759232-9b23a26b-aed7-4e6d-afd3-17beeca121cf.png)
![image](https://user-images.githubusercontent.com/49609175/209759244-b242fbf9-56dd-4a89-a40d-d2f05f3d33d7.png)

<행렬분해(MF)의 전반적인 절차>
   1. User Latent와 Item Latent 행렬을 임의의 값을 가진 행렬로 초기화
   2. User Latent와 Item Latent transposition 행렬을 내적한 후, 각 bias를 더하고 실제 R행렬과의 차이를 계산
   3. 차이를 최소화할 수 있도록 User Latent와 Item Latent 행렬의 값을 적절한 값으로 각각 업데이트
   4. 특정 임계치 아래로 수렴할 때까지 반복하면서 User Latent와 Item Latent행렬을 업데이트해 근사화 진행

## 데이터 변환

일반적으로 SGD 모델에 돌아가는 rating matrix 형태로 만들기 위해,
우리가 가지고 있는 데이터로 rating을 대처할 수 있다고 판단한 고객이 상품을 구매한 횟수를 count하여 value값으로 지정하였다.

추가적으로, 0이 아닌 값들은 standardscaler르 진행하여 분산되어 있더 값들을 표준화시켰다.

## 파라미터 설정

* epoch : 모델 진행 반복 횟수
* num_factors : latent fator 개수
* reg_param : L2 normalization 계수
* lr : learning rating

𝑐𝑜𝑠𝑡= √((∑▒〖(𝑅−(𝑃∙𝑄^𝑇+𝑏))" " 〗^2 )/𝑛) 
![image](https://user-images.githubusercontent.com/49609175/209759542-46ae39c3-5f15-4508-b72f-df4e3fd8475d.png)



