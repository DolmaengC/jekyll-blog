---
title: Deep learning
categories:
- Deep learning
excerpt: |
  딥러닝에 대한 간단한 정리 자료입니다.
feature_text: |
  ## The Pot Still
  The modern pot still is a descendant of the alembic, an earlier distillation device
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
---
#### Activation function

주요 활성화 함수

- logistic 함수 (Sigmoid function) : 요즘은 잘 안씀
- Hyperblolic Tangent (tanh) : 텍스트, sequence data, 다른 분야에선 ㄴㄴ
- Rectified Linear Unit (Relu) : 이미지, 주로 많이 사용됨
- Leaky Relu
- Exponential linear unit(ELU)



경사손실 문제 : 은닉층의 한 활성함수값이 0에 가까워지면 다른 은닉층의 값에 상관없이 0에 가까워지는 문제 -> 은닉층이 많을 수록 발생할 확률이 커져서 많이 쌓기 어렵다 -> Relu가 많이 사용되는 이유



## Optimizer

주로 Adam, RMSprop, Adadelta가 사용됨

- Momentum : 이전 업데이트 정보를 기억, 현재 업데이트에 반영하는 방법
  - 장점 (GD에 비해서)
    - local minimum을 잘 피한다.
    - 수렴하는 속도가 빠르다.
  - 주로 다른 기법과 같이 사용됨

- Adagrad (Adaptive Gradient)
  - GD -> 업데이트 횟수와 상관없이 learning rate를 동일하게
  - 하지만, Adagrad는 다르게
    - 지금까지 업데이트 된 정도를 반영한다.
    - 지금까지 업데이트가 많이된 파라미터는 learning rate를 작게
  - 주요 문제
    - GT가 갈수록 커진다. -> 업데이트가 거의 발생하지 않는다. -> 최소지점에 도착하기 전에 업데이트가 멈출 수 있다.
    - Adadelta 
      - 무조건적으로 줄어드는 learning rate 문제를 보완
      - 
    - RMSprop 
      - 무조건적으로 줄어드는 learning rate문제를 보완
      - 합이 아니라 평균을 사용
- Adam 
  - RMSprop (or Adadelta) + momentum



## 가중치 초기화

Random Value로 초기화

- 활성화 함수 : Sigmoid or tanh 인 경우

  - Xavier Weight Initialization : uniform 분포하는  기법

    - 일반적으로 많이 쓰임

    - Keras에서 기본신경망의 경우, kernel_initilization='glory_uniform'로 설정되어 있음 

      ->Xavier 방법

  - Normalized Xavier Weight Initialization : 첫번째랑 비슷한데 구간을 나눔, 잘 안쓰임

- 활성화 함수 : relu

  - He Weight Initialization
    - 주로 이미지 분석(CNN 알고리즘)에서 많이 사용됨





# Object detection

1단계 : 물체가 있을 만한 경계상자(regions of interest, ROI) 추출

2단계 : 추출된 ROI를 이용하여 localization과 classification을 수행

## One stage detectors

- 앞의 두 단계를 한번에 수행
- SSD, YOLO 등

## TWO stage detectors

- 앞의 두 단계를 구분해서 수행
- R-CNN family
- 속도가 느리다는 단점



## SSD

Feagure map 추출

- Object detection을 위해서 사용하는 feature map의 수 6개
  - 더 많은 feature map이 되출되지만 그중 특정 6개만 사용
- 1차적으로 VGG16과 같은 사전학습 모형을 사용하여 feature map 추출, 이후 추가적인 convolutional layer를 사용하여 6개의 feature map 추출
  - VGG16dl cncnfgks feature map 중 1개 사용 + 추가 convolutional layer가 추출한 6개중 5개 사용
    - Feature map 마다의 크기가 다음
  - Multi-scale object detection
    - 크기가 다른 여라개의 feature map을 사용하여 물체 탐지
    - 다양한 크기의 물체를 잘 찾을 수 있음
- Non-maximum suppression
  - 확률이 가장 큰 anchor box를 선택
  - 특정 클래서에 대해서
    - 확룰이 특정값 (예, 0.1) 이하인 상자는 모두 삭제
    - 남아있는 상자들에 대해서 확률값이 제일 큰 상자를 선택
      - 해당 상자와 IoU 값이 0.45 이상인 다른 상자들을 모두 삭제
    - 남아 있는 상자들에 대해서 동일한 과정 반복
      - 확률값이 제일 큰 상자를 선택 -> 해당 상자와 IoU 값이 0.45 이상인 다른 상자들을 모두 삭제



## R-CNN family

R-CNN in 2014, Fast R-CNN in 2015, Faster R-CNN in 2016

1. Input images
2. Extract region proposals(~2k)
3. Compute CNN features
4. Classify regions

- 크게 4부분으로 구성
  - Extract regions of interest(ROIs)
    - Selective search (최근엔 사용하지 않음)
  - Feature extraction module
    - 이전 단계에서 추출된 ROI들에 대해 CNN을 적용해서 feature map 추출
  - Classification module
    - 이전 단계에서 추출된 feature들에 대해서 classification 예측
    - Support vector machine (SVM) 사용
  - Localization module
    - ROI를 기준으로 GTBB에 대한 offset을 예측함
- R-CNN의 주요 단점
  - 계산량이 많아 시간이 오래 걸린다.
- Fast R-CNN
  - 사전 학습모형을 먼저 적용하여 feature map 추출-> 그 다음에 selective search 방법 적용하여 ROI 추출
  - SVM 대신 fully connected layer 사용
  - 단점 : 여전히 selective search 방법을 사용해서 속도가 느림
- Faster R-CNN
  - Region Proposal Network 방식 사용
    - VGG16 등의 사전학습모형을 사용하여 feature map 추출
    - Feature map의 각 셀에 대해서 크기와 형태가 다른 9개의 anchor box 생성
    - 각 anchor box에 대해서 두 가지 종류의 값들 예측
      - Objectness score : 물체가 있을 확률
      - GTBB와의 offset 값들
    - Anchor box에 예측된 offset을 적용하여 ROIs를 추출



## RNN (Recurrent Neural Networks)

- Sequence data를 다루기에 적합
- 텍스트 데이터 분석에 적합
  - 단어들의 문맥적 의미 추출에 용이
  - 단어들 간의 관계 정보 추출에 용이
  - 어떠한 단어들이 어떠한 순서로 언제 사용되었는지에 대한 정보 추출 용이
- 언어모형 (language model), 분류(예, 감성분석), 기계 번역



## LSTM

- RNN 기반의 신경망 알고리즘
- Simple RNN의 문제를 보완하기 위해 제안
  - Problem of long term dependency
    - 입려괸 문서에서 상대적으로 오래전에 사용된 단어의 정보가 잘 전달 되지 않는다.
    - 주요 원인 : 경사손실문제 
- Forget 게이트 
  - 역할 : 이전 기억 셀 (즉, ct-1)이 가지고 있는 정보 중에서 정답을 맞히는데 불필요한 정보는 잊어버리는 역할
    - Ct-1이 가지고 있는 각 원소의 정보 중에서 몇 %를 잊어버릴 것이냐를 결정하기 위해서 0~1사이의 값을 반환하는 sigmoid 함수를 사용
- Input 게이트
  - 일부의 정보가 삭제된 ct-1(즉, ct-1')에 새로운 정보를 추가하는 역할
  - 일단, 추가하고자 하는 정보를 계산 : ht-1와 단어t의 정보를 사용
  - 새롭게 추가되는 정보들의 긍부정 역할을 나타내기 위해 -1~1의 값을 출력하는 tanh()를 사용
  - 그대로 반영되는 것이 아니라, 정답을 맞히는데 있어서 기여하는정도에 따라서 적용되는 비율을 다르게 함 -> 이를 위해 sigmoid 함수 사용
- Output 게이트
  - Output 게이트의 역할은 forget게이트와 input 게이트를 이용하여 업데이트된 기억셀의 정보, 즉 ct,를 이용하여 ht를 계산하는 것
  - output 게이트는 ct가 갖는 원소의 값들을 조정하여 ht를 계산
  - ct가 갖고 있는 원소들 중에서 정답을 맞히는데 중요한 역할을 하는 원소의 비중은 크게하고, 그렇지 않은 원소들의 비중은 작게해서 ht를 계산
  - 각 원소의 비중을 계산하기 위해서 현재 LSTM층에 입력되는 ht-1과 단어t의 정보(xt) 인자로 갖는 sigmoid 함수를 사용
  - ct원소들의 긍부정 역할을 구분하기 위해 tanh 적용

<img width="868" alt="스크린샷 2023-07-21 오전 10 20 06" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/16a9ca88-8e1e-45fb-82c8-6170ec975bb3">

## Bidirectional LSTM

<img width="890" alt="스크린샷 2023-07-21 오전 10 22 09" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/466927bc-3d09-41a4-ae18-10c162481736">



## GRU

- LSTM보다 성능이 떨어져서 잘 사용되지 않음

- 게이트 개념 사용

  - 기억셀을 사용하지 않음

  - Rest 게이트와 update 게이트 

    - hidden state 정보를 업데이트하기 위해서 reset 게이트와 update 게이트 사용 

    - GRU는 LSTM 보다는 간단한 구조
      - 속도는 빠느라 정확도가 떨어짐

<img width="838" alt="스크린샷 2023-07-21 오전 10 09 57" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/1b06c03f-76f3-4410-b078-c3c334d4cfae">





## seq2seq

- 주요 목적

  - To convert sequences from one domain to sequences in anotjer domain, 번역이 대표적인 예
  - Encoder-decoder 구조
  - Encoder의 역할
    - 입력뙨 텍스트 데이터를 숫자 형태로 혹은 벡터 형태로 변환

  <img width="798" alt="스크린샷 2023-07-21 오전 11 06 12" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/610972e5-86b6-4a31-bf05-571b920d2c8c">

  - Decoder의 역할
    - Encoder에 의해 숫자로 변경된 정보를 다른 형태의 텍스트 데이터로 변환

  <img width="1261" alt="스크린샷 2023-07-21 오전 11 06 37" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/af3853a1-3362-4ade-ac83-214813756795">

  - Encoder와 Decoder를 위해 순환신경망 기반 모형 사용 가능 (예, RNN)
    - 첫번째 RNN이 encoder 역할, 두번째 RNN이 decoder 여할



## Transformer

- 소개
  - 2017년에 google에서 제안한 attention 기반의 encoder-decoder 알고리즘
    - 순환신경망 기반의 방법이 아니라 attention 사용
    - 주요 applications:
      - BERT (Bidirectional Encoder Representations from Transformers)
        - encoder만 사용
        - 단어 embedding
        - 문서 embedding
        - 분류
        - Q&A
      - GPT (Generative Pre-trained Transformer)
        - decoder만 사용
        - 생성모델 
      - BART (Bidirectional and Auto-Regressive Transformers)
        - Encoder, decoder 둘 다 사용
        - 텍스트 요약
    - Transfomer를 이해하기 위해서는 attention을 먼저 이해하는 것이 필요 



### Attention

#### Encoder-decoder attention

- Why was it proposed?
  - 순환신경망 기반의 seq2seq 모형이 갖는 문제점을 보완하기 위해
  - 순환신경망 기반의 seq2seq의 주요한 문제점
    - 입력된 sequence data에 대해서 하나의 고정된 벡터 정보(마지막 hidden state)만을 decoder로 전달한다는 것
  - 그렇다면 어떻게 하면 되는가?
    - Encoder 부분에서 생성되는 각 단어에 대한 hidden state 정보를 모두 decoder로 전달
    - 벡터정보들을 쌓아서 행렬로 만들어서 전달

<img width="1298" alt="스크린샷 2023-07-21 오후 12 47 34" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/d27533ad-e144-4a43-a557-c29066f9a241">



- 가중치의 계산
  - 가중치는 hs의 각 hidden state와 decoder에 예측하고자 하는 단어에 대한 hidden state와의 유사도를 가지고 계산
  - Hidden state 간의 유사도를 계산 -> 내적 연산
  - decoder 부분의 첫번째 RNN층에서 출력되는 'Today' 단어를 예측하는데 사용되는 hidden state -> hd,0
    - hd,0 = (1 0 0 0 2)
  - h0, h1, h2와 hd,0과의 내적 연산
    - (1 0 0 1 2) *(1 0 0 0 2) = 1 + 4 = 5
    - (1 0 0 1 1) *(1 0 0 0 2) = 1 + 2 = 3
    - (1 0 0 0 1) *(1 0 0 0 2) = 1 + 2 = 3
    - 이 값들을 attention score라고 함
      - Attention score의 값이 클수록 관련도가 크다는 것을 의미
  - Attention score를 가지고 가중치 계산
  - 가중치는 확률값으로 표현
  - 확률값을 계산하기 위해서 attention score에 softmax()를 적용
- 최종적으로 출력되는 값
  - Attention에서 출력되는 값과 RNN 층에서 출력되는 값 간의 이어붙이기(concatenation)
  - 즉, Concat((1.0 0.1 0 0.9 1.8), (1 0 0 0 2))



#### Self-attention

Attention과의 차이

- Attention은 encoder-decoder 모형에서 보통 decoder에서 encoder에서 넘어오는 정보에 가중치를 주는 식으로 작동
- Self-attention은 입력된 텍스트 데이터 내에 존재하는 단어들간의 관계를 파악하기 위해 사용
  - 관련이 높은 단어에 더 많은 가중치를 주기 위해 사용
- 지시대명사가 무엇을 의미하는지 등을 파악하는데 유용



Transformer에서의 self-attention (or attention)

- Transformer의 self-attention은 입력 받은 단어들 중에서 어떠한 단어에 더 많은 가중치를 줘야 하는지 파악하기 위해서 각 단어들에 대한 Query, Key, Value 라고 하는 서로 다른 3개의 벡터들을 사용
  - Key, Value 벡터들은 사전 형태의 데이터 의미 : Key는 단어의 id와 같은 역할, Value는 해당 단어에 대한 구체적 정보를 저장하는 역할
  - Query 벡터는 윳한 다른 단어를 찾을 때 사용되는 (질의) 벡터라고 생각 가능



- 작동 순서

  - 단계1 : 입력된 각 단어들에 대해서 Query, Key, Value 벡터를 계산
    - 이떄 각각의 가중치 행렬이 사용됨 

  <img width="755" alt="스크린샷 2023-07-21 오후 1 20 33" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/d5d77efa-74d0-48ff-8dd1-6f1380076e8a">

  <img width="650" alt="스크린샷 2023-07-21 오후 1 21 40" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/108fbabf-ef05-4249-82e6-90873b69710e">

  - 단계2 : Attention score 계산
    - Query를 이용하여 각 Key들하고의 유사한 정도를 계산 -> 내적 연산

  - 단계3 : Attention score를 이용하여 가중치 계산
    - Softmax() 함수를 적용

  <img width="707" alt="스크린샷 2023-07-21 오후 1 22 20" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/7b1d89bf-9dab-447f-bba4-061d39e93eae">

  - 단계4 : 가중치를 Value 벡터에 곱한다.

  <img width="703" alt="스크린샷 2023-07-21 오후 1 22 40" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/988f1dbc-a832-4e2f-8aa9-1de8a355ad9b">

  - 최종 결과물
    - 가중치가 곱해진 value vector들의 합 

<img width="805" alt="스크린샷 2023-07-21 오후 1 22 57" src="https://github.com/DolmaengC/DolmaengC.github.io/assets/107832431/1992c9bb-8aff-47dc-b597-c879559570ac">













 