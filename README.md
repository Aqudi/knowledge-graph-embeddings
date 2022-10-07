# KGE

## 프로젝트 목표

- KGE methods 구현

## 데이터셋

### 다운로드

#### TransE

- [프로젝트 페이지](https://everest.hds.utc.fr/doku.php?id=en:transe)
- 데이터셋 다운로드 링크
  - (wordnet-mlj12.tar.gz)[https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz]
  - (fb15k.tgz)[https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:fb15k.tgz]

## Models

### TransE

#### 문제점 및 해결

- 2022-10-07 논문에서 entity normalization을 잘못 이해하여 embedding을 entity 수로 나눴었다.
  - 이로 인해 loss 계산이 매우 잘못되어 학습이 제대로 진행이 안되는 문제가 있었음
  - 이를 torch.nn.functionla.normalize 함수를 사용하여 vector normalization으로 변경
