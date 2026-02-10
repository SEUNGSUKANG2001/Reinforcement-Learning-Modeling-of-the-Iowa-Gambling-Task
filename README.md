# Iowa Gambling Task 강화학습 모델링

Iowa Gambling Task(IGT)를 강화학습으로 모델링하고, 실험 데이터를 분석하여 참가자들의 의사결정 패턴을 연구하는 프로젝트입니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [Iowa Gambling Task란?](#iowa-gambling-task란)
- [프로젝트 구조](#프로젝트-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [주요 기능](#주요-기능)
- [실험 결과](#실험-결과)

## 🎯 프로젝트 개요

본 프로젝트는 Iowa Gambling Task를 강화학습 환경으로 구현하고, 실제 참가자 데이터를 분석하여 의사결정 과정을 모델링합니다. Q-learning과 Valence-Specific Q-learning 알고리즘을 사용하여 에이전트를 학습시키고, Maximum Likelihood Estimation을 통해 참가자의 행동 패턴을 분석합니다.

### 주요 목표
- IGT 환경의 강화학습 구현
- 참가자 데이터의 전처리 및 분석
- 최대우도추정(MLE)을 통한 파라미터 추정
- 클러스터링을 통한 참가자 그룹 분류
- 학습자와 비학습자의 행동 패턴 비교

## 🎮 Iowa Gambling Task란?

Iowa Gambling Task는 의사결정 과정을 연구하기 위한 심리학 실험 과제입니다.

### 게임 규칙
- **목표**: 최대한 많은 돈을 획득
- **시작 금액**: $2000
- **선택지**: 4개의 카드 덱 (A, B, C, D)
- **시행 횟수**: 100회 또는 150회

### 카드 덱 특성

| 덱 | 보상 | 손실 | 기대값 | 전략 |
|---|------|------|--------|------|
| **A** | +100 | 0 ~ -350 (균등) | 불리 | 높은 보상, 높은 위험 |
| **B** | +100 | 0 또는 -1250 (10%) | 불리 | 높은 보상, 매우 높은 위험 |
| **C** | +50 | 0 ~ -75 (변동) | 유리 | 낮은 보상, 낮은 위험 |
| **D** | +50 | 0 또는 -250 (10%) | 유리 | 낮은 보상, 중간 위험 |

## 📁 프로젝트 구조

```
Reinforcement-Learning-Modeling-of-the-Iowa-Gambling-Task/
│
├── code final/
│   │
│   ├── Data_Preprocessing/           # 데이터 전처리
│   │   ├── Data_Preprocessing.ipynb  # 원본 데이터 전처리 노트북
│   │   ├── choice_100.csv            # 100회 시행 선택 데이터
│   │   ├── choice_150.csv            # 150회 시행 선택 데이터
│   │   ├── wi_100.csv                # 100회 시행 보상 데이터
│   │   ├── wi_150.csv                # 150회 시행 보상 데이터
│   │   ├── lo_100.csv                # 100회 시행 손실 데이터
│   │   └── lo_150.csv                # 150회 시행 손실 데이터
│   │
│   ├── IGT_Environment_and_agent/    # 강화학습 환경 및 에이전트
│   │   ├── iowa_env.py               # IGT Gymnasium 환경 구현
│   │   ├── iowa_gambling_env.ipynb   # 환경 테스트 노트북
│   │   ├── Q_learning.ipynb          # Q-learning 에이전트
│   │   └── Valence-Specific Q-learning.ipynb  # Valence-Specific Q-learning
│   │
│   ├── IGT_Parameter Estimation_and_clustering/  # 파라미터 추정 및 클러스터링
│   │   ├── IGT_Maximum_Likelihood_Estimation.ipynb  # MLE 파라미터 추정
│   │   ├── clustering.ipynb          # 학습자/비학습자 클러스터링
│   │   ├── data_plotting.ipynb       # 전체 데이터 시각화
│   │   ├── learners_plotting.ipynb   # 학습자 데이터 시각화
│   │   ├── IGT_learners_logistic_plot.ipynb  # 로지스틱 회귀 분석
│   │   ├── data.json                 # 전처리된 전체 데이터
│   │   ├── learners_sub.json         # 학습자 데이터
│   │   └── params_sub.json           # 추정된 파라미터
│   │
│   ├── IGT_postclustering_parameter_applied/  # 클러스터링 후 파라미터 적용
│   │   ├── q_learning_label.ipynb
│   │   ├── Valence-Specific Q-learning_label.ipynb
│   │   └── iowa_env.py
│   │
│   ├── IGT_team_parameter_applied/   # 팀 파라미터 적용
│   │   ├── IGT_MLE_ours.ipynb
│   │   ├── q_learning_ours.ipynb
│   │   ├── Valence_Specific_Q_learning_ours.ipynb
│   │   └── iowa_env.py
│   │
│   └── IGT_game_playing.ipynb        # 대화형 IGT 게임 실행
│
├── final ppt.pptx                    # 최종 발표 자료
├── final report.docx                 # 최종 보고서
└── README.md                         # 프로젝트 문서 (본 파일)
```

## 🚀 설치 방법

### 필수 요구사항
- Python 3.8 이상
- Jupyter Notebook 또는 JupyterLab

### 패키지 설치

```bash
# 프로젝트 클론
git clone https://github.com/SEUNGSUKANG2001/Reinforcement-Learning-Modeling-of-the-Iowa-Gambling-Task.git
cd Reinforcement-Learning-Modeling-of-the-Iowa-Gambling-Task

# 필수 패키지 설치
pip install numpy pandas matplotlib gymnasium scikit-learn scipy
```

## 💻 사용 방법

### 1. IGT 게임 직접 플레이하기

```bash
cd "code final"
jupyter notebook IGT_game_playing.ipynb
```

노트북을 실행하면 대화형으로 IGT 게임을 플레이할 수 있습니다. 결과는 JSON 파일로 자동 저장됩니다.

### 2. 데이터 전처리

```bash
cd "code final/Data_Preprocessing"
jupyter notebook Data_Preprocessing.ipynb
```

원본 실험 데이터를 전처리하여 분석 가능한 형태로 변환합니다.

### 3. 강화학습 환경 테스트

```bash
cd "code final/IGT_Environment_and_agent"
jupyter notebook iowa_gambling_env.ipynb
```

IGT Gymnasium 환경이 올바르게 작동하는지 테스트합니다.

### 4. 강화학습 에이전트 학습

**Q-learning 에이전트:**
```bash
jupyter notebook Q_learning.ipynb
```

**Valence-Specific Q-learning 에이전트:**
```bash
jupyter notebook "Valence-Specific Q-learning.ipynb"
```

### 5. 파라미터 추정 및 분석

```bash
cd "code final/IGT_Parameter Estimation_and_clustering"

# 최대우도추정
jupyter notebook IGT_Maximum_Likelihood_Estimation.ipynb

# 클러스터링
jupyter notebook clustering.ipynb

# 시각화
jupyter notebook data_plotting.ipynb
jupyter notebook learners_plotting.ipynb
```

## ✨ 주요 기능

### 1. IGT Gymnasium 환경 (`iowa_env.py`)

강화학습 표준 인터페이스인 Gymnasium을 따르는 IGT 환경 구현:

- `reset()`: 환경 초기화
- `step(action)`: 행동 수행 및 보상 반환
- `get_history()`: 선택 및 보상 이력 조회
- `get_score()`: 현재 점수 조회
- `render()`: 현재 상태 출력

### 2. 강화학습 알고리즘

**Q-learning**
- 전통적인 Q-learning 알고리즘
- 모든 보상을 동일하게 처리

**Valence-Specific Q-learning**
- 긍정적 보상과 부정적 보상을 별도의 학습률로 처리
- 인간의 비대칭적 학습 패턴 반영

### 3. 파라미터 추정

**Maximum Likelihood Estimation (MLE)**를 통해 다음 파라미터 추정:
- **α (학습률)**: 새로운 정보에 대한 학습 속도
- **β (역온도 파라미터)**: 탐험 vs 활용 균형
- **α_pos, α_neg** (Valence-Specific): 긍정/부정 보상에 대한 별도 학습률

### 4. 클러스터링

참가자를 학습자(Learners)와 비학습자(Non-learners)로 분류:
- 학습자: 시행이 진행됨에 따라 유리한 덱(C, D)을 선택하는 비율 증가
- 비학습자: 무작위 선택 또는 불리한 덱(A, B) 지속 선택

### 5. 데이터 시각화

- 시행별 덱 선택 패턴
- 누적 보상 변화
- 학습 곡선
- 로지스틱 회귀를 통한 학습 추세 분석

## 📊 실험 결과

실험 결과는 다음 파일에서 확인할 수 있습니다:
- **최종 보고서**: `final report.docx`
- **발표 자료**: `final ppt.pptx`

### 주요 발견사항
- 참가자들의 학습 패턴은 개인차가 크게 나타남
- Valence-Specific Q-learning이 인간의 비대칭적 학습을 더 잘 설명
- 클러스터링을 통해 명확한 학습자/비학습자 그룹 구분 가능
- MLE를 통한 개인별 파라미터 추정으로 의사결정 전략 파악

---

**Keywords**: Iowa Gambling Task, 강화학습, Q-learning, 의사결정 모델링, 파라미터 추정, 클러스터링, 심리학 실험
