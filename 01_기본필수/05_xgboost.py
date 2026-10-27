"""실습 05: XGBoost도 같은 fit·predict 흐름으로 실행합니다."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# 제공: 이 파일 옆의 정상 CSV 20건을 읽습니다. 현재 터미널 위치와 무관합니다.
HERE = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "requests.csv")
develop, test = train_test_split(raw, test_size=0.2, stratify=raw["failed"], random_state=42)
train, valid = train_test_split(develop, test_size=0.25, stratify=develop["failed"], random_state=42)
# test 4건은 보관만 합니다. 입력은 실행 전에 아는 글자 수입니다.
X_train = train[["input_chars"]]
X_valid = valid[["input_chars"]]
y_train = train["failed"]
y_valid = valid["failed"]

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# 제공: 학습 회차 80·깊이 2·보정 크기 0.05를 고정합니다. CPU로 실행합니다.
model = XGBClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                      objective="binary:logistic", eval_metric="logloss",
                      tree_method="hist", random_state=42, n_jobs=1)
#  n_estimators: 생성할 트리의 개수 즉, 트리를 순차적으로 몇개를 추가할 것인지 나타내는 매개변수
#  max_depth: 각각의 트리를 얼마나 깊게 만들것인지 나타냄
#  learning_rate: 각 트리의 의견을 얼마나 강하게 반영할 것인가? 
#                 -> 어느 트리가 1만큼 변경한다고 하면 lr = 0.05일 경우 1*.05 = 0.05만큼만 변경함 
#  objective: 학습목적함수를 지정 즉, 모델이 무엇을 최적화 하면서 학습할 것인가를 지정 
#             -> binary:logistic의 경우 이진 분류를 수행하며 logistic transformation을 통해 확률을 출력한다.
#  eval_metric: 모델 성능을 어떤 기준으로 평가할 것인가.
#               -> logloss의 경우 예측 확률이 실제 정답과 얼마나 잘 맞는지를 평가하는 지표임. 트리 A와 B가 확률을 A: 0.95, B:0.51로 
#                  예측했다면 트리A가 훨씬 자신있게 정답에 가까운 확률을 냈음. 이러한 확률 예측의 품질까지 평가함 -> logloss는 낮을수록 좋음
#  TODO B05: train으로 학습하고 valid를 예측하세요.
model.fit(X_train, y_train)
prediction = model.predict(X_valid)
# TODO 끝
print("valid 예측:", prediction.tolist())
print("valid F1:", round(f1_score(y_valid, prediction, zero_division=0), 3))
print("학습 목적: 실패 분류 / 확인 지표: F1")
