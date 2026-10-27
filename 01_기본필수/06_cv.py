"""실습 06: train 12건 안에서만 3-fold 교차검증합니다."""
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 제공: 각 클래스 수를 고려해 기본필수는 3-fold입니다. valid4·test4는 넣지 않습니다.
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#  교차검증을 위해 테스트데이터를 3부분으로 나눈다. -> 모든 데이터가 한번씩 검증 데이터 역할을 한다. 
#  'Stratified'의미: 테스트 데이터의 'failed'항목이 만약 "성공70% 실패30%"로 이루어져 있다면 각 fold도 이 비율을 유지함
model = DecisionTreeClassifier(max_depth=3, random_state=42)

scores = cross_val_score( # cross_val_score: 실제 평가를 수행
    model,                
    X_train,
    y_train,
    cv = folds,
    scoring = 'f1'
)
#  model       → 평가할 모델
#  X_train     → 입력 데이터
#  y_train     → 정답
#  cv=folds    → 어떻게 데이터를 나눌 것인가
#  scoring="f1"→ 무엇으로 성능을 측정할 것인가

#  scores : ndarray of float of shape=(len(list(cv)),)
#          Array of scores of the estimator for each run of the cross validation.

print("fold별 F1:", scores.round(3).tolist())
print("평균 F1:", round(scores.mean(), 3))
print("표준편차:", round(scores.std(), 3))
