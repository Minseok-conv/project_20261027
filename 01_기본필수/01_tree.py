"""실습 01: 입력을 질문으로 나누는 트리를 만듭니다. 성능 평가가 아닌 원리 예제입니다."""
from sklearn.tree import DecisionTreeClassifier
#DicisionTree: 모델이 입력값과 출력값을 학습하여 어떤 값을 기준으로 분류할지 스스로 학습함
X = [[100], [200], [300], [700], [800], [900]]
y = [0, 0, 0, 1, 1, 1]  # 같은 순서의 정답: 성공 0, 실패 1
model = DecisionTreeClassifier(max_depth=1, random_state=42)
# TODO B01a: X와 y로 model을 학습하세요.
model.fit(X,y)

# TODO B01b: 250과 850을 예측하여 prediction에 넣으세요.
prediction = model.predict([[250],[850]])
#model.tree_는 학습된 결정트리의 내부 구조를 담고 있음 (끝에 '_'은 학습된 것을 의미함)
print(model.tree_)
print("분기 기준:", model.tree_.threshold)
print("250·850 예측:", prediction.tolist())
