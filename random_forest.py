import glob
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt


def adjust_length_tuple(data, target_length=1190):
    if len(data) < target_length:
        # 길이가 1000보다 작은 경우, (0, 0)으로 채움
        data.extend([(0, 0)] * (target_length - len(data)))
    elif len(data) > target_length:
        # 길이가 1000보다 큰 경우, 처음 1190개만 유지
        data = data[:target_length]
    adjusted_data = data
    return adjusted_data


def adjust_length_int(data, target_length=1190):
    if len(data) < target_length:
        # 길이가 1000보다 작은 경우, (0, 0)으로 채움
        data.extend([0] * (target_length - len(data)))
    elif len(data) > target_length:
        # 길이가 1000보다 큰 경우, 처음 1190개만 유지
        data = data[:target_length]
    adjusted_data = data
    return adjusted_data


file_source = "./data/labels/"
anomaly = ["fall", "broken", "fight", "fire", "smoke", "theft"]

input = []
label = []
for idx, item in enumerate(anomaly):
    files = glob.glob(os.path.join(file_source, f'{anomaly}*.txt'))
    for file in files:
        with open(os.path.join(file), "r", encoding="utf-8") as f:
            line = f.readline()
            split_line = line.split(', ')
            int_list = [int(x) for x in split_line[:-1]]
            # tuple_list = []
            # for i in range(len(int_list)):
            #     if i % 2 == 0:
            #         t = (int_list[i], int_list[i + 1])
            #         tuple_list.append(t)
            # input.append(adjust_length(tuple_list))
            input.append(adjust_length_int(int_list))
            label.append(split_line[-1])

# 데이터를 학습 및 테스트 세트로 분리
X_train, X_test, y_train, y_test = train_test_split(input, label, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성 및 하이퍼파라미터 튜닝
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [3, 4, 5],
    'max_depth': [5, 10, 15],
    # 'min_samples_split': [2, 4, 6]
}

# 그리드 서치를 이용한 최적의 하이퍼파라미터 탐색
grid_search = GridSearchCV(rf, param_grid, cv=3, verbose=1, n_jobs=-1, error_score='raise')
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터를 가진 모델
best_rf = grid_search.best_estimator_

# 테스트 데이터에 대한 예측
y_pred = best_rf.predict_proba(X_test)

# 로스 계산
loss = log_loss(y_test, y_pred)

# 로스 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_test)), y_pred[:, 1], label="Predicted Probability")
plt.plot(range(len(y_test)), y_test, label="Actual Label", alpha=0.7)
plt.title("Predicted Probabilities and Actual Labels")
plt.xlabel("Sample Index")
plt.ylabel("Probability / Label")
plt.legend()
plt.show()

