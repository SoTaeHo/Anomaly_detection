import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


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


anomaly_val = "./data/labels/anomaly/validation_anomaly"
anomaly_tr = "./data/labels/anomaly/training_anomaly"
normal_val = "./data/labels/normal/validation_normal"
normal_tr = "./data/labels/normal/training_normal"
anomaly = ["fall", "broken", "fight", "fire", "smoke", "theft"]
normal = ["buying", "select", "compare", "return", "test"]

input = []
label = []

files = glob.glob(os.path.join(anomaly_val, f'*.txt'))
for file in files:
    with open(os.path.join(file), "r", encoding="utf-8") as f:
        line = f.readline()
        split_line = line.split(', ')
        int_list = [int(x) for x in split_line[:-1]]
        input.append(adjust_length_int(int_list))
        if split_line[-1] in anomaly:
            label.append(1)
        else:
            label.append(0)

files = glob.glob(os.path.join(normal_val, f'*.txt'))
for file in files:
    with open(os.path.join(file), "r", encoding="utf-8") as f:
        line = f.readline()
        split_line = line.split(', ')
        int_list = [int(x) for x in split_line[:-1]]
        input.append(adjust_length_int(int_list))
        if split_line[-1] in anomaly:
            label.append(1)
        else:
            label.append(0)

files = glob.glob(os.path.join(normal_tr, f'*.txt'))
for file in files:
    with open(os.path.join(file), "r", encoding="utf-8") as f:
        line = f.readline()
        split_line = line.split(', ')
        int_list = [int(x) for x in split_line[:-1]]
        input.append(adjust_length_int(int_list))
        if split_line[-1] in anomaly:
            label.append(1)
        else:
            label.append(0)

files = glob.glob(os.path.join(anomaly_tr, f'*.txt'))
for file in files:
    with open(os.path.join(file), "r", encoding="utf-8") as f:
        line = f.readline()
        split_line = line.split(', ')
        int_list = [int(x) for x in split_line[:-1]]
        input.append(adjust_length_int(int_list))
        if split_line[-1] in anomaly:
            label.append(1)
        else:
            label.append(0)


print(len(input))
print(len(label))
# 데이터를 훈련, 검증, 테스트 세트로 분리
X_train, X_temp, y_train, y_temp = train_test_split(input, label, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 랜덤 포레스트 모델 생성 및 하이퍼파라미터 튜닝
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'max_features': ['sqrt', 'log2']
}

# 그리드 서치를 이용한 최적의 하이퍼파라미터 탐색
grid_search = GridSearchCV(rf, param_grid, cv=3, verbose=1, n_jobs=-1, error_score=np.nan)
grid_search.fit(X_train, y_train)

# 최적의 하이퍼파라미터를 가진 모델
best_rf = grid_search.best_estimator_

# 검증 데이터를 사용하여 성능 평가
val_predictions = best_rf.predict(X_val)
print("Validation Set Performance:")
print(classification_report(y_val, val_predictions))

# 테스트 데이터에 대한 예측
y_pred = best_rf.predict(X_test)

# 성능 지표 출력 및 시각화
print("Test Set Performance:")
print(classification_report(y_test, y_pred))

# 컨퓨전 매트릭스 시각화
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()