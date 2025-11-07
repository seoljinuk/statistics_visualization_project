import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42)
x = np.random.randn(20, 3) # 20행 3열의 랜덤 데이터 셋
# print(x)

y = np.concatenate([np.zeros(10), np.ones(10)])
# print(y)

data = np.column_stack((x, y))
print(data)
print('data.shape : ' + str(data.shape))

# x : 입력 데이터(시험 문제지)_특성(feature)
x = data[:, :-1] # 마지막 열을 제외한 추출

# y : 출력 데이터(답지_label)
y = data[:, -1] # 마지막 열만 추출

# x_train : 내가 열심히 공부하는 기출 문제지
# y_train : 공부하면서 같이 보는 답지
# x_test : 열심히 공부했는지 점검하기 위한 새로운 시험지
# y_test : 새로운 시험지에 대한 답지
x_train, x_test, y_train, y_test = \
    train_test_split(x, y, test_size=0.3, random_state=42, stratify=y)

print('\nx_train 특성')
print(x_train)

print('\ny_train 레이블')
print(y_train)

print('\nx_test 특성')
print(x_test)

print('\ny_test 레이블')
print(y_test)
