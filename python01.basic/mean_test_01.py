# 데이터 정의
data = [65, 80, 45, 75, 85, 70]

# 평균 계산 (방법 1: 내장 함수 사용)
mean_value = sum(data) / len(data)
print(mean_value)  # 70.0

# 평균 계산 (방법 2: numpy 사용)
import numpy as np
mean_np = np.mean(data)
print(mean_np)  # 70.0