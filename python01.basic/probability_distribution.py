import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')  # 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터
X = np.array([3, 2, 2, 1, 2, 1, 1, 0])

# 각 값의 빈도수 계산
values, counts = np.unique(X, return_counts=True)

# 확률 분포 계산 (빈도수를 전체 개수로 나누기)
prob = counts / len(X)

# 막대그래프 출력
plt.bar(values, prob)
plt.xlabel("확률 변수 X")
plt.ylabel("확률 P(X)")
plt.title("확률 분포 시각화")

# X축 눈금을 0, 1, 2, 3 정수 단위로 지정
plt.xticks(np.arange(min(values), max(values) + 1, 1))

# 그래프 저장 (파일명과 해상도 설정 가능)
plt.savefig("확률분포.png", dpi=300, bbox_inches='tight')
