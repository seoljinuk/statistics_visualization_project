# 표준 정규 분포(평균 키 : 0, 분산 : 1)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 한글 폰트 설정 (Windows용)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터 설정
np.random.seed(0)
mean = 0    # 표준 정규분포 평균
var = 1     # 표준 정규분포 분산
std = np.sqrt(var)

# 정규분포 곡선 (x는 -4 ~ 4 구간)
x = np.linspace(-4, 4, 200)

# 확률밀도함수
y = norm.pdf(x, mean, std)

# 그래프 그리기
plt.plot(x, y, color='blue', linewidth=2, label=f'평균={mean}, 분산={var}')

# 제목 및 축 라벨
plt.title("표준 정규 분포 곡선 (N(0, 1))")
plt.xlabel("z값")
plt.ylabel("확률 밀도")

# 축 범위 및 범례
plt.xlim([-4, 4])
plt.legend(loc='upper left', fontsize=10, frameon=False)
plt.grid(False)

plt.show()
