# 정규 분포 02(동일 분산에 평균이 다름)
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 한글 폰트 설정 (Windows용)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 데이터 설정
np.random.seed(0)
mean1 = 165   # 평균 1
mean2 = 175   # 평균 2
var = 50      # 분산 동일
std = np.sqrt(var)

# 정규분포 곡선 (100~230 구간)
x = np.linspace(100, 230, 200)

# 빈도수 비례로 스케일 조정
y1 = norm.pdf(x, mean1, std) * 100 * 10 / len(x)
y2 = norm.pdf(x, mean2, std) * 100 * 10 / len(x)

# 두 곡선 그리기 (범례에 평균·분산 표시)
plt.plot(x, y1, color='red', linewidth=2, label=f'평균={mean1}, 분산={var}')
plt.plot(x, y2, color='blue', linewidth=2, label=f'평균={mean2}, 분산={var}')

# 제목 및 축 라벨
plt.title("성인 100명의 키 정규 분포 곡선")
plt.xlabel("키(cm)")
plt.ylabel("빈도수(비례)")
plt.xlim([120, 230])

# 범례 표시 및 격자 제거
plt.legend(loc='upper left', fontsize=10, frameon=False)
plt.grid(False)

plt.show()
