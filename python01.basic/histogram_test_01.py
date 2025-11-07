import matplotlib.pyplot as plt
import numpy as np

# ✅ 한글 폰트 설정 (Windows)
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 점수 구간과 학생 수
bins = [(20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
freq = [1, 3, 8, 12, 18, 10, 6, 2]

# 각 구간의 중앙값 계산
mid_points = [(low + high) / 2 for low, high in bins]

# 도수에 따라 데이터를 반복 생성
data = []
for mid, f in zip(mid_points, freq):
    data.extend([mid] * f)

# ✅ 히스토그램 생성 (matplotlib)
plt.figure(figsize=(8, 5))
plt.hist(data, bins=[20,30,40,50,60,70,80,90,100], color='skyblue', edgecolor='black')

# ✅ 제목 및 축 라벨 설정
plt.title("점수 구간별 학생 수 히스토그램", fontsize=14)
plt.xlabel("점수 구간", fontsize=12)
plt.ylabel("학생 수", fontsize=12)
plt.xticks(np.arange(20, 110, 10))

# ✅ 그래프 출력
# plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(False)
filename = 'histogram_test_01.png'
plt.savefig(filename, dpi=400, bbox_inches='tight')

