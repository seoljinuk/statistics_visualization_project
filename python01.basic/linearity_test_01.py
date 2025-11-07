import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 정의
x1 = [3, 5, 8, 11, 13, 8]
y1 = [1, 2, 3, 4, 5, 3]

# 산점도
plt.figure(figsize=(6, 5))
plt.scatter(x1, y1, color='skyblue', edgecolor='black', s=100, label='데이터')

# 회귀직선(최소제곱선) 계산
m, b = np.polyfit(x1, y1, 1)  # 1차 직선 (기울기, 절편)
x_line = np.linspace(min(x1), max(x1), 100)
y_line = m * x_line + b

# 직선 그리기
plt.plot(x_line, y_line, color='red', linewidth=2, label=f'회귀직선: y = {m:.2f}x + {b:.2f}')

# 그래프 꾸미기
plt.title("산점도 (공부 시간 vs 학습 향상 정도)", fontsize=14)
plt.xlabel("공부 시간", fontsize=12)
plt.ylabel("학습 향상 정도", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 표시
plt.show()
