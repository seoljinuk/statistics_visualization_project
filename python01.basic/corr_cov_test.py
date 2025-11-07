import matplotlib.pyplot as plt

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 정의
x1 = [3, 5, 8, 11, 13, 8]
y1 = [1, 2, 3, 4, 5, 3]

# 단위를 크게 조정한 데이터
x2 = [v * 100 for v in x1]
y2 = [v * 50 for v in y1]

# 공분산 및 상관 계수 계산 함수
def calc_cov_corr(x, y):
    n = len(x)
    bar_x = sum(x) / n
    bar_y = sum(y) / n

    cov_xy = sum((xi - bar_x) * (yi - bar_y) for xi, yi in zip(x, y)) / (n - 1)
    std_x = (sum((xi - bar_x) ** 2 for xi in x) / (n - 1)) ** 0.5
    std_y = (sum((yi - bar_y) ** 2 for yi in y) / (n - 1)) ** 0.5
    corr_xy = cov_xy / (std_x * std_y)
    return cov_xy, corr_xy

cov1, corr1 = calc_cov_corr(x1, y1)
cov2, corr2 = calc_cov_corr(x2, y2)

# 그래프 생성
plt.figure(figsize=(12, 5))

# 원본 데이터
plt.subplot(1, 2, 1)
plt.scatter(x1, y1, color='skyblue', edgecolor='black', s=100)
plt.title("원본 데이터", fontsize=13)
plt.xlabel("X")
plt.ylabel("Y")
plt.text(min(x1)+0.5, max(y1)-0.5, f"공분산 = {cov1:.2f}\n상관 계수 = {corr1:.4f}",
         fontsize=11, color='darkblue', bbox=dict(facecolor='white', alpha=0.8))
plt.grid(True, linestyle='--', alpha=0.5)

# 확대 데이터
plt.subplot(1, 2, 2)
plt.scatter(x2, y2, color='salmon', edgecolor='black', s=100)
plt.title("단위 확대한 데이터", fontsize=13)
plt.xlabel("X (×100)")
plt.ylabel("Y (×50)")
plt.text(min(x2)*1.02, max(y2)*0.90, f"공분산 = {cov2:.2f}\n상관 계수 = {corr2:.4f}",
         fontsize=11, color='darkred', bbox=dict(facecolor='white', alpha=0.8))
plt.grid(True, linestyle='--', alpha=0.5)

plt.suptitle("공분산과 상관 계수 비교", fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
