import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# ✅ 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# x축 범위
x = np.linspace(0, 10, 500)

# 왜도 파라미터 설정
a_pos = 6   # 양의 왜도 (오른쪽 꼬리)
a_neg = -6  # 음의 왜도 (왼쪽 꼬리)

# PDF 계산
y_pos = skewnorm.pdf(x, a_pos, loc=5, scale=1.2)
y_neg = skewnorm.pdf(x, a_neg, loc=5, scale=1.2)

# 중심 위치 계산
mean_pos = skewnorm.mean(a_pos, loc=5, scale=1.2)
median_pos = skewnorm.median(a_pos, loc=5, scale=1.2)
mode_pos = 5 + 1.2 * (a_pos / np.sqrt(1 + a_pos**2)) * (-1)  # 근사식

mean_neg = skewnorm.mean(a_neg, loc=5, scale=1.2)
median_neg = skewnorm.median(a_neg, loc=5, scale=1.2)
mode_neg = 5 + 1.2 * (a_neg / np.sqrt(1 + a_neg**2)) * (-1)

# 그래프 생성
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

## 오른꼬리 분포 (양의 왜도)
axes[0].plot(x, y_pos, color='green', linewidth=2)
axes[0].fill_between(x, y_pos, color='green', alpha=0.15)

axes[0].axvline(mode_pos, color='gray', linestyle=':', linewidth=1)
axes[0].axvline(median_pos, color='gray', linestyle='--', linewidth=1)
axes[0].axvline(mean_pos, color='black', linestyle='-', linewidth=1)

axes[0].text(mode_pos - 0.4, max(y_pos)*0.9, "최빈값", fontsize=10)
axes[0].text(median_pos - 0.4, max(y_pos)*0.75, "중앙값", fontsize=10)
axes[0].text(mean_pos - 0.4, max(y_pos)*0.6, "평균", fontsize=10)

axes[0].set_title("오른 꼬리 분포 (양의 왜도)", fontsize=13)
axes[0].annotate("오른쪽 꼬리 →", xy=(8.5, 0.03), xytext=(7.0, max(y_pos)*0.5),
                 arrowprops=dict(arrowstyle="->", color='red'), color='red', fontsize=11)
axes[0].set_xlim(0, 10)
axes[0].set_xticks([])
axes[0].set_yticks([])

## 왼꼬리 분포 (음의 왜도)
axes[1].plot(x, y_neg, color='green', linewidth=2)
axes[1].fill_between(x, y_neg, color='green', alpha=0.15)

axes[1].axvline(mean_neg, color='black', linestyle='-', linewidth=1)
axes[1].axvline(median_neg, color='gray', linestyle='--', linewidth=1)
axes[1].axvline(mode_neg, color='gray', linestyle=':', linewidth=1)

axes[1].text(mean_neg - 0.3, max(y_neg)*0.9, "평균", fontsize=10)
axes[1].text(median_neg - 0.3, max(y_neg)*0.75, "중앙값", fontsize=10)
axes[1].text(mode_neg - 0.3, max(y_neg)*0.6, "최빈값", fontsize=10)

axes[1].set_title("왼꼬리 분포 (음의 왜도)", fontsize=13)
axes[1].annotate("← 왼쪽 꼬리", xy=(1.5, 0.03), xytext=(3.0, max(y_neg)*0.5),
                 arrowprops=dict(arrowstyle="->", color='red'), color='red', fontsize=11)
axes[1].set_xlim(0, 10)
axes[1].set_xticks([])
axes[1].set_yticks([])

# ✅ 자동으로 여유 있게 y축 설정
for ax in axes:
    y_max = max(ax.lines[0].get_ydata())
    ax.set_ylim(0, y_max * 1.2)

plt.tight_layout()
# 그래프 저장
plt.savefig("skew_distribution.png", dpi=300, bbox_inches='tight')

