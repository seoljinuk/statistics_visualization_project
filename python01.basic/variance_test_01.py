import numpy as np

def calc_variance(ban, data):
    # 평균
    mean_value = np.mean(data)

    # 표준 편차 (모집단 기준)
    std_population = np.std(data)

    # 표준 편차 (표본 기준)
    std_sample = np.std(data, ddof=1)

    # 분산
    variance_population = np.var(data)
    variance_sample = np.var(data, ddof=1)

    print(f"{ban}의 통계 요약")
    print(f"평균(Mean): {mean_value:.2f}")
    print(f"모집단 분산(Variance): {variance_population:.2f}")
    print(f"표본 분산(Variance, ddof=1): {variance_sample:.2f}")
    print(f"모집단 표준 편차(Std Dev): {std_population:.2f}")
    print(f"표본 표준 편차(Std Dev, ddof=1): {std_sample:.2f}")
    print("-" * 40)
# end def

# 반별 점수 데이터
datalist = {
    "1반": [70, 71, 69, 70, 70],
    "2반": [50, 70, 90, 50, 90]
}

# 각 반의 데이터를 반복 처리
for ban, scores in datalist.items():
    calc_variance(ban, scores)
