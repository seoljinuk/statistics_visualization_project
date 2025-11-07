import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

matplotlib.rcParams['font.family'] = ['Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

from scipy.stats import norm, skew, kurtosis

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataIn = './../dataIn/'
dataOut = './../dataOut/'

df = pd.read_csv(dataIn + '고객복지데이터셋.csv')

print("\n컬럼명 나열:")
print(df.columns)

object_columns = df.select_dtypes(include=['object']).columns.tolist()

print("\nobject 타입 컬럼명:")
print(object_columns)
# ['고객ID', '월소득', '성별', '결혼상태', '고용형태', '학력', '복지등급', '지역',
# '장애여부', '웰니스참여', '보육지원', '정신건강지원', '기록일자']

numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

print("\n숫자형 컬럼명:")
print(numeric_columns)
# ['나이', '근속연수', '복지비사용액', '만족도점수', '건강지수', '지원인원']

print("\n각 object 컬럼의 unique 개수 출력")
for col in object_columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count}개")

print("\n각 숫자형 컬럼의 unique 개수 출력")
for col in numeric_columns:
    unique_count = df[col].nunique()
    print(f"{col}: {unique_count}개")

print("\n전처리 시작")
df['정신건강지원'] = df['정신건강지원'].map({'아니오': 0, '예': 1}).astype(int)
print("정신건강지원의 고유값:", df['정신건강지원'].unique())
print("정신건강지원의 고유값 개수:", df['정신건강지원'].nunique())

# 성별의 고유값 및 개수 출력
print("성별의 고유값:", df['성별'].unique())
print("성별의 고유값 개수:", df['성별'].nunique())

print("\n성별 전처리")
# 성별의 고유값: ['여성' '남성' nan '기타']
# nan과 '기타' 제거
df = df[~df['성별'].isin(['기타'])].dropna(subset=['성별'])
print("성별의 고유값:", df['성별'].unique())
print("성별의 고유값 개수:", df['성별'].nunique())

# '월소득'이 문자열인가?
print("\n월소득 컬럼의 unique 값 예시:")
unique_sorted = sorted(df['월소득'].astype(str).unique())
print("앞에서 10개 추출")
print(unique_sorted[:10])

print("뒤에서 10개 추출")
print(unique_sorted[-10:])
# '?', 'nan' 항목이 잘못 되었음을 발견함

df['월소득'] = df['월소득'].replace('?', np.nan)
df['월소득'] = df['월소득'].astype('float')

print("\n데이터 프레임 요약 정보:")
print(df.info())

def print_missing_values(myframe):
    print('\n결측치 확인 하기')
    # 결측치가 있는 컬럼만 추출하여 개수 출력
    missing_columns = myframe.isnull().sum() # boolean의 sum은 개수를 의미합니다.
    missing_columns = missing_columns[missing_columns > 0]

    print("결측치가 있는 컬럼과 개수:")
    for col, cnt in missing_columns.items():
        print(f"{col} : {cnt}개")
# end def

print_missing_values(df)

print('\n결측치 처리하기')
# 결측치들을 비결측치들의 평균 값으로 대체하도록 합니다.
print('결측치를 평균 값으로 대체 : 나이, 월소득, 복지비사용액, 근속연수, 만족도점수, 건강지수 ')
fill_mean_cols = ['나이', '월소득', '복지비사용액', '근속연수', '만족도점수', '건강지수']

for col in fill_mean_cols:
    mean_value = df[col].mean()
    df[col] = df[col].fillna(mean_value)

print_missing_values(df)

print('결측치를 제거 : 성별, 결혼상태, 지원인원 ')
df = df.dropna(subset=['성별', '결혼상태', '지원인원'])
print_missing_values(df)

print('\n결측치 처리 후 데이터 확인')
print(df.info())

salary = df['월소득']

def print_descriptive_tatistics(concern):
    print('\n기술 통계(Descriptive Statistics)')
    # 기본 통계량
    mean_val = concern.mean()              # 평균
    median_val = concern.median()          # 중앙값
    min_val = concern.min()                # 최소값
    max_val = concern.max()                # 최대값
    mode_val = concern.mode().values[0]    # 최빈값
    var_val = concern.var()                # 분산
    std_val = concern.std()                # 표준 편차
    quantiles = concern.quantile([0.25, 0.5, 0.75])  # 사분위수

    # 빈도수 (Frequency)
    freq_table = concern.value_counts().sort_index()

    # 출력
    print(f"평균 (Mean): {mean_val:.2f}")
    print(f"중앙값 (Median): {median_val:.2f}")
    print(f"최소값 (Min): {min_val}")
    print(f"최대값 (Max): {max_val}")
    print(f"최빈값 (Mode): {mode_val}")
    print(f"분산 (Variance): {var_val:.2f}")
    print(f"표준 편차 (Standard Deviation): {std_val:.2f}")
    print("\n사분위수 (Quartiles):")
    print(quantiles)

    print("\n빈도 (Frequency, 상위 10개):")
    print(freq_table.head(10))
# end def

print_descriptive_tatistics(salary)


def draw_boxplot(boxdata, file_name):
    print('\nBoxplot 시각화')
    print('이상치 파악에 많은 도움이 되는 그래프입니다')
    plt.figure(figsize=(4, 6))
    plt.boxplot(boxdata, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='gray'),
                capprops=dict(color='gray'),
                flierprops=dict(marker='o', markerfacecolor='orange', markersize=6, linestyle='none'))

    plt.title('월소득(Boxplot)', fontsize=14)
    plt.ylabel('월소득', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(dataOut + file_name)
# end def

draw_boxplot(salary, 'a01.salary_boxplot_old.png')

def draw_histogram(mydata, file_name, mode='density'):
    """
    mode : 'density' (확률 밀도) 또는 'frequency' (빈도)
    """
    # 평균과 표준 편차 계산
    mean = mydata.mean()
    std = mydata.std()

    # 왜도와 첨도 계산
    skewness = skew(mydata)
    kurt = kurtosis(mydata)

    # 정규 분포 곡선을 위한 데이터 생성
    x = np.linspace(mydata.min(), mydata.max(), 100)
    y = norm.pdf(x, mean, std)

    # density 여부 설정
    density_mode = True if mode == 'density' else False

    # 그래프 설정
    plt.figure(figsize=(8, 5))
    counts, bins, patches = plt.hist(
        mydata, bins=30, density=density_mode, alpha=0.6, color='skyblue',
        edgecolor='black', label='실제 분포'
    )

    # 확률 밀도일 경우에만 정규 분포 곡선 표시
    if density_mode:
        plt.plot(x, y, 'r-', linewidth=2, label='정규 분포 곡선')

    # 제목 및 레이블
    plt.title(f'월소득 분포 시각화 ({ "확률 밀도" if density_mode else "빈도" })', fontsize=14)
    plt.xlabel('월소득')
    plt.ylabel('확률 밀도' if density_mode else '빈도')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(dataOut + file_name)

    # --- 추가 부분: 가장 많은/적은 빈도 구간 출력 ---
    max_idx = np.argmax(counts)
    min_idx = np.argmin(counts)

    print("\n[히스토그램 구간별 빈도 분석]")
    print(f"가장 많은 빈도: {counts[max_idx]:.2f} (구간: {bins[max_idx]:.2f} ~ {bins[max_idx+1]:.2f})")
    print(f"가장 적은 빈도: {counts[min_idx]:.2f} (구간: {bins[min_idx]:.2f} ~ {bins[min_idx+1]:.2f})")

    # --- 왜도 & 첨도 출력 ---
    print("\n[분포 형태 분석]")
    print(f"왜도 (Skewness): {skewness:.4f} → {'오른쪽 꼬리 (양의 왜도)' if skewness > 0 else '왼쪽 꼬리 (음의 왜도)' if skewness < 0 else '대칭'}")
    print(f"첨도 (Kurtosis): {kurt:.4f} → {'뾰족한 분포 (양의 첨도)' if kurt > 0 else '평평한 분포 (음의 첨도)' if kurt < 0 else '정규 분포와 유사'}")

# end def

print('\n이상치 때문에 그래프가 신빙성이 떨어집니다')
draw_histogram(salary, 'b01.salary_histogram_old.png')

# 사분위수 계산
Q1 = salary.quantile(0.25)
Q3 = salary.quantile(0.75)
IQR = Q3 - Q1

# 이상치 기준
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 탐지
outliers = salary[(salary < lower_bound) | (salary > upper_bound)]

print(f'Q1 (1사분위수): {Q1}')
print(f'Q3 (3사분위수): {Q3}')
print(f'IQR: {IQR}')
print(f'하한값: {lower_bound}, 상한값: {upper_bound}')
print(f'이상치 개수: {len(outliers)}')
print('\n이상치 (상위 20개):')
print(outliers.head(20))

print(f"\n이상치 제거 전: {len(df)}행")
df = df[~((df['월소득'] < lower_bound) | (df['월소득'] > upper_bound))]

print(f"이상치 제거 후: {len(df)}행")

newsalary = df['월소득']
print_descriptive_tatistics(newsalary)

draw_boxplot(newsalary, 'a02.salary_boxplot_new.png')
draw_histogram(newsalary, 'b02.salary_histogram_new.png')
draw_histogram(newsalary, 'b03.salary_histogram_new.png', 'frequency')


print("\n상관 분석")
print("상관 계수 히트맵")
# 필요한 컬럼만 선택 : 나이, 복지비사용액, 만족도점수, 건강지수
correlation_df = df[['나이', '복지비사용액', '만족도점수', '건강지수']]
print(correlation_df.head())

# 상관 계수 계산
corr_matrix = correlation_df.corr()

print("[상관 계수 행렬]")
print(corr_matrix)

# 히트맵 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)

plt.title('나이, 복지비사용액, 만족도점수, 건강지수 간 상관 계수 히트맵', fontsize=13)
plt.tight_layout()
plt.savefig(dataOut + 'c01.correlation_heatmap.png')

# 상삼각(upper triangle)을 마스크로 가리기
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# 히트맵 그리기
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=0.5, cbar=True)

plt.title('나이, 복지비사용액, 만족도점수, 건강지수 간 상관 계수 (하삼각만 표시)', fontsize=13)
plt.tight_layout()
plt.savefig(dataOut + 'c02.correlation_heatmap_lower_triangle.png')

# --- 가장 큰/작은 상관 계수 찾기 ---
# 대각선(자기 자신과의 상관)은 제외
corr_unstacked = corr_matrix.unstack().drop_duplicates()
corr_unstacked = corr_unstacked[corr_unstacked != 1.0]

max_corr = corr_unstacked.idxmax()
min_corr = corr_unstacked.idxmin()

print("\n[상관 계수 분석 결과]")
print(f"가장 큰 상관 계수: {max_corr[0]} ↔ {max_corr[1]} ({corr_unstacked[max_corr]:.2f})")
print(f"가장 작은 상관 계수: {min_corr[0]} ↔ {min_corr[1]} ({corr_unstacked[min_corr]:.2f})")


# CSV 파일로 저장
df.to_csv(dataIn + '고객복지데이터셋Cleaned.csv', index=False, encoding='utf-8-sig')
