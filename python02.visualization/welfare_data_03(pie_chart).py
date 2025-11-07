import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib.patches import ConnectionPatch

matplotlib.rcParams['font.family'] = ['Malgun Gothic']
matplotlib.rcParams['axes.unicode_minus'] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

dataIn = './../dataIn/'
dataOut = './../dataOut/'

df = pd.read_csv(dataIn + '고객복지데이터셋Cleaned.csv')

###############################################################################
def MakePieChart01(chartdata, title, file_name):
    """
    복지등급 등의 범주형 데이터 비율을 파이 차트로 시각화하는 함수

    Parameters
    ----------
    chartdata : pandas.Series
        index : 범주 이름 (예: 복지등급)
        values : 해당 범주의 개수

    title : str
        그래프 제목

    dataOut : str
        저장 경로 (예: 'D:/MLProject/output/')

    Returns
    -------
    None
    """
    # 레이블(등급명)
    mylabel = chartdata.index

    # 색상 (데이터 개수에 맞게 조정)
    mycolors = ['#66B2FF', '#99FF99', '#FFD700', '#FF9999', '#CC99FF'][:len(chartdata)]

    # 폭발 효과 (가장 큰 값만 살짝 강조)
    max_label = chartdata.idxmax()
    explode = [0.05 if grade == max_label else 0 for grade in chartdata.index]

    # 그래프 그리기
    plt.figure(figsize=(6, 6))
    plt.pie(chartdata,
            labels=mylabel,
            autopct='%1.2f%%',
            startangle=90,
            counterclock=False,
            explode=explode,
            colors=mycolors,
            shadow=True)

    plt.title(title)
    plt.legend(loc='best')
    plt.axis('equal')  # 원형 유지
    plt.tight_layout()

    plt.savefig(dataOut + file_name, dpi=400)
# end def MakePieChart01

print("\n기본형 파이 차트")
welfare_grade_counts = df['복지등급'].value_counts()
print(welfare_grade_counts)
MakePieChart01(welfare_grade_counts, '복지 등급별 인원 비율', 'e01.piechart_01.png')
###############################################################################
def MakePieChart02(chartdata, title, file_name):
    """
    복지등급 등의 범주형 데이터 비율을 시각적으로 표현하고
    퍼센트(%)와 실제 인원수를 함께 표시하는 파이 차트 생성

    Parameters
    ----------
    chartdata : pandas.Series
        index : 범주 이름 (예: 복지등급)
        values : 해당 범주의 개수
    title : str
        그래프 제목

    Returns
    -------
    None
    """

    def getLabelFormat(pct, allvals):
        absolute = int(pct / 100. * np.sum(allvals))
        return "{:.2f}%\n({:d} 명)".format(pct, absolute)

    # 차트 기본 설정
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))

    # 파이 차트 생성
    wedges, texts, autotexts = ax.pie(
        chartdata,
        autopct=lambda pct: getLabelFormat(pct, chartdata),
        textprops=dict(color="w"),
        startangle=90,
        counterclock=False
    )

    # 범례 추가
    ax.legend(
        wedges,
        chartdata.index,
        title="복지등급",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title)

    plt.savefig(dataOut + file_name)
# end def MakePieChart02

print("\n사용자 정의 포지셔닝")
MakePieChart02(welfare_grade_counts, '사용자 정의 라벨 지정', 'e01.piechart_02.png')
###############################################################################
def MakePieChart03(chartdata, title, file_name):
    """
    도우넛 형태의 파이 차트를 생성하는 함수

    Parameters
    ----------
    chartdata : pandas.Series
        인덱스 : 범주 이름 (예: 복지등급)
        값 : 각 범주의 개수
    title : str
        그래프 제목

    """

    # 데이터 준비
    labels = chartdata.index.tolist()
    values = chartdata.values.flatten()

    # 도넛 그래프 기본 설정
    fig, ax = plt.subplots(figsize=(6, 4), subplot_kw=dict(aspect="equal"))
    # wedgeprops : 내부 원 크기
    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.4), startangle=-40)

    # 박스 및 화살표 스타일
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    # 각 조각에 라벨 표시
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))

        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        ax.annotate(
            labels[i],
            xy=(x, y),
            xytext=(1.4 * np.sign(x), 1.2 * y),
            horizontalalignment=horizontalalignment,
            verticalalignment="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.8)
        )

    # 제목 설정
    ax.set_title(title)

    plt.savefig(dataOut + file_name)
# end def MakePieChart03

MakePieChart03(welfare_grade_counts, '도우넛 파이 그래프', 'e01.piechart_03.png')
###############################################################################
def MakePieChart04(cross_df_04, file_name):
    """
    복지등급별 보육지원 여부 중첩 도넛형 파이 차트
    """
    print('복지등급별 보육지원 여부 중첩 파이 그래프를 그려 봅니다.')

    # -------------------------------------------------------------------------
    # Step 1. 데이터 준비
    # -------------------------------------------------------------------------
    chartdf = cross_df_04.values
    outer_labels = cross_df_04.columns
    inner_labels = cross_df_04.index

    outer_sum = chartdf.sum(axis=0)
    total = outer_sum.sum()

    # -------------------------------------------------------------------------
    # Step 2. 색상 설정 (같은 계열의 톤으로 안팎 구분)
    # -------------------------------------------------------------------------
    cmap = plt.get_cmap("tab10")
    outer_colors = [cmap(i) for i in range(len(outer_labels))]
    inner_colors = [(*cmap(i)[:3], 0.6) for i in range(len(outer_labels))]  # 투명도 조정

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))

    # -------------------------------------------------------------------------
    # Step 3. 바깥쪽 도넛 (복지등급별 총합)
    # -------------------------------------------------------------------------
    wedges1, texts1, autotexts1 = ax.pie(
        outer_sum,
        radius=1.0,
        colors=outer_colors,
        labels=outer_labels,
        autopct=lambda p: f"{p:.1f}%",
        pctdistance=0.85,
        startangle=90,  # 고정된 시작점
        counterclock=False,
        wedgeprops=dict(width=0.3, edgecolor='white')
    )

    # -------------------------------------------------------------------------
    # Step 4. 안쪽 도넛 (각 복지등급 내부의 보육지원 분포)
    # -------------------------------------------------------------------------
    start_angle = 90  # 외부와 동일한 기준 유지
    for i, outer_val in enumerate(outer_sum):
        subvals = chartdf[:, i]
        subcolors = [inner_colors[i]] * len(subvals)

        wedges2, texts2, autotexts2 = ax.pie(
            subvals,
            radius=0.7,
            colors=subcolors,
            startangle=start_angle,
            counterclock=False,
            autopct=lambda p: f"{p:.1f}%" if p > 2 else "",
            pctdistance=0.7,
            wedgeprops=dict(width=0.3, edgecolor='white')
        )
        start_angle -= (outer_val / total) * 360  # 다음 섹션 시작각 계산

    # -------------------------------------------------------------------------
    # Step 5. 제목 및 범례
    # -------------------------------------------------------------------------
    ax.set_title("중첩 파이 그래프(문제 있음)", fontsize=13, pad=20)
    ax.legend(wedges1, outer_labels, title="복지등급", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400)
    plt.close(fig)
# end def MakePieChart04

subset_df_04 = df[['복지등급', '보육지원']]
# 교차표 생성
cross_df_04 = pd.crosstab(index=subset_df_04['보육지원'],
                       columns=subset_df_04['복지등급'])
print(cross_df_04)

MakePieChart04(cross_df_04, 'e01_piechart_04.png')
###############################################################################
def MakePieChart05(chartdata, file_name):
    """
    chartdata : pd.DataFrame
        행: 보육지원 (예/아니오), 열: 복지등급
        예)
                가족형   기본  실버   표준  프리미엄
        보육지원
        아니오    55  164  22  255   134
        예       9   25   0   31    14

    file_name : str
        저장할 파일명(경로 포함 가능)
    """

    # 데이터 준비
    labels = chartdata.columns.tolist()          # 복지등급들 (outer labels)
    inner_labels = chartdata.index.tolist()     # 보육지원 ['아니오','예']
    data = chartdata.values                     # shape (2, n_grades)

    # 각 복지등급 총합 (outer pie)
    outer_values = data.sum(axis=0)
    total = outer_values.sum()

    # 선택할 등급: 총합이 가장 큰 등급 (기본값)
    sel_idx = int(np.argmax(outer_values))
    sel_label = labels[sel_idx]
    sel_values = data[:, sel_idx]  # [아니오, 예] for selected grade

    # 시각화 설정
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, aspect='equal')
    ax2 = fig.add_subplot(122)
    fig.subplots_adjust(wspace=0.0)

    # 파이 차트(왼쪽) - outer
    STARTANGLE = 90
    explode = [0.05 if i == sel_idx else 0 for i in range(len(labels))]
    cmap = plt.get_cmap('tab20')
    outer_colors = [cmap(i) for i in range(len(labels))]

    wedges, texts, autotexts = ax1.pie(
        outer_values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=STARTANGLE,
        counterclock=False,
        explode=explode,
        colors=outer_colors
    )
    ax1.set_title('복지등급별 총합')

    # 오른쪽: 선택된 등급의 보육지원 분해(수직 막대)
    # 비율로 표시
    sel_total = sel_values.sum()
    if sel_total == 0:
        bar_heights = [0, 0]
    else:
        bar_heights = sel_values / sel_total

    width = 0.4
    xpos = 0
    bottom = 0
    bar_colors = ['#d9534f', '#5cb85c']  # 아니오(빨), 예(초록)
    labels_bar = inner_labels

    ax2.bar(xpos, bar_heights[0], width, bottom=bottom, color=bar_colors[0])
    ax2.text(xpos, bar_heights[0]/2, f"{sel_values[0]:d}\n({bar_heights[0]*100:.1f}%)",
             ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    bottom += bar_heights[0]

    ax2.bar(xpos, bar_heights[1], width, bottom=bottom, color=bar_colors[1])
    ax2.text(xpos, bottom + bar_heights[1]/2, f"{sel_values[1]:d}\n({bar_heights[1]*100:.1f}%)",
             ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    ax2.set_title(f"'{sel_label}' 등급의 보육지원 분포")
    ax2.axis('off')
    ax2.set_xlim(-2*width, 2*width)

    # ConnectionPatch 연결 - pie의 선택된 wedge와 bar의 상하 연결
    wedge = wedges[sel_idx]
    theta1, theta2 = wedge.theta1, wedge.theta2
    center, r = wedge.center, wedge.r

    # 상단 연결 (bar top)
    bar_height = sum([p.get_height() for p in ax2.patches])
    x_top = r * np.cos(np.deg2rad(theta2)) + center[0]
    y_top = r * np.sin(np.deg2rad(theta2)) + center[1]
    con_top = ConnectionPatch(
        xyA=( - width/2, bar_height ), coordsA=ax2.transData,
        xyB=( x_top, y_top ), coordsB=ax1.transData,
        arrowstyle='-', linewidth=2, color='gray'
    )
    ax2.add_artist(con_top)

    # 하단 연결 (bar bottom)
    x_bot = r * np.cos(np.deg2rad(theta1)) + center[0]
    y_bot = r * np.sin(np.deg2rad(theta1)) + center[1]
    con_bot = ConnectionPatch(
        xyA=( - width/2, 0 ), coordsA=ax2.transData,
        xyB=( x_bot, y_bot ), coordsB=ax1.transData,
        arrowstyle='-', linewidth=2, color='gray'
    )
    ax2.add_artist(con_bot)

    # 범례(외부)
    ax1.legend(wedges, labels, title='복지등급', bbox_to_anchor=(1.05, 0.8), loc='upper left')

    plt.tight_layout()
    plt.savefig(dataOut + file_name, dpi=400, bbox_inches='tight')
    plt.close(fig)

# end def MakePieChart05

MakePieChart05(cross_df_04, 'e01_piechart_05.png')

###############################################################################

# print("\n파이 그래프(Pie Chart)")
# marriage = df['결혼상태']
# print("\n데이터의 고유값:", marriage.unique())
# print("데이터의 고유값 개수:", marriage.nunique())
#
# # 빈도수 출력
# print("\n데이터의 빈도수:")
# # 결혼 상태 데이터 준비
# marriage_counts = marriage.value_counts()
# print(marriage_counts)
#
# # 파이 그래프 그리기
# plt.figure(figsize=(6,6))
# plt.pie(
#     marriage_counts,
#     labels=marriage_counts.index,
#     autopct='%1.2f%%',  # 퍼센트 표시
#     startangle=90,      # 12시 방향에서 시작
#     colors=['#ff9999','#66b3ff','#99ff99','#ffcc99']  # 원하는 색상 지정
# )
# plt.title("결혼 상태 비율")
# plt.savefig('e01.piechart_plot_01.png')
