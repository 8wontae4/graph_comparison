import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from plotly.colors import qualitative
import re
from datetime import datetime

# Streamlit UI 설정
st.title("Sloat 데이터 분석기 v1.0")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일 업로드", type=None)

if uploaded_file:
    if not uploaded_file.name.lower().endswith(".csv"):
        st.error("❌ .csv 확장자가 붙은 파일만 업로드할 수 있습니다.")
        st.stop()

    try:
        file_bytes = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        st.session_state["df"] = df
        st.success("✅ CSV 파일 로드 성공")
    except Exception as e:
        st.session_state.pop("df", None)
        st.error("CSV 파일을 읽는 중 오류 발생")
        st.text(str(e))

# 세션 상태 초기화
if "analyses" not in st.session_state:
    st.session_state.analyses = []

# CSV 업로드 확인
if "df" not in st.session_state:
    st.warning("📄 먼저 CSV 파일을 업로드해주세요.")
    st.stop()

# 세션에서 df 불러오기
df = st.session_state.df

st.write("### 업로드된 파일 미리보기 (최대 50,000행)")
st.dataframe(df.head(50000), key="dataframe_preview")

# 필수 컬럼 확인
if "Time" not in df.columns:
    st.error("CSV 파일에 'Time' 컬럼이 없습니다. 올바른 파일을 업로드하세요.")
    st.stop()

if "State" not in df.columns:
    st.error("CSV 파일에 'State' 컬럼이 없습니다. 올바른 파일을 업로드하세요.")
    st.stop()

# Sloat 컬럼 감지
y_column_options = [col for col in df.columns if col.startswith("Sloat")]
if not y_column_options:
    st.error("Sloat1~8 중 하나의 컬럼이 존재하지 않습니다.")
    st.stop()

# 분석 설정
y_column = st.selectbox("Y축 데이터 선택", y_column_options, index=0, key="y_column_select")
start_time = st.text_input("시작할 Time 값 입력", value="M/D/YYYY HH:MM:SS")
n_value = st.number_input("한 번에 선택할 데이터 개수", min_value=1, value=100, key="n_value_input")
r_value = st.number_input("반복 횟수", min_value=1, value=1, key="r_value_input")
analysis_name = st.text_input("분석 이름을 입력하세요", value=f"분석_{len(st.session_state.analyses) + 1}")


def find_advance_indices(df, start_idx, r_value):
    """시작 인덱스 이후의 Advance 위치들을 찾음"""
    advance_indices = []
    current_idx = start_idx

    while len(advance_indices) < r_value and current_idx < len(df):
        if str(df.loc[current_idx, "State"]).strip() == "Advance":
            advance_indices.append(current_idx)
        current_idx += 1

    return advance_indices


def apply_pattern(y_column, start_time, n_value, r_value):
    """패턴 적용 및 데이터 처리"""
    # 시작 시간 찾기
    if start_time not in df["Time"].astype(str).values:
        st.error("입력한 Time 값이 CSV 데이터에 없습니다.")
        return None, None, None

    start_idx = df[df["Time"].astype(str) == start_time].index[0]

    # Advance 위치 찾기
    advance_indices = find_advance_indices(df, start_idx, r_value)

    if len(advance_indices) < r_value:
        st.warning(f"요청한 반복 횟수({r_value})보다 적은 Advance가 발견되었습니다: {len(advance_indices)}개")
        if len(advance_indices) == 0:
            st.error("Advance를 찾을 수 없습니다.")
            return None, None, None

    # 패턴별 데이터 수집
    all_pattern_data = []
    tv_data = []

    for pattern_num, adv_idx in enumerate(advance_indices, 1):
        # Back 값: Advance 이후 11번째 행 (Advance 미포함)
        back_idx = adv_idx + 11
        if back_idx >= len(df):
            st.warning(f"패턴 {pattern_num}: Back 값 인덱스가 데이터 범위를 벗어났습니다.")
            continue

        back_value = df.loc[back_idx, y_column]

        # 분석 데이터: Advance 이후 12번째부터 (Back 다음 행부터)
        analysis_start_idx = adv_idx + 12

        # n_value는 Advance부터 카운트, 실제 분석은 제외 후 남은 행
        # Advance부터 n_value개 = adv_idx ~ adv_idx + n_value - 1
        # 제외: Advance(1) + 10개 + Back(1) = 12개
        # 분석: adv_idx + 12 ~ adv_idx + n_value - 1
        analysis_end_idx = adv_idx + n_value - 1

        if analysis_end_idx >= len(df):
            analysis_end_idx = len(df) - 1

        # 분석 데이터 추출
        for idx in range(analysis_start_idx, analysis_end_idx + 1):
            if idx < len(df):
                row_data = {
                    "Time": df.loc[idx, "Time"],
                    "Pattern": pattern_num,
                    y_column: df.loc[idx, y_column],
                    "Row_Index": idx - analysis_start_idx
                }
                all_pattern_data.append(row_data)

                # T-V 계산
                sloat_value = df.loc[idx, y_column]
                if back_value != 0:
                    tv_value = (back_value - sloat_value) / back_value * 10000
                else:
                    tv_value = 0

                tv_data.append({
                    "Pattern": pattern_num,
                    "Time": df.loc[idx, "Time"],
                    "Back": back_value,
                    y_column: sloat_value,
                    "T-V": tv_value
                })

    if not all_pattern_data:
        st.error("분석할 데이터가 없습니다.")
        return None, None, None

    # DataFrame 생성
    filtered_df = pd.DataFrame(all_pattern_data)
    tv_df = pd.DataFrame(tv_data)

    # 피벗 테이블 생성
    pivot_df = filtered_df.pivot(index="Row_Index", columns="Pattern", values=y_column)
    pivot_df.columns = [f"Pattern_{int(col)}" for col in pivot_df.columns]

    return filtered_df, pivot_df, tv_df


if st.button("패턴 적용", key="apply_pattern_button"):
    filtered_df, pivot_df, tv_df = apply_pattern(y_column, start_time, n_value, r_value)
    if filtered_df is not None:
        st.session_state.analyses.append({
            "name": analysis_name,
            "y_column": y_column,
            "start_time": start_time,
            "n_value": n_value,
            "r_value": r_value,
            "filtered_df": filtered_df,
            "pivot_df": pivot_df,
            "tv_df": tv_df
        })
        st.success(f"✅ '{analysis_name}' 분석 완료 - 아래 '기존 분석 결과'에서 확인하세요.")

st.markdown("---")


if st.session_state.analyses:
    st.write("## 기존 분석 결과")
    selected_analyses = {}
    graph_settings = {}

    # Plotly 기본 색상 팔레트
    default_colors = qualitative.Plotly

    for idx, analysis in enumerate(st.session_state.analyses):
        analysis_name = analysis.get("name", f"분석_{idx + 1}")
        selected_analyses[analysis_name] = st.checkbox(analysis_name, value=True, key=f"select_analysis_{idx}")

        with st.expander(f"{analysis_name} 그래프 설정"):
            graph_type = st.selectbox(
                f"{analysis_name} 그래프 유형 선택",
                ["Scatter+line", "line", "dash", "dash-dot"],
                key=f"type_{idx}"
            )

            default_color = default_colors[idx % len(default_colors)]
            color = st.color_picker(f"{analysis_name} 색상 선택", default_color, key=f"color_{idx}")

            marker_symbol = st.selectbox(
                f"{analysis_name} Scatter 도형 선택",
                ["circle", "square", "diamond", "triangle-up", "triangle-down", "star", "hexagon", "pentagon",
                "circle-open", "square-dot", "triangle-up-open", "triangle-down-open", "star-open",
                "hexagon-open", "pentagon-open"],
                key=f"marker_{idx}") if graph_type == "Scatter+line" else None

            marker_size = st.slider(
                f"{analysis_name} Scatter 도형 크기",
                min_value=1, max_value=20, value=7, key=f"size_{idx}") if graph_type == "Scatter+line" else None

            graph_settings[analysis_name] = {
                "type": graph_type,
                "color": color,
                "marker": marker_symbol,
                "size": marker_size
            }

        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.write(f"### 분석 {idx + 1}: {analysis_name}, {analysis['y_column']} 기준")
            st.dataframe(analysis["filtered_df"][["Time", analysis["y_column"], "Pattern"]], key=f"filtered_df_{idx}")
            st.write("### 가공된 데이터")

            # 선택된 패턴만으로 Average 재계산
            available_patterns = [col for col in analysis["pivot_df"].columns if col.startswith("Pattern_")]
            selected_patterns = [pattern for pattern in available_patterns if st.checkbox(f"{analysis_name} - {pattern} 포함", value=True, key=f"pattern_select_{idx}_{pattern}")]

            if selected_patterns:
                analysis["pivot_df"]["Average"] = analysis["pivot_df"][selected_patterns].mean(axis=1)
            else:
                analysis["pivot_df"]["Average"] = pd.NA

            st.dataframe(analysis["pivot_df"], key=f"pivot_df_{idx}")

            pivot_df = analysis["pivot_df"].drop(columns=["Average"], errors='ignore')
            first_value_mean = pivot_df.iloc[0].mean()
            del_values = pivot_df.iloc[0] - pivot_df.iloc[-1]
            del_value_mean = del_values.mean()
            del_value_max = del_values.max()
            del_value_min = del_values.min()
            del_value_std = del_values.std()

            # 선택된 패턴으로 계산된 값
            if selected_patterns:
                selected_df = pivot_df[selected_patterns]
                selected_first_value_mean = selected_df.iloc[0].mean()
                selected_del_values = selected_df.iloc[0] - selected_df.iloc[-1]
                selected_del_value_mean = selected_del_values.mean()
                selected_del_value_max = selected_del_values.max()
                selected_del_value_min = selected_del_values.min()
                selected_del_value_std = selected_del_values.std()
            else:
                selected_first_value_mean = selected_del_value_mean = selected_del_value_max = selected_del_value_min = selected_del_value_std = "-"

            stats_df = pd.DataFrame({
                "통계 항목": [
                    "첫 번째 값(Background)의 평균",
                    "Del Value의 평균",
                    "Del Value_MAX",
                    "Del Value_MIN",
                    "Del Value의 표준편차"
                ],
                "전체 값": [
                    first_value_mean,
                    del_value_mean,
                    del_value_max,
                    del_value_min,
                    del_value_std
                ],
                "선택된 패턴 값": [
                    selected_first_value_mean,
                    selected_del_value_mean,
                    selected_del_value_max,
                    selected_del_value_min,
                    selected_del_value_std
                ]
            })

            st.write(f"### 통계 계산값 - {analysis_name}")
            st.dataframe(stats_df, key=f"stats_df_{idx}")

            # T-V 데이터 표시
            if "tv_df" in analysis and analysis["tv_df"] is not None:
                st.write(f"### T-V 데이터 - {analysis_name}")
                st.dataframe(analysis["tv_df"], key=f"tv_df_{idx}")

        with col2:
            if st.button("삭제", key=f"delete_{idx}"):
                del st.session_state.analyses[idx]
                st.rerun()

    st.markdown("---")

    # 축 및 폰트 설정
    st.write("## 축 및 폰트 설정")
    col1, col2, col3 = st.columns(3)
    with col1:
        x_min = st.number_input("X축 최소값", value=0, key="x_min")
        x_font_size = st.number_input("X축 폰트 크기", min_value=8, max_value=30, value=20, key="x_font_size")
        x_axis_label = st.text_input("X축 이름", value="Row Index")
    with col2:
        x_max = st.number_input("X축 최대값", value=100, key="x_max")
        x_tick_font_size = st.number_input("X축 값 폰트 크기", min_value=8, max_value=30, value=15, key="x_tick_font_size")
    with col3:
        y_min = st.number_input("Y축 최소값", value=30000, key="y_min")
        y_max = st.number_input("Y축 최대값", value=45000, key="y_max")
        y_font_size = st.number_input("Y축 폰트 크기", min_value=8, max_value=30, value=20, key="y_font_size")
        y_tick_font_size = st.number_input("Y축 값 폰트 크기", min_value=8, max_value=30, value=15, key="y_tick_font_size")
        y_axis_label = st.text_input("Y축 이름", value="ADC")

    # 원본 비교 그래프
    st.write("## 분석 비교 그래프")
    fig = go.Figure()

    normalized_data = {}

    for analysis in st.session_state.analyses:
        if selected_analyses.get(analysis["name"], False):
            pivot_df = analysis["pivot_df"]
            settings = graph_settings[analysis["name"]]

            if "Average" in pivot_df.columns:
                if settings["type"] == "line":
                    fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df["Average"], mode="lines", name=f"{analysis['name']} - Average", line=dict(color=settings["color"])))
                elif settings["type"] == "Scatter+line":
                    fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df["Average"], mode="lines+markers", name=f"{analysis['name']} - Average", line=dict(color=settings["color"]), marker=dict(symbol=settings["marker"], size=settings["size"], color=settings["color"])))
                elif settings["type"] == "dash":
                    fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df["Average"], mode="lines", name=f"{analysis['name']} - Average", line=dict(color=settings["color"], dash="dash")))
                elif settings["type"] == "dash-dot":
                    fig.add_trace(go.Scatter(x=pivot_df.index, y=pivot_df["Average"], mode="lines", name=f"{analysis['name']} - Average", line=dict(color=settings["color"], dash="dashdot")))

    fig.update_layout(
        title="Result Comparison Graph",
        xaxis_title=x_axis_label,
        yaxis_title=y_axis_label,
        xaxis=dict(range=[x_min, x_max], title_font=dict(size=x_font_size), tickfont=dict(size=x_tick_font_size)),
        yaxis=dict(range=[y_min, y_max], title_font=dict(size=y_font_size), tickfont=dict(size=y_tick_font_size))
    )
    st.plotly_chart(fig, key="plotly_chart")

    # Normalized 비교 그래프
    st.write("## Normalized 비교 그래프")
    norm_fig = go.Figure()

    for analysis in st.session_state.analyses:
        if selected_analyses.get(analysis["name"], False):
            pivot_df = analysis["pivot_df"]
            settings = graph_settings[analysis["name"]]

            if "Average" in pivot_df.columns and not pivot_df["Average"].isna().any():
                first_value = pivot_df["Average"].iloc[0]
                normalized_values = pivot_df["Average"] / first_value if first_value != 0 else pivot_df["Average"]

                norm_fig.add_trace(go.Scatter(
                    x=pivot_df.index,
                    y=normalized_values,
                    mode="lines",
                    name=f"{analysis['name']} - Normalized",
                    line=dict(color=settings["color"])
                ))

                normalized_data[analysis["name"]] = normalized_values

    norm_fig.update_layout(
        title="Normalized Comparison Graph",
        xaxis_title=x_axis_label,
        yaxis_title="Normalized Value",
        xaxis=dict(title_font=dict(size=x_font_size), tickfont=dict(size=x_tick_font_size)),
        yaxis=dict(title_font=dict(size=y_font_size), tickfont=dict(size=y_tick_font_size))
    )
    st.plotly_chart(norm_fig, key="normalized_plotly_chart")

    # T-V 비교 그래프
    st.write("## T-V 비교 그래프")
    tv_fig = go.Figure()

    for analysis in st.session_state.analyses:
        if selected_analyses.get(analysis["name"], False):
            tv_df = analysis.get("tv_df")
            settings = graph_settings[analysis["name"]]

            if tv_df is not None and not tv_df.empty:
                # 패턴별로 평균 T-V 계산
                tv_avg = tv_df.groupby(tv_df.index % len(tv_df[tv_df["Pattern"] == 1]))["T-V"].mean()

                tv_fig.add_trace(go.Scatter(
                    x=list(range(len(tv_avg))),
                    y=tv_avg,
                    mode="lines",
                    name=f"{analysis['name']} - T-V",
                    line=dict(color=settings["color"])
                ))

    tv_fig.update_layout(
        title="T-V Comparison Graph",
        xaxis_title="Time Index",
        yaxis_title="T-V Value",
        xaxis=dict(title_font=dict(size=x_font_size), tickfont=dict(size=x_tick_font_size)),
        yaxis=dict(title_font=dict(size=y_font_size), tickfont=dict(size=y_tick_font_size))
    )
    st.plotly_chart(tv_fig, key="tv_plotly_chart")

    # Excel 다운로드
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        average_df = pd.DataFrame()
        normalized_df = pd.DataFrame()
        tv_combined_df = pd.DataFrame()

        for analysis in st.session_state.analyses:
            if selected_analyses.get(analysis["name"], False):
                original_sheet_name = analysis["name"]
                sanitized_name = re.sub(r'[\\/*?:\[\]]', '_', original_sheet_name)
                final_sheet_name = sanitized_name[:31]

                analysis["pivot_df"].to_excel(writer, sheet_name=final_sheet_name)

                average_df[analysis["name"]] = analysis["pivot_df"]["Average"]
                if analysis["name"] in normalized_data:
                    normalized_df[analysis["name"]] = normalized_data[analysis["name"]]

                # T-V 데이터 추가
                if "tv_df" in analysis and analysis["tv_df"] is not None:
                    tv_sheet_name = f"TV_{sanitized_name}"[:31]
                    analysis["tv_df"].to_excel(writer, sheet_name=tv_sheet_name, index=False)

        average_df.to_excel(writer, sheet_name="average")
        normalized_df.to_excel(writer, sheet_name="normalized")

    default_filename = datetime.now().strftime('%y%m%d') + "_sloat_analysis.xlsx"
    custom_filename = st.text_input("엑셀 파일 이름을 입력하세요", value=default_filename)

    st.download_button(
        label="분석 데이터 다운로드 (Excel)",
        data=excel_buffer.getvalue(),
        file_name=custom_filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_button"
    )
