import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import io
from plotly.colors import qualitative

# Streamlit UI 설정
st.title("CSV 데이터 분석-Portable.v2.2")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"], key="file_uploader")

# 세션 상태 초기화
if "analyses" not in st.session_state:
    st.session_state.analyses = []

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    st.write("### 업로드된 파일 미리보기 (최대 50,000행)")
    st.dataframe(df.head(50000), key="dataframe_preview")

    if "TIME" not in df.columns:
        st.error("CSV 파일에 'TIME' 컬럼이 없습니다. 올바른 파일을 업로드하세요.", key="time_column_error")
        st.stop()

    y_column_options = [col for col in df.columns if col.startswith("Value")]
    if not y_column_options:
        st.error("Value1, Value2, Value3, Value4 중 하나의 컬럼이 존재하지 않습니다.", key="value_column_error")
        st.stop()

    y_column = st.selectbox("Y축 데이터 선택", y_column_options, index=y_column_options.index("Value3"), key="y_column_select")
    start_time = st.text_input("시작할 TIME 값 입력", value="hh:mm:ss", key="start_time_input")
    n_value = st.number_input("한 번에 선택할 데이터 개수", min_value=1, value=120, key="n_value_input")
    r_value = st.number_input("반복 횟수", min_value=1, value=1, key="r_value_input")
    analysis_name = st.text_input("분석 이름을 입력하세요", value=f"분석_{len(st.session_state.analyses) + 1}", key="analysis_name_input")

    def apply_pattern(y_column, start_time, n_value, r_value):
        if start_time in df["TIME"].astype(str).values:
            start_idx = df[df["TIME"].astype(str) == start_time].index[0]
        else:
            st.error("입력한 TIME 값이 CSV 데이터에 없습니다.", key="start_time_error")
            return None, None

        df["Pattern"] = pd.NA
        for i in range(r_value):
            start = start_idx + (i * n_value)
            end = start + n_value - 1
            df.loc[start:end, "Pattern"] = i + 1

        filtered_df = df.dropna(subset=["Pattern"])
        filtered_df.loc[:, "Pattern"] = filtered_df["Pattern"].astype(int)
        filtered_df = filtered_df.reset_index(drop=True)

        filtered_df["Row_Index"] = filtered_df.groupby("Pattern").cumcount()
        pivot_df = filtered_df.pivot(index="Row_Index", columns="Pattern", values=y_column)
        pivot_df.columns = [f"Pattern_{int(col)}" for col in pivot_df.columns]

        return filtered_df, pivot_df

    if st.button("패턴 적용", key="apply_pattern_button"):
        filtered_df, pivot_df = apply_pattern(y_column, start_time, n_value, r_value)
        if filtered_df is not None:
            st.write("### 패턴이 적용된 데이터")
            st.dataframe(filtered_df[["TIME", y_column, "Pattern"]], key="filtered_data")
            st.write("### 가공된 데이터")
            st.dataframe(pivot_df, key="pivot_data")

            st.session_state.analyses.append({
                "name": analysis_name,
                "y_column": y_column,
                "start_time": start_time,
                "n_value": n_value,
                "r_value": r_value,
                "filtered_df": filtered_df,
                "pivot_df": pivot_df
            })

    st.markdown("---")  # 기본 수평선


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
                
                # 자동 색상 지정: 인덱스에 따른 기본 색상 선택
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
                st.dataframe(analysis["filtered_df"][ ["TIME", analysis["y_column"], "Pattern"] ], key=f"filtered_df_{idx}")
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

            with col2:
                if st.button("삭제", key=f"delete_{idx}"):
                    del st.session_state.analyses[idx]
                    st.rerun()

        st.markdown("---")  # 기본 수평선


        st.write("## 축 및 폰트 설정")
        col1, col2, col3 = st.columns(3)
        with col1:
            x_min = st.number_input("X축 최소값", value=-10, key="x_min")
            x_font_size = st.number_input("X축 폰트 크기", min_value=8, max_value=30, value=20, key="x_font_size")
            x_axis_label = st.text_input("X축 이름", value="TIME", key="x_axis_label")
        with col2:
            x_max = st.number_input("X축 최대값", value=130, key="x_max")
            x_tick_font_size = st.number_input("X축 값 폰트 크기", min_value=8, max_value=30, value=15, key="x_tick_font_size")
        with col3:
            y_min = st.number_input("Y축 최소값", value=200, key="y_min")
            y_max = st.number_input("Y축 최대값", value=2000, key="y_max")
            y_font_size = st.number_input("Y축 폰트 크기", min_value=8, max_value=30, value=20, key="y_font_size")
            y_tick_font_size = st.number_input("Y축 값 폰트 크기", min_value=8, max_value=30, value=15, key="y_tick_font_size")
            y_axis_label = st.text_input("Y축 이름", value="ADC", key="y_axis_label")

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

        # Normalized 비교 그래프 추가
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

                    # 저장할 정규화된 데이터
                    normalized_data[analysis["name"]] = normalized_values

        norm_fig.update_layout(
            title="Normalized Comparison Graph",
            xaxis_title=x_axis_label,
            yaxis_title="Normalized Value",
            xaxis=dict(title_font=dict(size=x_font_size), tickfont=dict(size=x_tick_font_size)),
            yaxis=dict(title_font=dict(size=y_font_size), tickfont=dict(size=y_tick_font_size))
        )
        st.plotly_chart(norm_fig, key="normalized_plotly_chart")

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            average_df = pd.DataFrame()
            normalized_df = pd.DataFrame()

            for analysis in st.session_state.analyses:
                if selected_analyses.get(analysis["name"], False):
                    analysis["pivot_df"].to_excel(writer, sheet_name=analysis["name"])
                    average_df[analysis["name"]] = analysis["pivot_df"]["Average"]
                    if analysis["name"] in normalized_data:
                        normalized_df[analysis["name"]] = normalized_data[analysis["name"]]

            average_df.to_excel(writer, sheet_name="average")
            normalized_df.to_excel(writer, sheet_name="normalized")

        from datetime import datetime
        default_filename = datetime.now().strftime('%y%m%d') + "_analysis.xlsx"
        custom_filename = st.text_input("엑셀 파일 이름을 입력하세요", value=default_filename, key="excel_filename")

        st.download_button(
            label="분석 데이터 다운로드 (Excel)",
            data=excel_buffer.getvalue(),
            file_name=custom_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_button"
        )
