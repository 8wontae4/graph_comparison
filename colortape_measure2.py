import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Streamlit UI 설정
st.title("CSV 데이터 분석-Portable.v1")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

# 세션 상태 초기화 (추가 분석을 위해 유지)
if "analyses" not in st.session_state:
    st.session_state.analyses = []

if uploaded_file:
    # CSV 파일 로드 및 컬럼명 공백 제거
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    
    # 업로드된 파일 미리보기 (최대 5000행)
    st.write("### 업로드된 파일 미리보기 (최대 5000행)")
    st.dataframe(df.head(5000))
    
    # TIME 컬럼이 있는지 확인
    if "TIME" not in df.columns:
        st.error("CSV 파일에 'TIME' 컬럼이 없습니다. 올바른 파일을 업로드하세요.")
        st.stop()
    
    # Y축 데이터 선택
    y_column_options = [col for col in df.columns if col.startswith("Value")]
    if not y_column_options:
        st.error("Value1, Value2, Value3, Value4 중 하나의 컬럼이 존재하지 않습니다.")
        st.stop()
    
    y_column = st.selectbox("Y축 데이터 선택", y_column_options, index=y_column_options.index("Value3"))  # Value3 값만 사용
    
    # 시작할 TIME 값 입력
    start_time = st.text_input("시작할 TIME 값 입력", value="hh:mm:ss")
    
    # N값 입력 (한 번에 선택할 데이터 개수)
    n_value = st.number_input("한 번에 선택할 데이터 개수", min_value=1, value=120)
    
    # R값 입력 (반복 횟수)
    r_value = st.number_input("반복 횟수", min_value=1, value=1)
    
    # 분석 이름 입력
    analysis_name = st.text_input("분석 이름을 입력하세요", value=f"분석_{len(st.session_state.analyses) + 1}")
    
    def apply_pattern(y_column, start_time, n_value, r_value):
        if start_time in df["TIME"].astype(str).values:
            start_idx = df[df["TIME"].astype(str) == start_time].index[0]
        else:
            st.error("입력한 TIME 값이 CSV 데이터에 없습니다.")
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
        pivot_df["Average"] = pivot_df.mean(axis=1, numeric_only=True)
        
        return filtered_df, pivot_df
    
    if st.button("패턴 적용"):
        filtered_df, pivot_df = apply_pattern(y_column, start_time, n_value, r_value)
        if filtered_df is not None:
            st.write("### 패턴이 적용된 데이터")
            st.dataframe(filtered_df[["TIME", y_column, "Pattern"]])
            st.write("### 가공된 데이터")
            st.dataframe(pivot_df)
            
            st.session_state.analyses.append({
                "name": analysis_name,
                "y_column": y_column,
                "start_time": start_time,
                "n_value": n_value,
                "r_value": r_value,
                "filtered_df": filtered_df,
                "pivot_df": pivot_df
            })
    
    if st.session_state.analyses:
        st.write("## 기존 분석 결과")
        selected_analyses = {}
        for idx, analysis in enumerate(st.session_state.analyses):
            analysis_name = analysis.get("name", f"분석_{idx + 1}")  # 기본값 제공
            selected_analyses[analysis_name] = st.checkbox(analysis_name, value=True)
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.write(f"### 분석 {idx + 1}: {analysis_name}, {analysis['y_column']} 기준")
                st.dataframe(analysis["filtered_df"][["TIME", analysis["y_column"], "Pattern"]])
                st.dataframe(analysis["pivot_df"])
            with col2:
                if st.button("🗑️", key=f"delete_{idx}"):
                    del st.session_state.analyses[idx]
                    st.rerun()
        
        # 엑셀 파일 저장 기능 추가
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            # 'Average' 데이터 저장을 위한 새로운 DataFrame 생성
            average_df = pd.DataFrame()
            for analysis in st.session_state.analyses:
                if selected_analyses.get(analysis["name"], False):
                    analysis["pivot_df"].to_excel(writer, sheet_name=analysis["name"])
                    average_df[analysis["name"]] = analysis["pivot_df"]["Average"]
            
            # 'Average' 데이터 저장
            average_df.to_excel(writer, sheet_name="average")
            for analysis in st.session_state.analyses:
                if selected_analyses.get(analysis["name"], False):
                    analysis["pivot_df"].to_excel(writer, sheet_name=analysis["name"])
            
        
        from datetime import datetime

        # 파일명 지정 (기본값: yymmdd_analysis.xlsx)
        default_filename = datetime.now().strftime('%y%m%d') + "_analysis.xlsx"
        custom_filename = st.text_input("엑셀 파일 이름을 입력하세요", value=default_filename)

        st.download_button(
            label="📥 분석 데이터 다운로드 (Excel)",
            data=excel_buffer.getvalue(),
            file_name=custom_filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # 시각화 추가
        st.write("## 분석 비교 그래프")
        plt.figure(figsize=(6, 5))
        color_palette = sns.color_palette("husl", len(st.session_state.analyses))
        color_map = {analysis["name"]: color_palette[i] for i, analysis in enumerate(st.session_state.analyses)}
        
        for analysis in st.session_state.analyses:
            if selected_analyses.get(analysis["name"], False):
                pivot_df = analysis["pivot_df"]
                for col in pivot_df.columns:
                    if col.startswith("Pattern_") or col == "Average":
                        sns.lineplot(data=pivot_df, x=pivot_df.index, y=col, 
                                     label=f"{analysis['name']} - {col}", 
                                     color=color_map[analysis['name']] if col != 'Average' else 'black', 
                                     linestyle='dashed' if col == 'Average' else 'solid')
        
        plt.xlabel("TIME")
        plt.ylabel("ADC")
        plt.title("Result Comparison Graph")
        plt.legend(loc='upper right', bbox_to_anchor=(1.45, 1.1))
        st.pyplot(plt)

