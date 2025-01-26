import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.title("다중 파일 업로드 - 데이터 비교 및 저장 v1.3")

# Function: 그래프 생성
def plot_graph(x, y, graph_type, label, ax):
    if graph_type == "Line (선형 그래프)":
        ax.plot(x, y, linestyle="-", label=label)
    elif graph_type == "Scatter (산포도)":
        ax.scatter(x, y, label=label)
    elif graph_type == "Dot-Dash (점선 그래프)":
        ax.plot(x, y, linestyle="-.", label=label)
    elif graph_type == "Line + Scatter (선 + 점 그래프)":
        ax.plot(x, y, linestyle="-", marker="o", label=label)

# CSV 파일 다중 업로드
uploaded_files = st.file_uploader("CSV 파일을 여러 개 업로드하세요", type="csv", accept_multiple_files=True)

if uploaded_files:
    file_data = []  # 각 파일별 데이터 저장

    for file in uploaded_files:
        try:
            # pandas로 CSV 파일 읽기
            data = pd.read_csv(file)

            # 파일이 비어 있는지 확인
            if data.empty:
                st.warning(f"파일 {file.name}이 비어 있습니다. 이 파일은 무시됩니다.")
                continue

            # 파일 이름 처리 (.csv 제거)
            file_name_no_ext = file.name.rsplit(".", 1)[0]

            # 파일 이름 출력
            st.success(f"파일 {file_name_no_ext}이 성공적으로 업로드되었습니다!")
            st.write(f"{file_name_no_ext} 데이터 미리보기:")
            st.dataframe(data)

            # 기본 X축과 Y축 설정
            x_axis_default = ' TIME' if ' TIME' in data.columns else data.columns[0]
            y_axis_default = ' Value1' if ' Value1' in data.columns else (data.columns[1] if len(data.columns) > 1 else data.columns[0])

            # Session state 초기화
            if f"start_row_{file_name_no_ext}" not in st.session_state:
                st.session_state[f"start_row_{file_name_no_ext}"] = 1
            if f"end_row_{file_name_no_ext}" not in st.session_state:
                st.session_state[f"end_row_{file_name_no_ext}"] = len(data)

            # 시작 행 설정
            start_row = st.number_input(
                f"{file_name_no_ext} - 시작 행 설정 (1부터 시작)",
                min_value=1,
                max_value=len(data),
                value=st.session_state[f"start_row_{file_name_no_ext}"],
                step=1,
                key=f"start_row_input_{file_name_no_ext}"
            )
            st.session_state[f"start_row_{file_name_no_ext}"] = start_row

            # 종료 행 설정
            end_row = st.number_input(
                f"{file_name_no_ext} - 종료 행 설정 (1부터 시작, 전체 데이터 선택 시 비워두세요)",
                min_value=start_row,
                max_value=len(data),
                value=st.session_state[f"end_row_{file_name_no_ext}"],
                step=1,
                key=f"end_row_input_{file_name_no_ext}"
            )
            st.session_state[f"end_row_{file_name_no_ext}"] = end_row

            # X축과 Y축 선택
            x_axis = st.selectbox(
                f"{file_name_no_ext} - X축 열 선택", 
                options=data.columns, 
                index=list(data.columns).index(x_axis_default),
                key=f"x_{file_name_no_ext}"
            )
            y_axis = st.selectbox(
                f"{file_name_no_ext} - Y축 열 선택", 
                options=data.columns, 
                index=list(data.columns).index(y_axis_default),
                key=f"y_{file_name_no_ext}"
            )

            # 그래프 유형 선택
            graph_type = st.selectbox(
                f"{file_name_no_ext} - 그래프 유형 선택",
                options=["Line (선형 그래프)", "Scatter (산포도)", "Dot-Dash (점선 그래프)", "Line + Scatter (선 + 점 그래프)"],
                key=f"graph_{file_name_no_ext}"
            )

            # 데이터 필터링: 시작 행부터 종료 행까지 데이터 읽기
            filtered_data = data.iloc[start_row - 1:end_row].reset_index(drop=True)
            y_data = filtered_data[y_axis].dropna()
            normalized_y = y_data / y_data.iloc[0]  # 첫 번째 값으로 나누어 정규화

            # 그래프 출력
            st.write(f"**파일 {file_name_no_ext} - {x_axis} vs {y_axis} 그래프 (행 {start_row} ~ {end_row})**")
            fig, ax = plt.subplots()
            plot_graph(filtered_data[x_axis], y_data, graph_type, file_name_no_ext, ax)
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"{file_name_no_ext}: {x_axis} vs {y_axis} (Rows {start_row}~{end_row})")
            ax.legend()
            st.pyplot(fig)

            # 데이터 저장
            file_data.append({
                "file_name": file_name_no_ext,
                "original_y": y_data,
                "normalized_y": normalized_y,
                "graph_type": graph_type
            })

        except Exception as e:
            st.error(f"파일 {file.name} 처리 중 오류가 발생했습니다: {e}")

    # 데이터 비교
    st.write("### 선택된 데이터 비교 그래프")
    included_files = {file["file_name"]: st.checkbox(file["file_name"], value=True) for file in file_data}

    def create_comparison_chart(data_key, title, y_label):
        fig, ax = plt.subplots()
        for file in file_data:
            if included_files[file["file_name"]]:
                y_data = file[data_key]
                plot_graph(range(1, len(y_data) + 1), y_data, file["graph_type"], file["file_name"], ax)
        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.set_ylabel(y_label)
        ax.legend()
        st.pyplot(fig)

    create_comparison_chart("original_y", "Original Data Comparison Graph", "ADC")
    create_comparison_chart("normalized_y", "Normalized Data Comparison Graph", "Normalized ADC")

    # 데이터 저장
    included_file_names = [file["file_name"] for file in file_data if included_files[file["file_name"]]]
    current_date = pd.Timestamp.now().strftime("%Y%m%d")
    default_filename = "_".join(included_file_names) + f"_{current_date}"

    user_filename = st.text_input("엑셀 파일명을 입력하세요 (확장자는 자동 추가됩니다)", value=default_filename)
    if st.button("데이터를 엑셀 파일로 저장"):
        final_filename = user_filename.strip() if user_filename.strip() else default_filename  # 사용자가 입력한 파일명 확인
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            for key, sheet_name in [("original_y", "원본 데이터"), ("normalized_y", "정규화 데이터")]:
                max_length = max(len(file[key]) for file in file_data if included_files[file["file_name"]])
                index_column = pd.Series(range(1, max_length + 1), name="Index")

                combined_data = pd.concat([
                    pd.DataFrame({
                        f"{file['file_name']}": file[key].reindex(range(max_length))
                    }) for file in file_data if included_files[file["file_name"]]
                ], axis=1)
                combined_data.insert(0, "Index", index_column)

                if not combined_data.empty:
                    combined_data.to_excel(writer, sheet_name=sheet_name, index=False)

        st.download_button(
            "엑셀 파일 다운로드",
            output.getvalue(),
            f"{final_filename}.xlsx",  # 최종 파일명 사용
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("CSV 파일을 업로드하면 내용을 확인할 수 있습니다.")
