import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

st.title("다중 파일 업로드 - 데이터 비교 및 저장")

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
            else:
                # 파일 이름 처리 (.csv 제거)
                file_name_no_ext = file.name.replace(".CSV", "").replace(".csv", "")

                # 파일 이름 출력
                st.success(f"파일 {file_name_no_ext}이 성공적으로 업로드되었습니다!")
                st.write(f"{file_name_no_ext} 데이터 미리보기:")
                st.dataframe(data)

                # 시작 행 설정
                start_row = st.number_input(
                    f"{file_name_no_ext} - 시작 행 설정 (1부터 시작)",
                    min_value=1,
                    max_value=len(data),
                    value=1,
                    step=1,
                    key=f"start_row_{file_name_no_ext}"
                )

                # X축과 Y축 선택
                st.write(f"파일 {file_name_no_ext}에서 그래프 축을 선택하세요:")
                x_axis = st.selectbox(f"{file_name_no_ext} - X축 열 선택", options=data.columns, key=f"x_{file_name_no_ext}")
                y_axis = st.selectbox(f"{file_name_no_ext} - Y축 열 선택", options=data.columns, key=f"y_{file_name_no_ext}")

                # 그래프 유형 선택
                graph_type = st.selectbox(
                    f"{file_name_no_ext} - 그래프 유형 선택",
                    options=["Line (선형 그래프)", "Scatter (산포도)", "Dot-Dash (점선 그래프)", "Line + Scatter (선 + 점 그래프)"],
                    key=f"graph_{file_name_no_ext}"
                )

                # 데이터 필터링: 시작 행부터 데이터 읽기
                filtered_data = data.iloc[start_row - 1:].reset_index(drop=True)
                y_data = filtered_data[y_axis].dropna()  # Y축 데이터
                normalized_y = y_data / y_data.iloc[0]  # 첫 번째 값으로 나누어 정규화

                # 그래프 출력 (개별 그래프)
                st.write(f"**파일 {file_name_no_ext} - {x_axis} vs {y_axis} 그래프 (시작 행 {start_row})**")
                fig, ax = plt.subplots()

                # 그래프 유형별로 그래프 생성
                if graph_type == "Line (선형 그래프)":
                    ax.plot(filtered_data[x_axis], y_data, linestyle="-", label=file_name_no_ext)
                elif graph_type == "Scatter (산포도)":
                    ax.scatter(filtered_data[x_axis], y_data, label=file_name_no_ext)
                elif graph_type == "Dot-Dash (점선 그래프)":
                    ax.plot(filtered_data[x_axis], y_data, linestyle="-.", label=file_name_no_ext)
                elif graph_type == "Line + Scatter (선 + 점 그래프)":
                    ax.plot(filtered_data[x_axis], y_data, linestyle="-", marker="o", label=file_name_no_ext)

                ax.set_xlabel(x_axis)
                ax.set_ylabel(y_axis)
                ax.set_title(f"{file_name_no_ext}: {x_axis} vs {y_axis} (Start Row {start_row})")
                ax.legend()
                st.pyplot(fig)

                # 각 파일 데이터를 저장
                file_data.append({
                    "file_name": file_name_no_ext,
                    "original_y": y_data,
                    "normalized_y": normalized_y,
                    "graph_type": graph_type
                })

        except pd.errors.EmptyDataError:
            st.error(f"파일 {file.name}은 빈 파일이거나 잘못된 형식입니다.")
        except pd.errors.ParserError:
            st.error(f"파일 {file.name}을 읽는 중 오류가 발생했습니다. CSV 파일 형식을 확인해주세요.")
        except Exception as e:
            st.error(f"파일 {file.name} 처리 중 오류가 발생했습니다: {e}")

    # 선택적 데이터 비교 체크박스 추가
    st.write("### 선택된 데이터 비교 그래프")
    included_files = {}  # 체크박스 상태 저장

    for file_info in file_data:
        file_name = file_info["file_name"]
        included_files[file_name] = st.checkbox(f"{file_name} 비교에 포함", value=True)

    # 원본 데이터 비교 그래프
    st.write("#### 원본 데이터 비교 그래프")
    fig, ax = plt.subplots()
    for file_info in file_data:
        file_name = file_info["file_name"]
        if included_files[file_name]:  # 체크된 파일만 그래프에 포함
            y_data = file_info["original_y"]
            x_data = range(1, len(y_data) + 1)  # X축: 단순히 인덱스 기반
            graph_type = file_info["graph_type"]

            # 그래프 유형별로 표시
            if graph_type == "Line (선형 그래프)":
                ax.plot(x_data, y_data, linestyle="-", label=file_name)
            elif graph_type == "Scatter (산포도)":
                ax.scatter(x_data, y_data, label=file_name)
            elif graph_type == "Dot-Dash (점선 그래프)":
                ax.plot(x_data, y_data, linestyle="-.", label=file_name)
            elif graph_type == "Line + Scatter (선 + 점 그래프)":
                ax.plot(x_data, y_data, linestyle="-", marker="o", label=file_name)

    ax.set_xlabel("TIME (based length of data)")
    ax.set_ylabel("ADC")
    ax.set_title("Original Data Comparison")
    ax.legend()  # 범례 추가
    st.pyplot(fig)

    # 정규화된 데이터 비교 그래프
    st.write("#### 정규화된 데이터 비교 그래프")
    fig, ax = plt.subplots()
    for file_info in file_data:
        file_name = file_info["file_name"]
        if included_files[file_name]:  # 체크된 파일만 그래프에 포함
            normalized_y = file_info["normalized_y"]
            x_data = range(1, len(normalized_y) + 1)  # X축: 단순히 인덱스 기반
            graph_type = file_info["graph_type"]

            # 그래프 유형별로 표시
            if graph_type == "Line (선형 그래프)":
                ax.plot(x_data, normalized_y, linestyle="-", label=file_name)
            elif graph_type == "Scatter (산포도)":
                ax.scatter(x_data, normalized_y, label=file_name)
            elif graph_type == "Dot-Dash (점선 그래프)":
                ax.plot(x_data, normalized_y, linestyle="-.", label=file_name)
            elif graph_type == "Line + Scatter (선 + 점 그래프)":
                ax.plot(x_data, normalized_y, linestyle="-", marker="o", label=file_name)

    ax.set_xlabel("TIME (based length of data)")
    ax.set_ylabel("Normalized ADC")
    ax.set_title("Normalized Data Comparison")
    ax.legend()  # 범례 추가
    st.pyplot(fig)

    # 엑셀 데이터 저장
    if st.button("데이터를 엑셀 파일로 저장"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            # 원본 데이터 저장
            original_data = pd.DataFrame()
            for file_info in file_data:
                file_name = file_info["file_name"]
                if included_files[file_name]:  # 체크된 파일만 저장
                    temp_df = pd.DataFrame({
                        "Index": range(1, len(file_info["original_y"]) + 1),
                        f"{file_name}_Y값": file_info["original_y"].values
                    })
                    original_data = pd.concat([original_data, temp_df], axis=1)

            if not original_data.empty:
                original_data.to_excel(writer, sheet_name="원본 데이터 비교", index=False)

            # 정규화된 데이터 저장
            normalized_data = pd.DataFrame()
            for file_info in file_data:
                file_name = file_info["file_name"]
                if included_files[file_name]:  # 체크된 파일만 저장
                    temp_df = pd.DataFrame({
                        "Index": range(1, len(file_info["normalized_y"]) + 1),
                        f"{file_name}_정규화_Y값": file_info["normalized_y"].values
                    })
                    normalized_data = pd.concat([normalized_data, temp_df], axis=1)

            if not normalized_data.empty:
                normalized_data.to_excel(writer, sheet_name="정규화 데이터 비교", index=False)

        # 다운로드 버튼 생성
        output.seek(0)
        st.download_button(
            label="엑셀 파일 다운로드",
            data=output,
            file_name="비교_데이터.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("CSV 파일을 업로드하면 내용을 확인할 수 있습니다.")
