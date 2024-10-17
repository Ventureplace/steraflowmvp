import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI
import utils

# Initialize session
utils.init()

def get_ai_response(prompt, data):
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data analysis assistant."},
            {"role": "user", "content": f"Given the following data:\n\n{data.to_string()}\n\nUser question: {prompt}"}
        ]
    )
    return response.choices[0].message.content

def modify_data_with_ai(prompt, data):
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data modification assistant. Provide Python code to modify the data based on the user's request."},
            {"role": "user", "content": f"Given the following data:\n\n{data.to_string()}\n\nUser request: {prompt}\n\nProvide Python code to modify the data:"}
        ]
    )
    return response.choices[0].message.content

def show(project_name):
    st.header(f"Dashboard Summary for Project: {project_name}")

    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found.")
        return

    project_data = st.session_state.projects[project_name]

    if 'cleaned_data' not in project_data:
        st.warning(f"No cleaned data available for {project_name}. Please clean the data first.")
        return

    data = project_data['cleaned_data']

    # Check if data is a DataFrame, if not, try to convert it
    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except:
            st.error(f"Unable to process the data for {project_name}. Please ensure the data is in the correct format.")
            return

    col1, col2 = st.columns(2)
    col1.metric("Rows", data.shape[0])
    col2.metric("Columns", data.shape[1])

    st.subheader("Data Sample")
    st.dataframe(data.head())

    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Type': data.dtypes,
        'Non-Null Count': data.notnull().sum(),
        'Null Count': data.isnull().sum(),
        'Unique Values': data.nunique()
    })
    st.dataframe(col_info)

    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if not num_cols.empty:
        st.subheader("Numerical Columns Summary")
        st.dataframe(data[num_cols].describe())

        if len(num_cols) > 1:
            st.subheader("Correlation Heatmap")
            fig = px.imshow(data[num_cols].corr(), color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig.update_layout(title='Correlation Heatmap')
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        st.subheader("Categorical Columns Summary")
        for col in cat_cols:
            st.write(f"{col} - Top 5 Categories")
            top_5 = data[col].value_counts().nlargest(5)
            fig = px.bar(x=top_5.index, y=top_5.values)
            fig.update_layout(title=f'Top 5 Categories in {col}', xaxis_title=col, yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

    st.subheader("Data Processing Steps")
    steps = []
    if 'data' in project_data:
        steps.append("✅ Data Uploaded")
    if 'cleaned_data' in project_data:
        steps.append("✅ Data Cleaned")
    if 'engineered_data' in project_data:
        steps.append("✅ Features Engineered")
    for step in steps:
        st.write(step)

    st.subheader("Chat with AI about the Data")
    user_input = st.text_input("Ask a question about the data:")
    if user_input:
        ai_response = get_ai_response(user_input, data)
        st.write("AI Response:", ai_response)

    st.subheader("Modify Data with AI")
    modification_prompt = st.text_input("Describe how you want to modify the data:")
    if modification_prompt:
        modification_code = modify_data_with_ai(modification_prompt, data)
        st.code(modification_code, language="python")
        if st.button("Apply Modification"):
            try:
                exec(modification_code)
                st.session_state.projects[project_name]['cleaned_data'] = data
                st.success("Data modified successfully!")
                st.dataframe(data)
            except Exception as e:
                st.error(f"Error modifying data: {str(e)}")

    # Download full report as CSV
    csv = data.to_csv(index=False)
    st.download_button(
        label=f"Download {project_name} Data as CSV",
        data=csv,
        file_name=f"{project_name}_cleaned_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    show(st.session_state.current_project)