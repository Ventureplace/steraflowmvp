import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI
import utils
import base64
from scipy import stats
import io

# Initialize session
utils.init()

def get_openai_client():
    return OpenAI(api_key=st.secrets["openai_api_key"])

def get_ai_response(prompt, data, charts):
    client = get_openai_client()
    try:
        context = f"Data summary:\n{data.describe().to_string()}\n\n"
        context += f"Charts:\n{charts}\n\n"
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful data analysis assistant with expertise in interpreting charts and data insights."},
                {"role": "user", "content": f"Given the following data and charts:\n\n{context}\n\nUser question: {prompt}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def modify_data_with_ai(prompt, data):
    client = get_openai_client()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful data modification assistant. Provide Python code to modify the data based on the user's request."},
            {"role": "user", "content": f"Given the following data:\n\n{data.to_string()}\n\nUser request: {prompt}\n\nProvide Python code to modify the data:"}
        ]
    )
    return response.choices[0].message.content

def create_charts(data, source_name):
    charts = []
    
    # Numerical columns summary
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if not num_cols.empty:
        st.subheader(f"Numerical Columns Summary for {source_name}")
        st.dataframe(data[num_cols].describe())

        if len(num_cols) > 1:
            st.subheader(f"Correlation Heatmap for {source_name}")
            fig = px.imshow(data[num_cols].corr(), color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig.update_layout(title=f'Correlation Heatmap - {source_name}')
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            charts.append(f"Correlation Heatmap - {source_name}")

            # Add insights on positive and negative correlations
            corr_matrix = data[num_cols].corr()
            st.subheader(f"Correlation Insights for {source_name}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Top Positive Correlations:")
                top_pos = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
                top_pos = top_pos[(top_pos != 1.0) & (top_pos > 0)].head(5)
                st.write(top_pos)
            with col2:
                st.write("Top Negative Correlations:")
                top_neg = corr_matrix.unstack().sort_values().drop_duplicates()
                top_neg = top_neg[top_neg < 0].head(5)
                st.write(top_neg)
            charts.append(f"Top Positive/Negative Correlations - {source_name}")

        # Scatter plot with regression line
        st.subheader(f"Scatter Plot with Regression for {source_name}")
        x_col = st.selectbox("Select X-axis", num_cols, key=f"{source_name}_scatter_x")
        y_col = st.selectbox("Select Y-axis", num_cols, key=f"{source_name}_scatter_y")
        regression_type = st.selectbox("Select Regression Type", ["Linear", "Polynomial", "Exponential"], key=f"{source_name}_regression_type")
        
        fig = px.scatter(data, x=x_col, y=y_col, trendline=regression_type.lower())
        fig.update_layout(title=f'{regression_type} Regression: {y_col} vs {x_col} - {source_name}')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        charts.append(f"{regression_type} Regression: {y_col} vs {x_col} - {source_name}")

        # Distribution plot
        st.subheader(f"Distribution Plot for {source_name}")
        dist_col = st.selectbox("Select Column for Distribution", num_cols, key=f"{source_name}_dist_col")
        fig = px.histogram(data, x=dist_col, marginal="box")
        fig.update_layout(title=f'Distribution of {dist_col} - {source_name}')
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        charts.append(f"Distribution of {dist_col} - {source_name}")

    # Categorical columns summary
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        st.subheader(f"Categorical Columns Summary for {source_name}")
        for col in cat_cols:
            st.write(f"{col} - Top 5 Categories")
            top_5 = data[col].value_counts().nlargest(5)
            fig = px.bar(x=top_5.index, y=top_5.values)
            fig.update_layout(title=f'Top 5 Categories in {col} - {source_name}', xaxis_title=col, yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            charts.append(f"Top 5 Categories in {col} - {source_name}")

        # Categorical vs Numerical
        if not num_cols.empty:
            st.subheader(f"Categorical vs Numerical for {source_name}")
            cat_col = st.selectbox("Select Categorical Column", cat_cols, key=f"{source_name}_cat_col")
            num_col = st.selectbox("Select Numerical Column", num_cols, key=f"{source_name}_num_col")
            fig = px.box(data, x=cat_col, y=num_col)
            fig.update_layout(title=f'{num_col} by {cat_col} - {source_name}')
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            charts.append(f"{num_col} by {cat_col} - {source_name}")

    return charts

def harmonize_data(data_sources):
    # This is a simple example of data harmonization
    # You may need to implement more sophisticated harmonization based on your specific data
    harmonized_data = pd.concat(data_sources.values(), axis=1)
    return harmonized_data

def generate_data_report(data, charts):
    buffer = io.StringIO()
    buffer.write(f"Data Report\n\n")
    buffer.write(f"Data Shape: {data.shape}\n\n")
    buffer.write(f"Data Types:\n{data.dtypes.to_string()}\n\n")
    buffer.write(f"Data Summary:\n{data.describe().to_string()}\n\n")
    buffer.write(f"Charts Generated:\n{', '.join(charts)}\n\n")
    buffer.write(f"Correlation Matrix:\n{data.corr().to_string()}\n\n")
    
    for col in data.columns:
        buffer.write(f"Column: {col}\n")
        buffer.write(f"Unique Values: {data[col].nunique()}\n")
        buffer.write(f"Top 5 Values:\n{data[col].value_counts().nlargest(5).to_string()}\n\n")

    return buffer.getvalue()

def show(project_name):
    st.header(f"Dashboard Summary for Project: {project_name}")

    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found.")
        return

    project_data = st.session_state.projects[project_name]

    if 'cleaned_data_sources' not in project_data or not project_data['cleaned_data_sources']:
        st.warning(f"No cleaned data available for {project_name}. Please clean the data first.")
        return

    # Create tabs for each data source and harmonized data
    data_sources = project_data['cleaned_data_sources']
    tab_names = list(data_sources.keys()) + ["Harmonized Data"]
    tabs = st.tabs(tab_names)

    all_charts = []

    for i, (source, data) in enumerate(data_sources.items()):
        with tabs[i]:
            st.subheader(f"Data from {source}")
            
            # Check if data is a DataFrame, if not, try to convert it
            if not isinstance(data, pd.DataFrame):
                try:
                    data = pd.DataFrame(data)
                except:
                    st.error(f"Unable to process the data for {source}. Please ensure the data is in the correct format.")
                    continue

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

            # Create charts for this data source
            source_charts = create_charts(data, source)
            all_charts.extend(source_charts)

            # Download data as CSV
            csv = data.to_csv(index=False)
            st.download_button(
                label=f"Download {source} Data as CSV",
                data=csv,
                file_name=f"{project_name}_{source}_cleaned_data.csv",
                mime="text/csv",
            )

            # Download data report
            report = generate_data_report(data, source_charts)
            st.download_button(
                label=f"Download {source} Data Report",
                data=report,
                file_name=f"{project_name}_{source}_data_report.txt",
                mime="text/plain",
            )

    # Harmonized Data tab
    with tabs[-1]:
        st.subheader("Harmonized Data")
        harmonized_data = harmonize_data(data_sources)
        
        col1, col2 = st.columns(2)
        col1.metric("Rows", harmonized_data.shape[0])
        col2.metric("Columns", harmonized_data.shape[1])

        st.subheader("Harmonized Data Sample")
        st.dataframe(harmonized_data.head())

        # Create charts for harmonized data
        harmonized_charts = create_charts(harmonized_data, "Harmonized Data")
        all_charts.extend(harmonized_charts)

        # Download harmonized data as CSV
        csv = harmonized_data.to_csv(index=False)
        st.download_button(
            label="Download Harmonized Data as CSV",
            data=csv,
            file_name=f"{project_name}_harmonized_data.csv",
            mime="text/csv",
        )

        # Download harmonized data report
        report = generate_data_report(harmonized_data, harmonized_charts)
        st.download_button(
            label="Download Harmonized Data Report",
            data=report,
            file_name=f"{project_name}_harmonized_data_report.txt",
            mime="text/plain",
        )

    st.subheader("Data Processing Steps")
    steps = []
    if 'data_sources' in project_data:
        steps.append("✅ Data Uploaded")
    if 'cleaned_data_sources' in project_data:
        steps.append("✅ Data Cleaned")
    if 'engineered_data' in project_data:
        steps.append("✅ Features Engineered")
    for step in steps:
        st.write(step)

    # Chat functionality
    if 'chat_open' not in st.session_state:
        st.session_state.chat_open = False

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Floating chat button
    chat_button = st.button("💬 Chat with Your Data", key="floating_chat_button")

    if chat_button:
        st.session_state.chat_open = not st.session_state.chat_open

    # Floating chat interface
    if st.session_state.chat_open:
        chat_container = st.container()
        with chat_container:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Chat input
            if prompt := st.chat_input("Ask about your data and insights..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = get_ai_response(prompt, harmonized_data, all_charts)
                    message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    show(st.session_state.current_project)
