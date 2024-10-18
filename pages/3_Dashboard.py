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

def get_ai_response(prompt, data, charts, tab_name):
    client = get_openai_client()
    try:
        context = f"Data summary:\n{data.describe().to_string()}\n\n"
        context += f"Data columns:\n{', '.join(data.columns)}\n\n"
        context += f"Data sample:\n{data.head().to_string()}\n\n"
        
        if tab_name == "Harmonized Data":
            system_content = "You are a data analysis assistant specializing in harmonized data interpretation. Provide insights on data integration and consistency across sources."
        elif tab_name == "Export Data":
            system_content = "You are a data export specialist. Provide guidance on exporting data to various systems and formats, considering data integrity and compatibility."
        else:
            system_content = "You are a helpful data analysis assistant with expertise in interpreting data insights for specific data sources."

        tab_context = f"Current tab: {tab_name}\n\n"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"{tab_context}Given the following data:\n\n{context}\n\nUser question: {prompt}"}
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
    
    # Create a 2x2 grid for charts
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    # Numerical columns summary
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    if not num_cols.empty:
        with col1:
            st.subheader(f"Numerical Columns Summary for {source_name}")
            st.dataframe(data[num_cols].describe())

        if len(num_cols) > 1:
            try:
                with col2:
                    st.subheader(f"Correlation Heatmap for {source_name}")
                    fig = px.imshow(data[num_cols].corr(), color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                    fig.update_layout(title=f'Correlation Heatmap - {source_name}')
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                    charts.append(f"Correlation Heatmap - {source_name}")

                # Add insights on positive and negative correlations
                corr_matrix = data[num_cols].corr()
                with col3:
                    st.subheader(f"Correlation Insights for {source_name}")
                    st.write("Top Positive Correlations:")
                    top_pos = corr_matrix.unstack().sort_values(ascending=False).drop_duplicates()
                    top_pos = top_pos[(top_pos != 1.0) & (top_pos > 0)].head(5)
                    st.write(top_pos)
                with col4:
                    st.write("Top Negative Correlations:")
                    top_neg = corr_matrix.unstack().sort_values().drop_duplicates()
                    top_neg = top_neg[top_neg < 0].head(5)
                    st.write(top_neg)
                charts.append(f"Top Positive/Negative Correlations - {source_name}")
            except Exception as e:
                st.warning(f"Unable to generate correlation heatmap and insights for {source_name}. Error: {str(e)}")

        # Scatter plot with regression line
        with col1:
            st.subheader(f"Scatter Plot with Regression for {source_name}")
            x_col = st.selectbox("Select X-axis", num_cols, key=f"{source_name}_scatter_x")
            y_col = st.selectbox("Select Y-axis", num_cols, key=f"{source_name}_scatter_y")
            
            try:
                fig = px.scatter(data, x=x_col, y=y_col, trendline="ols")
                fig.update_layout(title=f'Linear Regression: {y_col} vs {x_col} - {source_name}')
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                charts.append(f"Linear Regression: {y_col} vs {x_col} - {source_name}")
            except Exception as e:
                st.warning(f"Unable to generate scatter plot with regression for {source_name}. Error: {str(e)}")

        # Distribution plot
        with col2:
            st.subheader(f"Distribution Plot for {source_name}")
            dist_col = st.selectbox("Select Column for Distribution", num_cols, key=f"{source_name}_dist_col")
            try:
                fig = px.histogram(data, x=dist_col, marginal="box")
                fig.update_layout(title=f'Distribution of {dist_col} - {source_name}')
                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                charts.append(f"Distribution of {dist_col} - {source_name}")
            except Exception as e:
                st.warning(f"Unable to generate distribution plot for {source_name}. Error: {str(e)}")

    # Categorical columns summary
    cat_cols = data.select_dtypes(include=['object', 'category']).columns
    if not cat_cols.empty:
        with col3:
            st.subheader(f"Categorical Columns Summary for {source_name}")
            for col in cat_cols:
                try:
                    st.write(f"{col} - Top 5 Categories")
                    top_5 = data[col].value_counts().nlargest(5)
                    fig = px.bar(x=top_5.index, y=top_5.values)
                    fig.update_layout(title=f'Top 5 Categories in {col} - {source_name}', xaxis_title=col, yaxis_title='Count')
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                    charts.append(f"Top 5 Categories in {col} - {source_name}")
                except Exception as e:
                    st.warning(f"Unable to generate categorical summary for {col} in {source_name}. Error: {str(e)}")

        # Categorical vs Numerical
        if not num_cols.empty:
            with col4:
                st.subheader(f"Categorical vs Numerical for {source_name}")
                cat_col = st.selectbox("Select Categorical Column", cat_cols, key=f"{source_name}_cat_col")
                num_col = st.selectbox("Select Numerical Column", num_cols, key=f"{source_name}_num_col")
                try:
                    fig = px.box(data, x=cat_col, y=num_col)
                    fig.update_layout(title=f'{num_col} by {cat_col} - {source_name}')
                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                    charts.append(f"{num_col} by {cat_col} - {source_name}")
                except Exception as e:
                    st.warning(f"Unable to generate categorical vs numerical plot for {source_name}. Error: {str(e)}")

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
    
    buffer.write("Summary Statistics:\n")
    buffer.write(data.describe().to_string() + "\n\n")
    
    buffer.write("Generated Charts:\n")
    for chart in charts:
        buffer.write(f"- {chart}\n")
    buffer.write("\n")
    
    try:
        buffer.write(f"Correlation Matrix:\n{data.corr().to_string()}\n\n")
    except Exception as e:
        buffer.write(f"Unable to generate correlation matrix. Error: {str(e)}\n\n")
    
    for col in data.columns:
        buffer.write(f"Column: {col}\n")
        buffer.write(f"Unique Values: {data[col].nunique()}\n")
        buffer.write("Top 5 Values:\n")
        buffer.write(data[col].value_counts().nlargest(5).to_string() + "\n\n")
    
    return buffer.getvalue()

def get_ai_export_script(data, export_format, target_system):
    client = get_openai_client()
    try:
        context = f"Data summary:\n{data.describe().to_string()}\n\n"
        context += f"Data columns:\n{', '.join(data.columns)}\n\n"
        context += f"Data sample:\n{data.head().to_string()}\n\n"
        
        prompt = f"Given the following data summary, columns, and sample, generate a {export_format} script to export this data to {target_system}:"
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are an expert in generating {export_format} scripts for data integration with {target_system}."},
                {"role": "user", "content": f"{prompt}\n\n{context}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"An error occurred: {str(e)}"

def show(project_name):
    st.header(f"Dashboard Summary for Project: {project_name}")

    if project_name not in st.session_state.projects:
        st.warning(f"Project '{project_name}' not found.")
        return

    project_data = st.session_state.projects[project_name]

    if 'cleaned_data_sources' not in project_data or not project_data['cleaned_data_sources']:
        st.warning(f"No cleaned data available for {project_name}. Please clean the data first.")
        return

    # Create tabs for each data source, harmonized data, and export data
    data_sources = project_data['cleaned_data_sources']
    tab_names = list(data_sources.keys()) + ["Harmonized Data", "Export Data"]
    tabs = st.tabs(tab_names)

    all_charts = []

    # Initialize chat_open as a dictionary if it doesn't exist or is not a dictionary
    if 'chat_open' not in st.session_state or not isinstance(st.session_state.chat_open, dict):
        st.session_state.chat_open = {tab: False for tab in tab_names}

    # Initialize messages as a dictionary if it doesn't exist or is not a dictionary
    if 'messages' not in st.session_state or not isinstance(st.session_state.messages, dict):
        st.session_state.messages = {tab: [] for tab in tab_names}

    for i, (source, data) in enumerate(data_sources.items()):
        with tabs[i]:
            st.subheader(f"Data from {source}")
            
            # Check if data is a DataFrame, if not, try to convert it
            if not isinstance(data, pd.DataFrame):
                try:
                    data = pd.DataFrame(data)
                except Exception as e:
                    st.error(f"Unable to process the data for {source}. Error: {str(e)}")
                    continue

            col1, col2 = st.columns(2)
            col1.metric("Rows", data.shape[0])
            col2.metric("Columns", data.shape[1])

            st.subheader("Data Sample")
            st.dataframe(data.head())

            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Type': data.dtypes.astype(str),
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

            # Add chat button for each tab
            chat_button = st.button("💬 Chat with Your Data", key=f"chat_button_{source}")

            if chat_button:
                if source not in st.session_state.chat_open:
                    st.session_state.chat_open[source] = False
                st.session_state.chat_open[source] = not st.session_state.chat_open[source]

            # Chat interface for each tab
            if source in st.session_state.chat_open and st.session_state.chat_open[source]:
                chat_container = st.container()
                with chat_container:
                    # Display chat messages
                    if source not in st.session_state.messages:
                        st.session_state.messages[source] = []
                    for message in st.session_state.messages[source]:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                    # Chat input
                    if prompt := st.chat_input(f"Ask about {source} data...", key=f"chat_input_{source}"):
                        st.session_state.messages[source].append({"role": "user", "content": prompt})
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        with st.chat_message("assistant"):
                            message_placeholder = st.empty()
                            full_response = get_ai_response(prompt, data, source_charts, source)
                            message_placeholder.markdown(full_response)
                        st.session_state.messages[source].append({"role": "assistant", "content": full_response})

    # Harmonized Data tab
    with tabs[-2]:  # Change this from [-1] to [-2]
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

        # Add chat button for harmonized data
        chat_button = st.button("💬 Chat with Harmonized Data", key="chat_button_harmonized")

        if chat_button:
            if "Harmonized Data" not in st.session_state.chat_open:
                st.session_state.chat_open["Harmonized Data"] = False
            st.session_state.chat_open["Harmonized Data"] = not st.session_state.chat_open["Harmonized Data"]

        # Chat interface for harmonized data
        if "Harmonized Data" in st.session_state.chat_open and st.session_state.chat_open["Harmonized Data"]:
            chat_container = st.container()
            with chat_container:
                # Display chat messages
                for message in st.session_state.messages["Harmonized Data"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("Ask about harmonized data...", key="chat_input_harmonized"):
                    st.session_state.messages["Harmonized Data"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = get_ai_response(prompt, harmonized_data, harmonized_charts, "Harmonized Data")
                        message_placeholder.markdown(full_response)
                    st.session_state.messages["Harmonized Data"].append({"role": "assistant", "content": full_response})

    # Export Data tab
    with tabs[-1]:  # This will now correctly refer to the last tab
        st.subheader("Export Data to Legacy Systems")
        
        # Select data source for export
        export_source = st.selectbox("Select data source for export", 
                                     options=list(data_sources.keys()) + ["Harmonized Data"])
        
        if export_source == "Harmonized Data":
            export_data = harmonize_data(data_sources)
        else:
            export_data = data_sources[export_source]
        
        # Define legacy systems and their icons
        legacy_systems = {
            "SAP ERP": "🏢",
            "Oracle E-Business Suite": "🔮",
            "Microsoft Dynamics": "🪟",
            "Infor": "🔷",
            "Sage": "🌿",
            "JD Edwards": "🏗️",
            "Epicor": "🔺",
            "IBM AS/400": "🖥️",
            "Salesforce": "☁️",
            "Custom Legacy System": "🔧"
        }
        
        # Display legacy system icons
        st.write("Select target legacy system:")
        NUM_COLS = 4
        cols = st.columns(NUM_COLS)
        selected_system = None
        
        for i, (system_name, icon) in enumerate(legacy_systems.items()):
            if cols[i % NUM_COLS].button(f"{icon} {system_name}", key=f"system_{system_name}", use_container_width=True):
                selected_system = system_name
        
        if selected_system:
            st.subheader(f"Exporting to {selected_system}")
            
            # Define export formats for the selected system
            export_formats = {
                "SQL": "📊",
                "CSV": "📁",
                "XML": "🗂️",
                "JSON": "📋",
                "API": "🔌",
                "Flat File": "📄"
            }
            
            st.write("Select export format:")
            format_cols = st.columns(len(export_formats))
            selected_format = None
            
            for i, (format_name, icon) in enumerate(export_formats.items()):
                if format_cols[i].button(f"{icon} {format_name}", key=f"export_{format_name}", use_container_width=True):
                    selected_format = format_name
            
            if selected_format:
                st.subheader(f"Generated {selected_format} Script for {selected_system}")
                script = get_ai_export_script(export_data, selected_format, selected_system)
                st.code(script, language=selected_format.lower())
                
                # Download button for the generated script
                script_extension = ".sql" if selected_format == "SQL" else ".xml" if selected_format == "XML" else ".json" if selected_format == "JSON" else ".txt"
                st.download_button(
                    label=f"Download {selected_format} Script for {selected_system}",
                    data=script,
                    file_name=f"{project_name}_{export_source}_to_{selected_system}{script_extension}",
                    mime="text/plain",
                )
                
                # Display additional information
                st.info(f"This script demonstrates how to export data from your project to {selected_system} using {selected_format} format. In a real-world scenario, you would need to configure specific connection details and may require additional middleware or ETL tools for seamless integration.")

        # Add chat button for export data
        chat_button = st.button("💬 Chat about Data Export", key="chat_button_export")

        if chat_button:
            if "Export Data" not in st.session_state.chat_open:
                st.session_state.chat_open["Export Data"] = False
            st.session_state.chat_open["Export Data"] = not st.session_state.chat_open["Export Data"]

        # Chat interface for export data
        if "Export Data" in st.session_state.chat_open and st.session_state.chat_open["Export Data"]:
            chat_container = st.container()
            with chat_container:
                # Display chat messages
                for message in st.session_state.messages["Export Data"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("Ask about data export...", key="chat_input_export"):
                    st.session_state.messages["Export Data"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        full_response = get_ai_response(prompt, export_data, all_charts, "Export Data")
                        message_placeholder.markdown(full_response)
                    st.session_state.messages["Export Data"].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    show(st.session_state.current_project)
