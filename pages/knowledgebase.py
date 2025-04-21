import streamlit as st
import sys, os
import tests.utils as utils
from google.cloud import bigquery
from google.oauth2 import service_account
from openai import OpenAI
import json, time
import openai
from google.cloud import firestore as fs
from google.oauth2 import service_account as fs_sa

# --- Boilerplate for imports & session init ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir  = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
utils.init()


# --- BigQuery client ---
bq_creds = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
bq_client = bigquery.Client(credentials=bq_creds, project=bq_creds.project_id)

# Firestore client
fs_creds = fs_sa.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
fs_client = fs.Client(
    project=fs_creds.project_id,
    credentials=fs_creds,
    database="steraflow-firestore"  # ✅ correct
)

# Permission test
try:
    fs_client.collection("test_write_check").document("ping").set({"check": "success"})
    st.success("Firestore write test: ✅ Success")
except Exception as e:
    st.error(f"Firestore write test failed: {e}")

def generate_context(prompt, scope):
    with st.spinner("Generating context..."):
        # Extract schema and sample data if provided in scope
        schema_desc = ""
        sample_desc = ""
        if isinstance(scope, dict):
            if 'schema' in scope:
                schema_desc = "\n".join([f"- {f.name} ({f.field_type})" for f in scope['schema']])
            if 'sample' in scope:
                if isinstance(scope['sample'], str):
                    sample_desc = scope['sample']
                else:
                    sample_desc = scope['sample'].to_csv(index=False)
            scope_text = scope.get('text', '')
        else:
            scope_text = scope

        response = openai.chat.completions.create(                        
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a context‑generation assistant. Use the provided schema and sample rows to write accurate context."},
                {"role": "user", "content": (
                    f"Scope: {scope_text}\n\n"
                    f"Schema:\n{schema_desc}\n\n"
                    f"Sample Rows:\n{sample_desc}\n\n"
                    f"User wants: {prompt}"
                )}
            ]
        )
    return response.choices[0].message.content

def add_column_context(dataset_table: str, column_name: str, column_type: str, df_sample=None):
    with st.expander(f"🔍 {column_name} ({column_type})"):
        prompt = st.text_area(
            "Describe the context for this column:",
            placeholder="E.g., This column contains customer IDs that uniquely identify each customer...",
            key=f"prompt_{column_name}",
            height=100
        )
        if st.button("Generate Context", key=f"gen_{column_name}"):
            col_sample = df_sample[column_name].dropna().unique().tolist()[:5] if df_sample is not None else []
            sample_desc = ", ".join(map(str, col_sample))
            context = generate_context(prompt, {
                'text': f"Column: {column_name} ({column_type})",
                'sample': f"Sample Values: {sample_desc}"
            })
            st.session_state[f'context_{column_name}'] = context
        
        if f'context_{column_name}' in st.session_state:
            st.info(st.session_state[f'context_{column_name}'])

def show():
    st.title("Knowledge Base Explorer")
    st.write("Explore and document your BigQuery datasets with AI-powered context generation.")

    try:
        datasets = list(bq_client.list_datasets())
        dataset_names = [ds.dataset_id for ds in datasets]
    except Exception as e:
        st.error(f"Error fetching datasets: {e}")
        return

    if not datasets:
        st.warning("No BigQuery datasets found.")
        return

    col1, col2 = st.columns(2)
    with col1:
        selected_dataset = st.selectbox(
            "Select a Dataset",
            options=dataset_names,
            key="dataset_dropdown"
        )

    if selected_dataset:
        try:
            tables = list(bq_client.list_tables(selected_dataset))
            table_names = [tbl.table_id for tbl in tables]
        except Exception as e:
            st.error(f"Error listing tables in {selected_dataset}: {e}")
            return

        with col2:
            selected_table = st.selectbox(
                "Select a Table",
                options=table_names,
                key="table_dropdown"
            )

        if selected_table:
            full_table_ref = f"{selected_dataset}.{selected_table}"
            try:
                table = bq_client.get_table(full_table_ref)
                schema = table.schema
                df_sample = bq_client.list_rows(table, max_results=5).to_dataframe()
                
                # Table-level context
                with st.expander("📝 Table Context", expanded=True):
                    prompt = st.text_area(
                        "Describe the context for this table:",
                        placeholder="E.g., This table contains customer transaction data...",
                        height=100
                    )
                    if st.button("Generate Table Context"):
                        context = generate_context(prompt, {
                            'text': f"Table: {full_table_ref}",
                            'schema': schema,
                            'sample': df_sample
                        })
                        st.session_state['table_context'] = context
                    
                    if st.button("Generate Full Table + Column Context"):
                        table_context = generate_context(prompt, {
                            'text': f"Table: {full_table_ref}",
                            'schema': schema,
                            'sample': df_sample
                        })
                        st.session_state['table_context'] = table_context

                        for field in schema:
                            col_sample = df_sample[field.name].dropna().unique().tolist()[:5]
                            sample_desc = ", ".join(map(str, col_sample))
                            col_prompt = f"Generate context for column `{field.name}` of type `{field.field_type}`."
                            col_context = generate_context(col_prompt, {
                                'text': f"Column: {field.name} ({field.field_type})",
                                'sample': f"Sample Values: {sample_desc}"
                            })
                            st.session_state[f'context_{field.name}'] = col_context

                        # Build payload
                        from datetime import datetime

                        payload = {
                            "table": full_table_ref,
                            "generated_at": datetime.utcnow().isoformat(),
                            "table_context": st.session_state["table_context"],
                            "column_contexts": {
                                field.name: st.session_state[f"context_{field.name}"]
                                for field in schema
                            }
                        }
                        # Write to Firestore
                        doc_id = full_table_ref.replace(".", "_")
                        try:
                            fs_client.collection("kb_contexts").document(doc_id).set(payload)
                            st.success("Context saved to Firestore.")
                        except Exception as e:
                            import traceback
                            st.error("�� Firestore write failed.")
                            st.code(traceback.format_exc())
                        
                        if 'table_context' in st.session_state:
                            st.info(st.session_state['table_context'])
                # Schema with column-level context
                st.subheader("Table Schema")
                for field in table.schema:
                    add_column_context(full_table_ref, field.name, field.field_type, df_sample)
                
            except Exception as e:
                st.error(f"Error fetching schema for {full_table_ref}: {e}")
                return

if __name__ == "__main__":
    show()