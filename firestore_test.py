import streamlit as st

from google.cloud import firestore
from google.oauth2 import service_account

creds = service_account.Credentials.from_service_account_file(
    "production-build-v1-firebase-adminsdk.json"
)
client = firestore.Client(
    project="production-build-v1",
    database="steraflow-firestore",
    credentials=creds
)


