from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient

def _get_secret(name: str):
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return None

def _get_client():
    endpoint = _get_secret("AZURE_LANGUAGE_ENDPOINT")
    key = _get_secret("AZURE_LANGUAGE_KEY")

    if not endpoint or not key:
        return None

    return TextAnalyticsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key),
    )

def extract_key_phrases_from_azure(text: str) -> dict:
    client = _get_client()

    if client is None:
        return {
            "status": "not_configured",
            "message": "Azure secrets belum diisi.",
            "key_phrases": []
        }

    try:
        response = client.extract_key_phrases([text])[0]

        if response.is_error:
            return {
                "status": "error",
                "message": f"{response.error.code}: {response.error.message}",
                "key_phrases": []
            }

        return {
            "status": "ok",
            "message": "Azure AI Language aktif.",
            "key_phrases": response.key_phrases[:10]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "key_phrases": []
        }
