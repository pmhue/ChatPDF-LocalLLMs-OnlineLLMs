import streamlit as st
import requests
from dotenv import load

ollama_endpoint = os.getenv('OLLAMA_ENDPOINT') or "http://localhost:11434"

def pull_model(self):
    """Pull the specified model from the Ollama server."""
    st.spinner(f"Pulling model {self.model_name}...")
    response = requests.post(
        f"{self.base_url}/api/pull",
        json={"model": self.model_name}
    )

    if response.status_code != 200:
        st.error(f"Failed to pull model {self.model_name}: {response.text}")
        raise Exception(f"Model pull failed: {response.text}")

    if self.position_noti == "content":
        st.success(f"Model {self.model_name} pulled successfully.")
    else:
        st.sidebar.success(f"Model {self.model_name} pulled successfully.")