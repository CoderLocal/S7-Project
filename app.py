# streamlit_app.py (Streamlit Frontend)
import streamlit as st
from PIL import Image
import io
import requests

# FastAPI URL
fastapi_url = "http://127.0.0.1:8000"

# Streamlit UI Components
st.title("Heirarichal Multimodal Classification Web App")

with st.form(key="model_form"):
    age = st.text_input("Enter the age: ")
    image = st.file_uploader("Choose the image : ",type=["jpg"])
    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    if image and age:
        response = requests.get(f"{fastapi_url}/get_item/{item_name}")
        if response.status_code == 200:
            item_data = response.json()
            st.write(f"Item Name: {item_data['item_name']}")
            st.write(f"Description: {item_data['description']}")
        else:
            st.error("Failed to fetch item.")
    else:
        st.error("Please fill all the fields and upload an image")

# Create an item via FastAPI
st.subheader("Create a new item:")
name = st.text_input("Item Name:")
description = st.text_area("Item Description:")

if st.button("Create Item"):
    new_item = {"name": name, "description": description}
    response = requests.post(f"{fastapi_url}/create_item/", json=new_item)
    if response.status_code == 200:
        st.success(f"Item '{name}' created successfully!")
    else:
        st.error("Failed to create item.")

