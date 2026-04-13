import streamlit as st
import pandas as pd
import numpy as np
from infrencing.predict_freight import predict_freight_cost
from infrencing.predict_invoice import predict_invoice_flag

# page configuration

st.set_page_config(
    page_title='Vendor Invoice Intelligence Portal',
    page_icon='📦',
    layout='wide',
)

# Header section
st.markdown(
    """
    # Vendor Invoice Intelligence Portal
    ### AI - Driven Freight Cost Prediction and Invoice Flagging System
    
    This internal annalytics portal leverages machine learning to
    - **Forcast freight accurately**
    - **Detect risky or abnormal vendor invoices**
    - **Reduce financial leakage and improve operational efficiency**"""
)

st.divider()

st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio(
    "Choose a model:",
    ["Freight Cost Prediction", "Invoice Flagging"]
)

st.sidebar.markdown("""
---
**Buisness Impact**
- Improved cost forcasting
- Reduces invoice fraud & annomalies
- Faster financial decision making
""")   

# fREIGHT COST PREDICTION
if selected_model == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")
    st.markdown(
        """
        This model predicts the freight cost based on various input features such as:
        - **Distance**
        - **Weight**
        - **Shipping Method**
        - **Historical Data**
        
        The model uses advanced machine learning algorithms to provide accurate cost forcasts, helping to optimize logistics and reduce expenses.
        """
    )
    
    with st.form("freight_form"):
        col1, col2 = st.columns(2) 
        with col1:
            quantity = st.number_input("Quantity", min_value=1, value=1200)

        with col2:
            dollars_per_unit = st.number_input("Dollars per Unit", min_value=0.01, value=1850.0)
        
        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        dollars = quantity * dollars_per_unit
        input_data = {
            "Dollars": [dollars]
        }
        prediction_df = predict_freight_cost(input_data)
        prediction = prediction_df["predicted_freight"]

        st.success(f"Predicted Freight Cost: ${prediction.iloc[0]:,.2f}")

# Invoice Flagging

else:
    st.subheader("Invoice Flagging")

    st.markdown(
        """
        This model flags potentially risky or abnormal vendor invoices based on historical data and predefined rules.
        It helps in identifying fraudulent activities and ensuring financial compliance.
        """
    )

    with st.form("invoice_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            invoice_quantity = st.number_input("Invoice Quantity", min_value=1, value=50)
            freight = st.number_input("Freight Cost", min_value=0.0, value=1.73)
        
        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value=1.0,
                value=352.95
            )
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value=1,
                value=162
            )
        with col3:
            total_item_dollars = st.number_input(
                "Total Item Dollars",
                min_value=1.0,
                value=2476.0
            )
        submit_flag = st.form_submit_button("Evaluate Invoice Risk")
    if submit_flag:
        input_data = {
            "invoice_quantity": [invoice_quantity],
            "invoice_dollars": [invoice_dollars],
            "freight": [freight],
            "total_item_quantity": [total_item_quantity],
            "total_item_dollars": [total_item_dollars]
        }
        flag_prediction = predict_invoice_flag(input_data)["predicted_flag"]

        is_flagged = bool(flag_prediction[0])

        if is_flagged:
            st.error("This invoice has been flagged as potentially risky or abnormal.")  
        else:
            st.success("This invoice appears to be normal and does not require further review.") 



