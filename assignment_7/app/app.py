import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.express as px

def load_data():
    df = pd.read_csv("assignment_7/dataset/refined_data.csv")
    return df

def train_model(df):
    X = df.drop("Price", axis=1)
    y = df["Price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Laptop Price Predictor")
    st.write("Adjust the configuration below to predict the laptop's price.")

    df = load_data()
    model = train_model(df)

    typename = st.selectbox("TypeName", sorted(df["TypeName"].unique()))
    screen_res = st.slider("Screen Resolution", int(df["ScreenResolution"].min()), int(df["ScreenResolution"].max()), 20)
    cpu = st.slider("CPU", int(df["Cpu"].min()), int(df["Cpu"].max()), 60)
    ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))
    memory = st.slider("Memory", int(df["Memory"].min()), int(df["Memory"].max()), 20)
    gpu = st.slider("GPU", int(df["Gpu"].min()), int(df["Gpu"].max()), 50)

    if st.button("Predict Price"):
        input_df = pd.DataFrame([{
            "TypeName": typename,
            "ScreenResolution": screen_res,
            "Cpu": cpu,
            "Ram": ram,
            "Memory": memory,
            "Gpu": gpu
        }])
        
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Laptop Price: Rupees- {prediction:,.2f}")

        fig = px.scatter(df, x="Cpu", y="Price", title="CPU vs Price")
        fig.add_scatter(x=[cpu], y=[prediction],
                        mode='markers',
                        marker=dict(size=12, color='red'),
                        name='Your Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()

