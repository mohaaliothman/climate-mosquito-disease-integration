import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Correlation Dashboard", layout="centered")
st.title("Correlation Dashboard (df_final)")

df = pd.read_csv("final_output.csv")
df.columns = [c.strip() for c in df.columns]

st.write("Dataset preview")
st.dataframe(df.head(10))

num_df = df.select_dtypes(include="number")
if num_df.shape[1] < 2:
    st.error("Not enough numeric columns for correlation.")
    st.stop()

method = st.selectbox("Correlation method", ["spearman", "pearson"], index=0)
use_log = st.checkbox("Apply log1p for large count columns", value=True)

data = num_df.copy()
if use_log:
    for c in data.columns:
        col = data[c].dropna()
        if len(col) and (col >= 0).all() and col.max() > 1000:
            data[c] = np.log1p(data[c])

corr = data.corr(method=method)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)
