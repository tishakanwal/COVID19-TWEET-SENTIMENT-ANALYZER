import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="📊 Mini EDA Dashboard", layout="wide")
st.title("📊 Mini EDA Dashboard")

# Sidebar
st.sidebar.header("Upload Your CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display Metadata
    st.subheader("🧾 Data Preview & Info")
    st.write("Shape of dataset:", df.shape)
    st.write("Column names:", list(df.columns))
    
    with st.expander("📂 Preview Data"):
        st.dataframe(df)

    # Filter Section
    st.subheader("🔍 Filter Options")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if cat_cols:
        selected_col = st.selectbox("Select categorical column to filter", cat_cols)
        unique_vals = df[selected_col].dropna().unique()
        selected_vals = st.multiselect(f"Select values for {selected_col}", unique_vals)
        
        if selected_vals:
            df = df[df[selected_col].isin(selected_vals)]
            st.write(f"Filtered Data ({len(df)} rows):")
            st.dataframe(df)

    # Visualizations
    st.subheader("📊 Visualizations")

    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if num_cols:
        col_to_plot = st.selectbox("Select numeric column to visualize", num_cols)
        
        # Bar chart
        st.markdown("#### 📈 Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[col_to_plot], kde=True, ax=ax)
        st.pyplot(fig)
        
        # Pie chart
        if cat_cols:
            st.markdown("#### 🥧 Pie Chart of a Categorical Column")
            pie_col = st.selectbox("Select column for pie chart", cat_cols)
            pie_data = df[pie_col].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
    else:
        st.warning("No numeric columns available for visualization.")
else:
    st.info("📤 Please upload a CSV file to begin.")