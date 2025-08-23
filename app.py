import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import google.generativeai as genai # <-- Updated import
import os
import io

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Interactive EDA Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_data(file, file_extension):
    """Load data from CSV or Excel file."""
    try:
        if file_extension == 'csv':
            return pd.read_csv(file)
        elif file_extension in ['xls', 'xlsx']:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_summary(df):
    """Generate a string summary of the dataframe for the AI prompt."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=np.number)
    
    summary = f"""
    Here is a summary of the dataset:

    1.  **Shape of the data:** {df.shape} (rows, columns)

    2.  **Column Names and Data Types:**
    {info_str}

    3.  **Missing Values Count per Column:**
    {df.isnull().sum().to_string()}

    4.  **Descriptive Statistics for Numerical Columns:**
    {numeric_df.describe().to_string()}
    """
    
    # Add correlation matrix summary only if there are at least 2 numeric columns
    if numeric_df.shape[1] >= 2:
        summary += f"""
    5.  **Correlation Matrix (Top 10 absolute correlations):**
    (Note: This is a sample of the strongest correlations, not the full matrix)
    {numeric_df.corr().unstack().sort_values(ascending=False, key=abs).drop_duplicates().head(10).to_string()}
    """
    else:
        summary += "\n5. **Correlation Matrix:** Not available (less than 2 numeric columns)."
        
    return summary

# --- INITIALIZE SESSION STATE ---
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.original_df = None
    st.session_state.file_name = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("🔬 Interactive EDA Tool")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file",
        type=['csv', 'xls', 'xlsx']
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.file_name:
            st.session_state.file_name = uploaded_file.name
            file_extension = uploaded_file.name.split('.')[-1]
            df = load_data(uploaded_file, file_extension)
            if df is not None:
                st.session_state.original_df = df.copy()
                st.session_state.df = df.copy()
                st.sidebar.success("File uploaded successfully!")
    
    if st.session_state.df is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("Navigation")
        app_mode = st.sidebar.radio(
            "Go to",
            ["Dataset Overview", "Data Cleaning", "Data Visualization", "Generate EDA Report"]
        )
    else:
        app_mode = "Dataset Overview"

# --- MAIN CONTENT ---
if st.session_state.df is None:
    st.info("👋 Welcome! Please upload a dataset using the sidebar to get started.")

else:
    df = st.session_state.df.copy() # Use a copy to avoid modifying session state directly in sections

    # --- 1. DATASET OVERVIEW ---
    if app_mode == "Dataset Overview":
        st.header("1. Dataset Overview")
        st.subheader("First 5 Rows of the Dataset")
        st.dataframe(df.head())
        # ... (rest of the code is identical to the previous version) ...
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Dataset Shape")
            st.write(f"Rows: `{df.shape[0]}`")
            st.write(f"Columns: `{df.shape[1]}`")
        with col2:
            st.subheader("Data Types")
            st.dataframe(df.dtypes.rename("Data Type"))
        st.subheader("Missing Values")
        missing_values = df.isnull().sum().sort_values(ascending=False)
        missing_percentage = (missing_values / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Count': missing_values,
            'Missing Percentage (%)': missing_percentage
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0])
        st.subheader("Descriptive Statistics (Numerical Columns)")
        st.dataframe(df.describe())

    # --- 2. DATA CLEANING ---
    elif app_mode == "Data Cleaning":
        st.header("2. Data Cleaning")
        st.subheader("Handle Missing Values")
        # ... (rest of the code is identical to the previous version) ...
        cols_with_missing = df.columns[df.isnull().any()].tolist()
        if not cols_with_missing:
            st.success("No missing values found in the dataset! 🎉")
        else:
            col_to_impute = st.selectbox("Select a column to handle missing values", options=cols_with_missing)
            if pd.api.types.is_numeric_dtype(df[col_to_impute]):
                impute_method = st.radio(f"Choose imputation method for '{col_to_impute}'", ('Mean', 'Median', 'Mode', 'Drop Rows'))
            else:
                impute_method = st.radio(f"Choose imputation method for '{col_to_impute}'", ('Mode', 'Drop Rows'))
            if st.button(f"Apply '{impute_method}' to '{col_to_impute}'"):
                if impute_method == 'Drop Rows':
                    st.session_state.df.dropna(subset=[col_to_impute], inplace=True)
                else:
                    if impute_method == 'Mean': fill_value = st.session_state.df[col_to_impute].mean()
                    elif impute_method == 'Median': fill_value = st.session_state.df[col_to_impute].median()
                    else: fill_value = st.session_state.df[col_to_impute].mode()[0]
                    st.session_state.df[col_to_impute].fillna(fill_value, inplace=True)
                st.success(f"Successfully applied imputation to '{col_to_impute}'.")
                st.rerun()
        st.markdown("---")
        st.subheader("Outlier Detection and Removal (IQR Method)")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numerical_cols:
            st.warning("No numerical columns available for outlier detection.")
        else:
            col_to_scan = st.selectbox("Select a numerical column to scan for outliers", options=numerical_cols)
            Q1 = df[col_to_scan].quantile(0.25); Q3 = df[col_to_scan].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR; upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col_to_scan] < lower_bound) | (df[col_to_scan] > upper_bound)]
            st.write(f"Number of outliers detected in '{col_to_scan}': `{len(outliers)}`")
            if not outliers.empty:
                st.dataframe(outliers)
                if st.button(f"Remove outliers from '{col_to_scan}'"):
                    st.session_state.df = df[(df[col_to_scan] >= lower_bound) & (df[col_to_scan] <= upper_bound)]
                    st.success(f"Outliers removed. The dataset now has {st.session_state.df.shape[0]} rows.")
                    st.rerun()

    # --- 3. DATA VISUALIZATION ---
    elif app_mode == "Data Visualization":
        st.header("3. Data Visualization")
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # ... (rest of the code is identical to the previous version) ...
        tab1, tab2, tab3 = st.tabs(["Univariate", "Bivariate", "Multivariate"])
        with tab1:
            st.subheader("Univariate Analysis")
            col_to_plot = st.selectbox("Select a column", options=df.columns)
            if col_to_plot in numerical_cols:
                plot_type = st.radio("Select plot type", ("Histogram", "Box Plot"))
                if plot_type == "Histogram": fig = px.histogram(df, x=col_to_plot, title=f"Histogram of {col_to_plot}")
                else: fig = px.box(df, y=col_to_plot, title=f"Box Plot of {col_to_plot}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("Count Plot")
                counts = df[col_to_plot].value_counts().reset_index()
                counts.columns = [col_to_plot, 'count']
                fig = px.bar(counts, x=col_to_plot, y='count', title=f"Count Plot of {col_to_plot}")
                st.plotly_chart(fig, use_container_width=True)
        with tab2:
            st.subheader("Bivariate Analysis")
            st.write("Scatter Plot (Numerical vs. Numerical)")
            if len(numerical_cols) >= 2:
                x_col = st.selectbox("Select X-axis", options=numerical_cols, index=0)
                y_col = st.selectbox("Select Y-axis", options=numerical_cols, index=1)
                color_col = st.selectbox("Select column for color (optional)", options=[None] + categorical_cols + numerical_cols)
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot: {x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("You need at least two numerical columns for a scatter plot.")
        with tab3:
            st.subheader("Multivariate Analysis")
            st.write("Correlation Matrix Heatmap")
            if len(numerical_cols) >= 2:
                corr = df[numerical_cols].corr()
                fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            else: st.warning("You need at least two numerical columns for a correlation matrix.")
            st.write("Pair Plot")
            if len(numerical_cols) >= 2:
                st.warning("Pair plots can be slow to render for datasets with many columns.")
                selected_cols = st.multiselect("Select columns for pair plot", options=numerical_cols, default=numerical_cols[:min(5, len(numerical_cols))])
                if st.button("Generate Pair Plot"):
                    if selected_cols:
                        pair_plot_fig = sns.pairplot(df[selected_cols])
                        st.pyplot(pair_plot_fig)
                    else: st.error("Please select at least two columns.")

    # --- 4. GENERATE EDA REPORT (GEMINI VERSION) ---
    elif app_mode == "Generate EDA Report":
        st.header("4. Generate Comprehensive EDA Report")

        if st.button("Generate Report"):
            try:
                # Check for API key
                if "GOOGLE_API_KEY" not in st.secrets:
                    st.error("Google API key not found. Please add it to your Streamlit secrets (`.streamlit/secrets.toml`).")
                else:
                    with st.spinner("AI is analyzing the data and writing the report... This may take a moment."):
                        # 1. Configure the Gemini API client
                        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
                        
                        # 2. Initialize the model
                        model = genai.GenerativeModel('gemini-1.5-flash')

                        # 3. Get Data Summary
                        eda_summary = get_summary(df)

                        # 4. Create a prompt for the AI
                        prompt = f"""
                        You are an expert data analyst. Your task is to generate a comprehensive Exploratory Data Analysis (EDA) report based on the provided data summary.

                        **Instructions:**
                        1.  Start with a high-level executive summary of the dataset.
                        2.  Analyze the data types and comment on their suitability for analysis.
                        3.  Discuss the key findings from the descriptive statistics (mean, median, std dev, etc.). What initial insights can you draw from the central tendency and dispersion of the data?
                        4.  Address the missing values. If there are any, discuss their potential impact and suggest appropriate strategies for handling them (e.g., imputation, removal).
                        5.  Interpret the top correlations. What do they suggest about the relationships between variables? Are there any strong positive or negative correlations worth noting?
                        6.  Provide actionable recommendations and suggest next steps. This could include feature engineering ideas, specific models to try, or further analyses to conduct.
                        7.  Structure the report with clear headings and bullet points using Markdown (e.g., ## Overview, ## Key Findings).
                        8.  The report should be professional, clear, and insightful, aimed at both technical and non-technical stakeholders.

                        **Here is the data summary:**
                        ---
                        {eda_summary}
                        ---
                        """

                        # 5. Call the Gemini API
                        response = model.generate_content(prompt)
                        
                        # 6. Display the report
                        st.markdown(response.text)

            except Exception as e:
                st.error(f"An error occurred while generating the report: {e}")