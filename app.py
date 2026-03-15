import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Auto Sales EDA App", layout="wide")

st.title("🚗 Auto Sales Exploratory Data Analysis App")

st.write("Upload the Auto Sales dataset to explore insights.")

# Upload dataset
uploaded_file = st.file_uploader("Upload Auto Sales CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))

    st.write("Dataset Shape:", df.shape)

    # Data Types
    st.subheader("Data Types")
    st.write(df.dtypes)

    # Duplicate Rows
    st.subheader("Duplicate Records")
    duplicates = df.duplicated().sum()
    st.write("Duplicate Rows:", duplicates)

    df = df.drop_duplicates()
    st.write("Shape After Removing Duplicates:", df.shape)

    # Outlier Detection
    st.subheader("Outlier Detection (IQR Method)")

    num_cols = df.select_dtypes(include=np.number)

    for col in num_cols.columns:

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        st.write(f"{col} Outliers:", outliers.shape[0])

    # Check required columns before using them
    required_cols = ['CUSTOMERNAME', 'COUNTRY', 'ORDERDATE', 'SALES']

    if all(col in df.columns for col in required_cols):

        # Unique values
        st.subheader("Unique Values")
        st.write("Unique Customers:", df['CUSTOMERNAME'].nunique())
        st.write("Unique Countries:", df['COUNTRY'].nunique())

        # Convert Date
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
        df['Year'] = df['ORDERDATE'].dt.year
        df['Month'] = df['ORDERDATE'].dt.month

        st.subheader("Date Conversion")
        st.dataframe(df[['ORDERDATE','Year','Month']].head())

        # Monthly Sales
        monthly_sales = df.groupby(['Year','Month'])['SALES'].sum().reset_index()

        st.subheader("Monthly Sales Data")
        st.dataframe(monthly_sales.head())

        # Monthly Trend
        st.subheader("Monthly Sales Trend")

        fig, ax = plt.subplots()
        ax.plot(monthly_sales['Month'], monthly_sales['SALES'])
        ax.set_xlabel("Month")
        ax.set_ylabel("Total Sales")
        ax.set_title("Monthly Sales Trend")

        st.pyplot(fig)

        # Top Countries
        st.subheader("Top 3 Countries by Sales")

        country_sales = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False)
        st.write(country_sales.head(3))

        # Average Order Value
        if 'DEALSIZE' in df.columns:

            st.subheader("Average Order Value by Deal Size")
            st.write(df.groupby('DEALSIZE')['SALES'].mean())

        # Above MSRP
        if 'PRICEEACH' in df.columns and 'MSRP' in df.columns:

            st.subheader("Items Sold Above MSRP")
            above_msrp = df[df['PRICEEACH'] > df['MSRP']]
            st.write("Items Sold Above MSRP:", above_msrp.shape[0])

        # Product Line Sales
        if 'PRODUCTLINE' in df.columns:

            st.subheader("Sales by Product Line")

            fig2, ax2 = plt.subplots()

            df.groupby('PRODUCTLINE')['SALES'].sum().plot(kind='bar', ax=ax2)

            ax2.set_xlabel("Product Line")
            ax2.set_ylabel("Total Sales")

            st.pyplot(fig2)

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")

        heat_cols = [c for c in ['QUANTITYORDERED','PRICEEACH','SALES'] if c in df.columns]

        if len(heat_cols) > 1:

            corr = df[heat_cols].corr()

            fig3, ax3 = plt.subplots()

            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax3)

            st.pyplot(fig3)

        # Days Since Last Order
        if 'DAYS_SINCE_LASTORDER' in df.columns:

            st.subheader("Customers with Above Avg Days Since Last Order")

            avg_days = df['DAYS_SINCE_LASTORDER'].mean()

            st.dataframe(
                df[df['DAYS_SINCE_LASTORDER'] > avg_days][
                    ['CUSTOMERNAME','DAYS_SINCE_LASTORDER']
                ]
            )

        # Multiple Orders
        if 'ORDERNUMBER' in df.columns:

            st.subheader("Customers with Multiple Orders")

            multi_orders = df.groupby('CUSTOMERNAME')['ORDERNUMBER'].count().reset_index(name='Order_Count')

            multi_orders = multi_orders[multi_orders['Order_Count'] > 1]

            st.dataframe(multi_orders)

    # Missing Values
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    # Numerical and Categorical
    st.subheader("Numerical & Categorical Data")

    num_df = df.select_dtypes(include=np.number)
    cat_df = df.select_dtypes(include='object')

    st.write("Numerical Data")
    st.dataframe(num_df.head())

    st.write("Categorical Data")
    st.dataframe(cat_df.head())

    # Rename Columns
    st.subheader("Renamed Columns")

    df.rename(columns={
        'ORDERNUMBER':'order_number',
        'QUANTITYORDERED':'qty_ordered'
    }, inplace=True)

    st.write(df.columns)

    # Feature Scaling
    st.subheader("Feature Scaling (Standard Scaler)")

    scaler = StandardScaler()

    scaled_df = pd.DataFrame(
        scaler.fit_transform(num_df),
        columns=num_df.columns
    )

    st.dataframe(scaled_df.head())

    # Skewness
    st.subheader("Skewness Analysis")

    skewness = num_df.skew()

    def skew_tag(x):
        if abs(x) < 0.5:
            return "Low Skewed"
        elif abs(x) < 1:
            return "Moderately Skewed"
        else:
            return "Highly Skewed"

    skew_df = pd.DataFrame({
        'Skewness': skewness,
        'Category': skewness.apply(skew_tag)
    })

    st.dataframe(skew_df)

else:

    st.info("Please upload Auto_Sales.csv to start analysis.")