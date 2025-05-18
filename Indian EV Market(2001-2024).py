#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import zipfile
import os


# In[54]:


zip_path = "India EV Market Data.zip"        
extract_to = "ev_dataset"      


# In[55]:


with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print(f"Files extracted to: {extract_to}")


# In[56]:


for root, dirs, files in os.walk(extract_to):
    for file in files:
        print(os.path.join(root, file))


# In[57]:


import pandas as pd

 #Loading one csv file
df = pd.read_csv("ev_dataset/ev_sales_by_makers_and_cat_15-24.csv")
df.head()


# In[58]:


df.info()
df.describe()
df.isnull().sum()


# In[59]:


import numpy as np
# Select only numeric columns
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Fill missing values for numeric columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())


# In[60]:


# Select non-numeric (categorical) columns
categorical_columns = df.select_dtypes(exclude=[np.number]).columns

# Fill missing values in categorical columns with the mode (most frequent value)
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])


# In[61]:


# Check if there are any remaining missing values
print(df.isnull().sum())


# In[62]:


# Check for duplicates
print(df.duplicated().sum())  # Prints the number of duplicate rows

# Drop duplicates if necessary
df.drop_duplicates(inplace=True)


# In[63]:


# Check the data types of all columns
print(df.dtypes)

# Convert columns to appropriate data types if necessary
df['Maker'] = df['Maker'].astype('category')  # For categorical columns
# If the year columns are not numeric, convert them
year_columns = [str(year) for year in range(2015, 2025)]
df[year_columns] = df[year_columns].apply(pd.to_numeric, errors='coerce')


# In[64]:


# Convert 'Cat' and 'Maker' columns to categorical type
df['Cat'] = df['Cat'].astype('category')
df['Maker'] = df['Maker'].astype('category')


# In[65]:


# Check the data types after conversion
print(df.dtypes)


# In[66]:


df.head()


# In[67]:


df_ev_makers = pd.read_csv("ev_dataset\EV Maker by Place.csv")


# In[68]:


# Clean text columns by stripping spaces and standardizing the case
df_ev_makers['EV Maker'] = df_ev_makers['EV Maker'].str.strip().str.title()
df_ev_makers['Place'] = df_ev_makers['Place'].str.strip().str.title()
df_ev_makers['State'] = df_ev_makers['State'].str.strip().str.title()

# Check for any missing values or duplicates
print(df_ev_makers.isnull().sum())
print(df_ev_makers.duplicated().sum())

# Drop duplicates if needed
df_ev_makers.drop_duplicates(inplace=True)


# In[69]:


# Drop duplicate rows
df_ev_makers.drop_duplicates(inplace=True)

# Verify after removing duplicates
print(df_ev_makers.duplicated().sum())


# In[70]:


df_pcs = pd.read_csv("ev_dataset\OperationalPC.csv")


# In[71]:


# Strip spaces and standardize text
df_pcs['State'] = df_pcs['State'].str.strip().str.title()

# Check for missing values
print(df_pcs.isnull().sum())

# Check for duplicates
print(df_pcs.duplicated().sum())

# Drop duplicates if needed
df_pcs.drop_duplicates(inplace=True)


# In[72]:


df_pcs['State'] = df_pcs['State'].str.strip().str.title()  # For standardizing state names


# In[73]:


# Check if the 'No. of Operational PCS' column contains only numerical values
df_pcs['No. of Operational PCS'] = pd.to_numeric(df_pcs['No. of Operational PCS'], errors='coerce')

# Check for any missing or invalid values after conversion
print(df_pcs.isnull().sum())


# In[74]:


df_vehicle_class = pd.read_csv("ev_dataset\Vehicle Class - All.csv")


# In[75]:


import pandas as pd

# Load the CSV, treating '#######' as NaN
df_vehicle_class = pd.read_csv("ev_dataset\Vehicle Class - All.csv", na_values=['#######'])
na_values=(['#######'])
# Convert 'Total Registration' to string to handle string operations
df_vehicle_class['Total Registration'] = df_vehicle_class['Total Registration'].astype(str)

# Remove commas
df_vehicle_class['Total Registration'] = df_vehicle_class['Total Registration'].str.replace(',', '')

# Convert to numeric, coercing errors (e.g., 'nan' strings)
df_vehicle_class['Total Registration'] = pd.to_numeric(df_vehicle_class['Total Registration'], errors='coerce')

# Fill missing values with 0 (assign back to the column to avoid warning)
df_vehicle_class['Total Registration'] = df_vehicle_class['Total Registration'].fillna(0)




# In[76]:


print(df_vehicle_class)


# In[77]:


print(df_vehicle_class['Total Registration'].isnull().sum())
print(df_vehicle_class['Total Registration'].unique())
df_vehicle_class['Total Registration'] = (
    df_vehicle_class['Total Registration']
    .astype(str)
    .str.replace(',', '')
    .astype(int)
)


#This means it didn't have any missing values. It was just excel display format for handling long numbers so this csv file is also cleaned now".


# In[78]:


df_vehicle_class.to_csv('vehicle_class_cleaned.csv', index=False)


# In[79]:


df_ev_cat = pd.read_csv("ev_dataset\ev_cat_01-24.csv")


# In[80]:


# Step 1: Identify the date column
date_col = df_ev_cat.columns[0]

# Step 2: Parse dates in the format DD/MM/YY (like 01/01/01)
df_ev_cat[date_col] = pd.to_datetime(
    df_ev_cat[date_col],
    format='%d/%m/%y',
    errors='coerce'  # Invalid dates become NaT
)

# Step 3: Drop rows with invalid/missing dates
df_ev_cat = df_ev_cat.dropna(subset=[date_col])

# Step 4: Clean numeric columns (remove commas, convert to integers)
for col in df_ev_cat.columns[1:]:  # Skip date column
    df_ev_cat[col] = (
        df_ev_cat[col]
        .astype(str)
        .str.replace(',', '', regex=False)
        .pipe(pd.to_numeric, errors='coerce')  # Convert to float, handle non-numeric
        .fillna(0)
        .astype(int)  # Convert to int if needed
    )

# Step 5: Display cleaned data
print(df_ev_cat.head())


# In[81]:


print(df_ev_cat.isnull().sum())
#Data is cleaned successfully!


# In[82]:


print("Data is cleaned successfully!")


# In[ ]:





# In[123]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from st_on_hover_tabs import on_hover_tabs

# --- Global Custom CSS ---
st.markdown("""
    <style>
        body, .main { background-color: #111111; }
        [data-testid="stSidebar"] {
            background-color: #111 !important;
        }
        [data-testid="stSidebar"] * {
            color: white !important;
        }
        .stMetric {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 15px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    sales = pd.read_csv("ev_dataset/ev_sales_by_makers_and_cat_15-24.csv")
    ev_cat = pd.read_csv("ev_dataset/ev_cat_01-24.csv")
    ev_makers = pd.read_csv("ev_dataset/EV Maker by Place.csv")
    pcs = pd.read_csv("ev_dataset/OperationalPC.csv")
    vehicle_class = pd.read_csv("vehicle_class_cleaned.csv")
    return sales, ev_cat, ev_makers, pcs, vehicle_class

sales, ev_cat, ev_makers, pcs, vehicle_class = load_data()

# --- Sidebar Navigation ---
with st.sidebar:
    selected_tab = on_hover_tabs(
        tabName=[
            "Getting Started",
            "EV Makers by Place",
            "EV Categories",
            "EV Sales by Maker & Category",
            "Charging Infrastructure",
            "Vehicle Class"
        ],
        iconName=[
            'arrow_forward', 'arrow_forward', 'arrow_forward',
            'arrow_forward', 'arrow_forward', 'arrow_forward'
        ],
        styles={
            'navtab': {
                'background-color': '#111',
                'color': 'white',
                'font-size': '16px',   
                'font-weight': 'bold',
                'transition': '.3s',
                'white-space': 'nowrap',
                'text-transform': 'none'
            },
            'tabStyle': {
                'list-style-type': 'none',
                'margin-bottom': '20px',
                'padding-left': '20px'
            },
            'tabStyle:active': {
                'color': '#00FFAA',
                'background-color': '#222'
            },
            'tabStyle:hover': {
                'color': '#FFD700',
                'cursor': 'pointer'
            },
            'iconStyle': {
                'margin-right': '15px'
            },
        },
        default_choice=0,
        key="nav"
    )

# --- Tab: Getting Started ---
if selected_tab == "Getting Started":
    st.markdown("""
        <style>
        .main-title {
            font-size: 40px;
            font-weight: bold;
            text-align: center;
            color: #2E8B57;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 22px;
            text-align: center;
            color: #000000;
            margin-top: 0px;
        }
        .marquee {
            font-size: 18px;
            font-weight: 500;
            color: #FF6347;
            margin-top: 10px;
            white-space: nowrap;
            overflow: hidden;
            box-sizing: border-box;
        }
        .marquee-text {
            display: inline-block;
            padding-left: 100%;
            animation: marquee 15s linear infinite;
        }
        @keyframes marquee {
            0% { transform: translate(0, 0); }
            100% { transform: translate(-100%, 0); }
        }
        .description {
            font-size: 18px;
            text-align: justify;
            margin: 30px auto;
            max-width: 900px;
            line-height: 1.6;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🚗 Indian EV Market (2001–2024)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Welcome to the Indian EV Market Dashboard!</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="marquee">
      <div class="marquee-text">⚡ Explore EV trends, top manufacturers, vehicle types, and charging stations across India from 2001 to 2024! ⚡</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="description">
        This interactive dashboard provides comprehensive insights into the Indian Electric Vehicle (EV) market from 2001 to 2024. 📊<br><br>
        🔹 Use the navigation panel on the left to explore data on:
        <ul>
            <li>EV makers and their regional presence</li>
            <li>Vehicle categories and classes</li>
            <li>Annual sales across brands and types</li>
            <li>Charging infrastructure evolution</li>
        </ul>
        🧭 Navigate through the sections to discover key trends, market leaders, and regional adoption patterns.
    </div>
    """, unsafe_allow_html=True)

# --- Tab: EV Makers by Place ---
if selected_tab == "EV Makers by Place":
    st.header("Manufacturing Geography Analysis")

    state_counts = ev_makers['State'].value_counts()
    city_counts = ev_makers['Place'].value_counts()
    total_ev_makers = ev_makers['EV Maker'].nunique()
    top_state = state_counts.idxmax()
    top_state_count = state_counts.max()
    top_city = city_counts.idxmax()
    top_city_count = city_counts.max()
    top_states = state_counts.head(4).index.tolist()
    emerging_hubs = city_counts.head(3).index.tolist()

    specializations = ['Passenger Vehicles', 'Commercial Vehicles', '3-Wheelers', '2-Wheelers']
    regional_specialization = list(zip(top_states, specializations))

    # KPIs
    col1, col2, col3 = st.columns(3)

    
   
    with col1:
        st.metric("Total EV Makers", total_ev_makers)
        st.caption(f"_There are {total_ev_makers} unique EV manufacturers in India._")
    with col2:
        st.metric("Top Manufacturing State", top_state)
        st.caption(f"_{top_state} leads with {top_state_count} EV manufacturers._")
    with col3:
        st.metric("Top Manufacturing Hub", top_city)
        st.caption(f"_{top_city} has {top_city_count} EV makers._")



    st.markdown("---")

    # State Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.barplot(x=state_counts.values, y=state_counts.index, palette="viridis", ax=ax1)
    ax1.set_title('State-wise EV Manufacturing Presence', color='white', fontsize=14)
    ax1.set_xlabel('Number of Manufacturers', color='white')
    ax1.set_ylabel('States', color='white')
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#111111')
    fig1.patch.set_facecolor('#111111')
    st.pyplot(fig1)
    st.caption(f"_{top_state} leads with {top_state_count} manufacturers, followed by {state_counts.index[1]} ({state_counts[1]}) and {state_counts.index[2]} ({state_counts[2]})._")

    # City Clusters
    fig2, ax2 = plt.subplots(figsize=(10, 3))
    sns.barplot(x=city_counts.head(10).values, y=city_counts.head(10).index, palette="mako", ax=ax2)
    ax2.set_title('Top 10 Manufacturing Cities', color='white', fontsize=14)
    ax2.set_xlabel('Number of Manufacturers', color='white')
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#111111')
    fig2.patch.set_facecolor('#111111')
    st.pyplot(fig2)
    st.caption(f"_{', '.join(emerging_hubs)} are the top manufacturing hubs in India._")

    # Manufacturer Type Pie
    types = ['Automotive Giant' if 'Motor' in x else 'Startup' for x in ev_makers['EV Maker']]
    pie_series = pd.Series(types).value_counts()
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax3.pie(
        pie_series.values, autopct='%1.1f%%', startangle=140,
        colors=['#4B8BBE', '#306998'], textprops={'color': 'white'}
    )
    ax3.legend(wedges, pie_series.index, title="Type", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    ax3.set_title('Manufacturer Profile', color='white')
    fig3.patch.set_facecolor('#111111')
    ax3.set_facecolor('#111111')
    st.pyplot(fig3)
    st.caption(f"_{pie_series.index[0]}s form {pie_series.values[0]/pie_series.sum()*100:.1f}%, {pie_series.index[1]}s form {pie_series.values[1]/pie_series.sum()*100:.1f}%._")

    # Regional Specialization Table
    st.subheader("Regional Manufacturing Specialization")
    spec_df = pd.DataFrame(regional_specialization, columns=["State", "Specialization"])
    st.dataframe(spec_df, use_container_width=True, hide_index=True)
    st.caption(f"_{', '.join([f'{row[0]}: {row[1]}' for row in regional_specialization])}_")

    # Emerging Cities List
    st.subheader("🚀 Emerging Manufacturing Hubs")
    st.markdown("  \n".join([f"- **{city}**: Among the top EV manufacturing cities" for city in emerging_hubs]))
    st.caption(f"_{', '.join(emerging_hubs)} are shaping India's EV manufacturing landscape._")


# In[93]:


import os

if selected_tab == "EV Categories":
    st.markdown(
        '<div class="section-heading"><span class="emoji">📊</span>EV Category Analysis</div>',
        unsafe_allow_html=True
    )

    # Load and prepare dataset
    ev_cat_data = pd.read_csv(os.path.join("ev_dataset", "ev_cat_01-24.csv"))

    # Ensure 'Date' column exists and convert to datetime if possible
    if 'Date' in ev_cat_data.columns:
        ev_cat_data['Date'] = pd.to_datetime(ev_cat_data['Date'], errors='coerce')
        ev_cat_data.set_index('Date', inplace=True)

    # Automatically detect numeric columns (vehicle categories)
    numeric_cols = ev_cat_data.select_dtypes(include='number').columns
    ev_cat_data[numeric_cols] = ev_cat_data[numeric_cols].fillna(0)

    # Insight 1: Total registrations
    total_reg = ev_cat_data[numeric_cols].sum().sum()

    # Insight 2: Top category
    category_sums = ev_cat_data[numeric_cols].sum()
    top_category = category_sums.idxmax()
    top_category_count = category_sums.max()

    # Insight 3: Two-wheeler share
    two_wheeler_cols = [col for col in numeric_cols if 'TWO WHEELER' in col.upper()]
    two_wheeler_share = (ev_cat_data[two_wheeler_cols].sum().sum() / total_reg * 100) if two_wheeler_cols else 0

    # KPIs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-label">Total Registrations</div><div class="kpi-text-orange">{int(total_reg):,}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-caption">Total EVs registered across all categories</div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-label">Top Category</div><div class="kpi-text-blue">{top_category}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-caption">Most popular EV category with {int(top_category_count):,} registrations</div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-label">Two-Wheeler Share</div><div class="kpi-text-green">{two_wheeler_share:.1f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi-caption">Two-wheelers\' contribution to EV market</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Category-wise Distribution
    st.markdown('<div class="chart-title">Category-wise Distribution</div>', unsafe_allow_html=True)
    category_totals = category_sums.sort_values(ascending=False)
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=category_totals.values, y=category_totals.index, palette="viridis", ax=ax1)
    ax1.set_title('Total Registrations by Category', color='white')
    ax1.set_xlabel('Number of Registrations', color='white')
    ax1.set_ylabel('Vehicle Categories', color='white')
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#111111')
    fig1.patch.set_facecolor('#111111')
    st.pyplot(fig1)

    # Pie Chart
    st.markdown('<div class="chart-title">Market Share: Top Categories</div>', unsafe_allow_html=True)
    top_n = 3
    top_categories = category_totals.head(top_n)
    other_sum = category_totals[top_n:].sum()
    labels = list(top_categories.index) + ['Other Categories']
    sizes = list(top_categories.values) + [other_sum]
    fig2, ax2 = plt.subplots(figsize=(7, 7))
    wedges, _, autotexts = ax2.pie(
        sizes, labels=None, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(labels)),
        textprops={'color': 'white', 'fontsize': 12}
    )
    ax2.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=10)
    ax2.set_title('Market Share by Vehicle Category', color='white')
    fig2.patch.set_facecolor('#111111')
    ax2.set_facecolor('#111111')
    st.pyplot(fig2)

    # Registration Trends
    st.markdown('<div class="chart-title">Registration Trends Over Time</div>', unsafe_allow_html=True)
    top_trend_categories = category_totals.head(4).index.tolist()
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    for category in top_trend_categories:
        ax3.plot(ev_cat_data.index, ev_cat_data[category], label=category)
    ax3.set_title('Monthly Registration Trends', color='white')
    ax3.set_xlabel('Date', color='white')
    ax3.set_ylabel('Registrations', color='white')
    ax3.tick_params(colors='white')
    ax3.legend(fontsize=10)
    ax3.set_facecolor('#111111')
    fig3.patch.set_facecolor('#111111')
    st.pyplot(fig3)

    # Fastest Growing Categories
    growth_rates = []
    for col in numeric_cols:
        col_series = ev_cat_data[col].dropna()
        non_zero = col_series[col_series > 0]
        if len(non_zero) >= 2:
            first_val = non_zero.iloc[0]
            last_val = non_zero.iloc[-1]
            if first_val > 0:
                growth = ((last_val - first_val) / first_val) * 100
                growth_rates.append((col, growth))
    growth_df = pd.DataFrame(growth_rates, columns=['Category', 'Growth Rate (%)']).sort_values('Growth Rate (%)', ascending=False)
    st.markdown('<div class="chart-title">Fastest Growing Categories</div>', unsafe_allow_html=True)
    st.dataframe(growth_df.head(5), use_container_width=True, hide_index=True)

    # Emerging Categories
    midpoint = len(ev_cat_data) // 2
    recent_growth = []
    for col in numeric_cols:
        first_half = ev_cat_data[col].iloc[:midpoint].mean()
        second_half = ev_cat_data[col].iloc[midpoint:].mean()
        if first_half > 0:
            growth = ((second_half / first_half) - 1) * 100
            recent_growth.append((col, growth))
    emerging_df = pd.DataFrame(recent_growth, columns=['Category', 'Recent Growth (%)']).sort_values('Recent Growth (%)', ascending=False)
    st.markdown('<div class="chart-title">Emerging Categories</div>', unsafe_allow_html=True)
    st.dataframe(emerging_df.head(5), use_container_width=True, hide_index=True)

    # Peak Registration Months
    monthly_peaks = pd.DataFrame({
        'Category': numeric_cols,
        'Peak Month': [ev_cat_data[col].idxmax().strftime('%b %Y') if pd.notna(ev_cat_data[col].idxmax()) else '-' for col in numeric_cols],
        'Peak Value': [ev_cat_data[col].max() for col in numeric_cols]
    }).sort_values('Peak Value', ascending=False)
    st.markdown('<div class="chart-title">Peak Registration Months</div>', unsafe_allow_html=True)
    st.dataframe(monthly_peaks.head(5), use_container_width=True, hide_index=True)


# 

# 

# 

# In[85]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
plt.style.use("dark_background")

# Load dataset
df = pd.read_csv("ev_dataset/OperationalPC.csv")  # Ensure the path is correct

# Create
if selected_tab == "Charging Infrastructure":


    # Sidebar: Select top N states
    st.sidebar.title("Filter Options")
    top_n = st.sidebar.slider("Select Top N States", min_value=1, max_value=len(df), value=5)

    # Insights Calculations
    total_pcs = df['No. of Operational PCS'].sum()
    avg_pcs = df['No. of Operational PCS'].mean()
    max_pcs_state = df.loc[df['No. of Operational PCS'].idxmax()]
    min_pcs_state = df.loc[df['No. of Operational PCS'].idxmin()]

    # Header and Key Stats
    st.title("Operational Public Charging Stations (PCS) in Indian States")
    st.write(f"**Total number of operational PCS**: {total_pcs}")
    st.write(f"**Average number of operational PCS per state**: {avg_pcs:.2f}")
    st.write(f"**State with the most operational PCS**: {max_pcs_state['State']} with {max_pcs_state['No. of Operational PCS']} PCS")
    st.write(f"**State with the least operational PCS**: {min_pcs_state['State']} with {min_pcs_state['No. of Operational PCS']} PCS")

    # Filter top N states
    top_states_df = df.nlargest(top_n, 'No. of Operational PCS')

    # --- Visualizations ---

    # Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='No. of Operational PCS', y='State', data=top_states_df, palette='viridis', ax=ax)
    ax.set_title(f"Top {top_n} States by Operational PCS")
    ax.set_xlabel("Number of Operational PCS")
    ax.set_ylabel("State")
    st.pyplot(fig)

    # Pie Chart
    fig_pie = px.pie(df, names='State', values='No. of Operational PCS', title="Proportion of Operational PCS by State")
    st.plotly_chart(fig_pie)

    # Histogram
    fig_hist, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['No. of Operational PCS'], kde=True, color='blue', bins=10, ax=ax)
    ax.set_title("Distribution of Operational PCS across States")
    ax.set_xlabel("Number of Operational PCS")
    ax.set_ylabel("Frequency")
    st.pyplot(fig_hist)


 


# In[119]:


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("dark_background")

# Load dataset
df = pd.read_csv("ev_dataset/ev_sales_by_makers_and_cat_15-24.csv")  


# Create tabs
if selected_tab ==  "EV Sales by Maker & Category":

    # Clean column names and ensure correct dtypes
    df.columns = df.columns.str.strip()
    df['Maker'] = df['Maker'].astype(str).str.strip().str.upper()
    df['Cat'] = df['Cat'].astype(str).str.strip().str.upper()

    # Extract year columns
    year_cols = [col for col in df.columns if col.isdigit()]
    df[year_cols] = df[year_cols].fillna(0).astype(int)

    # Total registrations per maker
    df['Total'] = df[year_cols].sum(axis=1)

    # -------------------------- INSIGHTS --------------------------- #

    st.title("📈 EV Maker Insights (2015–2024)")

    col1, col2, col3 = st.columns(3)

    # 1. Total EV registrations
    total_reg = df['Total'].sum()
    col1.metric("Total EV Registrations", f"{total_reg:,}")

    # 2. Top maker overall
    top_maker = df.loc[df['Total'].idxmax()]
    col2.metric("Top Maker (Overall)", top_maker['Maker'], f"{top_maker['Total']:,} units")

    # 3. Top maker in latest year
    latest_year = max(map(int, year_cols))
    top_latest = df[['Maker', str(latest_year)]].sort_values(str(latest_year), ascending=False).iloc[0]
    col3.metric(f"Top Maker in {latest_year}", top_latest['Maker'], f"{top_latest[str(latest_year)]:,} units")

    # 4. Most consistent maker (most active years)
    df['Active_Years'] = (df[year_cols] > 0).sum(axis=1)
    most_consistent = df.sort_values('Active_Years', ascending=False).iloc[0]
    st.success(f"📌 Most Consistent Maker: **{most_consistent['Maker']}** with activity in **{most_consistent['Active_Years']} years**")

    # 5. Highest growth from first active year to latest year
    growth_data = []
    for idx, row in df.iterrows():
        active_years = row[year_cols][row[year_cols] > 0]
        if len(active_years) >= 2:
            growth = ((active_years.iloc[-1] - active_years.iloc[0]) / active_years.iloc[0]) * 100 if active_years.iloc[0] > 0 else 0
            growth_data.append((row['Maker'], growth))
    growth_df = pd.DataFrame(growth_data, columns=['Maker', 'Growth (%)']).sort_values('Growth (%)', ascending=False)
    top_growth = growth_df.iloc[0]
    st.info(f"🚀 Highest Growth: **{top_growth['Maker']}** with **{top_growth['Growth (%)']:.1f}%** growth from first active year to {latest_year}")

    # 6. Most popular EV category
    cat_totals = df.groupby('Cat')['Total'].sum().sort_values(ascending=False)
    most_popular_cat = cat_totals.idxmax()
    st.markdown(f"📊 Most Popular EV Category: **{most_popular_cat}** with **{cat_totals.max():,} registrations**")

    # 7. Fastest growing category (2015–2019 vs 2020–2024)
    first_half = [str(y) for y in range(2015, 2020)]
    second_half = [str(y) for y in range(2020, 2025)]
    df['H1'] = df[first_half].sum(axis=1)
    df['H2'] = df[second_half].sum(axis=1)
    growth_cat_df = df.groupby('Cat')[['H1', 'H2']].sum()
    growth_cat_df['Growth %'] = ((growth_cat_df['H2'] - growth_cat_df['H1']) / growth_cat_df['H1'].replace(0, 1)) * 100
    fastest_cat = growth_cat_df.sort_values('Growth %', ascending=False).iloc[0]
    st.markdown(f"📈 Fastest Growing Category: **{fastest_cat.name}** with **{fastest_cat['Growth %']:.1f}%** growth")

    # 8. Emerging Makers (only active after 2020)
    emerging = df[(df[year_cols[:6]].sum(axis=1) == 0) & (df[year_cols[6:]].sum(axis=1) > 0)]
    st.markdown(f"🌱 Emerging Makers (post-2020): **{len(emerging)}** makers started EV production after 2020")

    # -------------------------- VISUALS --------------------------- #

    st.markdown("### 📊 Top 10 Makers by Total Registrations")
    top10 = df[['Maker', 'Total']].sort_values('Total', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=top10, x='Total', y='Maker', palette='magma', ax=ax)
    ax.set_title('Top 10 EV Makers (2015–2024)')
    ax.set_xlabel('Total Registrations')
    st.pyplot(fig)
    
    st.markdown("### 📅 Year-wise Total EV Registrations")
    year_totals = df[year_cols].sum()
    fig, ax = plt.subplots()
    sns.lineplot(x=year_totals.index, y=year_totals.values, marker='o', ax=ax, color='cyan')
    ax.set_title("Total EV Registrations per Year")
    ax.set_ylabel("Registrations")
    ax.set_xlabel("Year")
    st.pyplot(fig)
    
    st.markdown("### 🚗 Category-wise EV Registration Trend (2015–2024)")
    cat_year_trend = df.groupby('Cat')[year_cols].sum().T
    fig, ax = plt.subplots()
    cat_year_trend.plot(ax=ax, marker='o')
    ax.set_title("EV Category Trends Over Time")
    ax.set_xlabel("Year")
    ax.set_ylabel("Registrations")
    st.pyplot(fig)

    if not emerging.empty:
        st.markdown("### Emerging Makers (Heatmap of Activity Post-2020)")
        emerging_heatmap = emerging.set_index('Maker')[year_cols[6:]]
        fig, ax = plt.subplots(figsize=(12, min(0.4 * len(emerging_heatmap), 8)))
        sns.heatmap(emerging_heatmap, cmap='YlGnBu', ax=ax, annot=True, fmt='d')
        ax.set_title("Post-2020 Registrations by Emerging Makers")
        st.pyplot(fig)
    
    st.markdown("### 📉 Year-over-Year (YoY) Growth - Top 5 Makers")
    top5_makers = df.sort_values("Total", ascending=False).head(5)
    top5_growth = top5_makers.set_index('Maker')[year_cols].diff(axis=1).fillna(0)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(top5_growth, annot=True, cmap='coolwarm', fmt=".0f", ax=ax)
    ax.set_title("YoY Growth in Registrations for Top 5 Makers")
    st.pyplot(fig)


 



# In[87]:


import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use("dark_background")

# Set Streamlit page layout (optional)
# st.set_page_config(layout="wide")

# Load dataset
df = pd.read_csv("ev_dataset/Vehicle Class - All.csv")  # Use forward slash for cross-platform compatibility


if selected_tab == "Vehicle Class":
    # Clean "Total Registration" to integer
    df['Total Registration'] = df['Total Registration'].str.replace(",", "")
    df['Total Registration'] = df['Total Registration'].astype(int)

    # Show DataFrame
    st.header("📊 Total EV Registrations by Vehicle Class (India)")
    st.dataframe(df)

    # Sorting option
    sort_by = st.selectbox("Sort by", ["Total Registration (High to Low)", "Total Registration (Low to High)"])
    df_sorted = df.sort_values("Total Registration", ascending=(sort_by == "Total Registration (Low to High)"))

    # 1. Basic scatter plot
    st.subheader("🔵 Total EV Registrations by Vehicle Class")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=df_sorted, x='Total Registration', y='Vehicle Class', s=200, color='darkgreen', ax=ax1)
    ax1.set_title("Total Registrations per Vehicle Class")
    st.pyplot(fig1)
    st.markdown("*This chart shows the absolute scale of EV registrations for each vehicle class, helping to identify the most and least popular categories.*")

    
    # 3. Horizontal scatter plot
    
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.scatter(df_sorted["Vehicle Class"], df_sorted["Total Registration"], color='tomato', s=100)
    ax3.set_xticklabels(df_sorted["Vehicle Class"], rotation=90)
    ax3.set_title("Registration Scatter by Vehicle Class")
    st.pyplot(fig3)
    st.markdown("*This plot focuses on categorical variation, highlighting how registrations are distributed across vehicle types horizontally.*")

    # 4. Highlight Top N categories
    st.subheader("🔵 Highlight Top N Categories")
    top_n = st.slider("Select Top N", 3, len(df_sorted), 5)
    highlight_df = df_sorted.head(top_n)
    fig4, ax4 = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=highlight_df, x="Total Registration", y="Vehicle Class", color='navy', s=300, ax=ax4)
    ax4.set_title(f"Top {top_n} Vehicle Classes by Registration")
    st.pyplot(fig4)
    st.markdown(f"*This dynamic plot shows the top {top_n} vehicle classes by registration volume, making it easy to zoom in on the leaders.*")

    # 5. Log scale scatter
    st.subheader("🔵 Log-Scaled Scatter (High Range Support)")
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.set_xscale("log")
    sns.scatterplot(data=df_sorted, x='Total Registration', y='Vehicle Class', color='purple', s=150, ax=ax5)
    ax5.set_title("Log-Scaled Registration Plot")
    st.pyplot(fig5)
    st.markdown("*This log-scale plot helps detect lower-volume classes and highlights disparities.*")


# In[ ]:





# In[ ]:




