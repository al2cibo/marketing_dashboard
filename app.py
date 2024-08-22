import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta, datetime

@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file, engine='openpyxl')
        df['Week'] = pd.to_datetime(df['Week'].str.split('-').str[0], format='%m/%d/%y')
        df['Week'] = df['Week'] - pd.to_timedelta(df['Week'].dt.dayofweek, unit='D')  # Ensure weeks start on Monday
        numeric_columns = ['Conversions', '% of Conversions', 'Revenue', '% of Revenue', 'Spend', '% of Spend', 'CPA', 'ROAS']
        
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
        df = df.dropna(subset=numeric_columns, how='all')
        
        tier_columns = ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5']
        for i in range(len(tier_columns)-1, -1, -1):
            df[tier_columns[i]] = df[tier_columns[i]].fillna(df[tier_columns[i-1]] if i > 0 else '')
        
        df[tier_columns] = df[tier_columns].astype(str)
        
        df['RB Conv'] = df['Revenue'] / df['Conversions']
        df['RB CPO'] = df['Spend'] / df['Conversions']
        df['AO'] = df['Revenue'] / df['Spend']
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def filter_data(df, filters):
    for col, value in filters.items():
        if value and value != 'All':
            df = df[df[col] == value]
    return df

def plot_metrics(df, x, y, title, color=None, sort=True):
    if sort:
        df = df.sort_values(y, ascending=False)
    fig = px.bar(df, x=x, y=y, title=title, color=color)
    fig.update_layout(xaxis_title=x, yaxis_title=y, height=400)
    return fig

def plot_time_series(df, title):
    weekly_revenue = df.groupby('Week')['Revenue'].sum().reset_index()
    fig = px.line(weekly_revenue, x='Week', y='Revenue', title=title)
    fig.update_layout(
        xaxis_title='Week',
        yaxis_title='Revenue ($)',
        height=500
    )
    return fig

def plot_scatter(df, x, y, size, color, title):
    fig = px.scatter(df, x=x, y=y, size=size, color=color, hover_name=df.index, title=title)
    fig.update_layout(xaxis_title=x, yaxis_title=y, height=400)
    return fig

def plot_pie_chart(df, values, names, title):
    fig = px.pie(df, values=values, names=names, title=title)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def display_metrics(df):
    total_spend = df['Spend'].sum()
    total_revenue = df['Revenue'].sum()
    total_conversions = df['Conversions'].sum()
    overall_roas = total_revenue / total_spend if total_spend > 0 else np.inf
    overall_cpa = total_spend / total_conversions if total_conversions > 0 else np.inf
    overall_rb_conv = total_revenue / total_conversions if total_conversions > 0 else np.inf
    overall_rb_cpo = total_spend / total_conversions if total_conversions > 0 else np.inf
    overall_ao = total_revenue / total_spend if total_spend > 0 else np.inf
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spend", f"${total_spend:,.2f}")
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col1.metric("Total Conversions", f"{total_conversions:,.0f}")
    
    col2.metric("Overall ROAS", f"{overall_roas:.2f}" if not np.isinf(overall_roas) else "∞")
    col2.metric("Overall CPA", f"${overall_cpa:.2f}" if not np.isinf(overall_cpa) else "∞")
    col2.metric("Overall AO", f"{overall_ao:.2f}" if not np.isinf(overall_ao) else "∞")
    
    col3.metric("Overall RB Conv", f"${overall_rb_conv:.2f}" if not np.isinf(overall_rb_conv) else "∞")
    col3.metric("Overall RB CPO", f"${overall_rb_cpo:.2f}" if not np.isinf(overall_rb_cpo) else "∞")

def create_filters(df):
    filters = {}
    
    # Tier 2 selection
    tier2_options = ['All'] + sorted(df['Tier 2'].dropna().astype(str).unique().tolist())
    selected_tier2 = st.selectbox("Select Tier 2", tier2_options)
    filters['Tier 2'] = selected_tier2
    
    # Tier 3 filtering
    if selected_tier2 != 'All':
        tier3_options = ['All'] + sorted(df[df['Tier 2'] == selected_tier2]['Tier 3'].dropna().astype(str).unique().tolist())
        filters['Tier 3'] = st.selectbox("Filter Tier 3", tier3_options)
    
    return filters

def create_date_filter(df):
    min_date = df['Week'].min().date()
    max_date = df['Week'].max().date()
    
    weeks = pd.date_range(start=min_date, end=max_date, freq='W-MON')
    week_options = [f"Week {week.strftime('%Y-%m-%d')} to {(week + timedelta(days=6)).strftime('%Y-%m-%d')}" for week in weeks]
    
    selected_week = st.selectbox("Select Week", week_options)
    start_date = datetime.strptime(selected_week.split(' ')[1], '%Y-%m-%d').date()
    end_date = start_date + timedelta(days=6)
    
    return start_date, end_date

def analyze_data(df, groupby_column):
    grouped_data = df.groupby(groupby_column).agg({
        'Spend': 'sum',
        'Revenue': 'sum',
        'Conversions': 'sum'
    }).reset_index()
    grouped_data['ROAS'] = grouped_data['Revenue'] / grouped_data['Spend']
    grouped_data['CPA'] = grouped_data['Spend'] / grouped_data['Conversions']
    grouped_data['RB Conv'] = grouped_data['Revenue'] / grouped_data['Conversions']
    grouped_data['RB CPO'] = grouped_data['Spend'] / grouped_data['Conversions']
    grouped_data['AO'] = grouped_data['Revenue'] / grouped_data['Spend']
    grouped_data = grouped_data.replace([np.inf, -np.inf], np.nan)
    return grouped_data

def analyze_best_performers(df, tier):
    tier_data = analyze_data(df, tier)
    
    metrics = ['Revenue', 'Conversions', 'ROAS', 'CPA', 'AO']
    top_performers = {}
    
    for metric in metrics:
        top_performers[metric] = tier_data.nlargest(5, metric).dropna(subset=[metric])
        # Reorder columns to show the metric first
        cols = [tier, metric] + [col for col in top_performers[metric].columns if col not in [tier, metric]]
        top_performers[metric] = top_performers[metric][cols]
    
    return top_performers

def display_best_performers(top_performers):
    metrics = ['Revenue', 'Conversions', 'ROAS', 'CPA', 'AO']
    
    for metric in metrics:
        st.subheader(f"Top 5 by {metric}")
        st.dataframe(top_performers[metric].set_index(top_performers[metric].columns[0]), use_container_width=True)

def historical_overview(df):
    st.header("Historical Overview")
    
    # Overall metrics
    display_metrics(df)
    
    # Time series plot (Revenue only)
    st.subheader("Revenue Over Time")
    st.plotly_chart(plot_time_series(df, "Revenue Over Time"), use_container_width=True)
    
    # Tier distribution
    st.subheader("Revenue Distribution by Tier 2")
    tier2_revenue = df.groupby('Tier 2')['Revenue'].sum().reset_index()
    st.plotly_chart(plot_pie_chart(tier2_revenue, 'Revenue', 'Tier 2', "Revenue Distribution by Tier 2"), use_container_width=True)
    
    st.subheader("Spend Distribution by Tier 2")
    tier2_spend = df.groupby('Tier 2')['Spend'].sum().reset_index()
    st.plotly_chart(plot_pie_chart(tier2_spend, 'Spend', 'Tier 2', "Spend Distribution by Tier 2"), use_container_width=True)
    
    # Scatter plot
    st.subheader("Spend vs. Revenue")
    st.plotly_chart(plot_scatter(df, 'Spend', 'Revenue', 'Conversions', 'ROAS', 'Spend vs. Revenue'), use_container_width=True)

def weekly_analysis(df):
    st.header("Weekly Analysis")
    
    start_date, end_date = create_date_filter(df)
    weekly_df = df[(df['Week'].dt.date >= start_date) & (df['Week'].dt.date <= end_date)]
    
    # Display metrics for the selected week
    display_metrics(weekly_df)
    
    # Plots for the selected week
    st.subheader("Revenue by Tier 3")
    tier3_revenue = weekly_df.groupby('Tier 3')['Revenue'].sum().reset_index()
    st.plotly_chart(plot_metrics(tier3_revenue, 'Tier 3', 'Revenue', 'Revenue by Tier 3', sort=True), use_container_width=True)
    
    st.subheader("Conversions by Tier 3")
    tier3_conversions = weekly_df.groupby('Tier 3')['Conversions'].sum().reset_index()
    st.plotly_chart(plot_metrics(tier3_conversions, 'Tier 3', 'Conversions', 'Conversions by Tier 3', sort=True), use_container_width=True)
    
    # Best performers for the selected week
    st.subheader("Best Performers")
    st.subheader("Tier 4 Best Performers")
    top_performers_tier4 = analyze_best_performers(weekly_df, 'Tier 4')
    display_best_performers(top_performers_tier4)
    
    st.subheader("Tier 5 Best Performers")
    top_performers_tier5 = analyze_best_performers(weekly_df, 'Tier 5')
    display_best_performers(top_performers_tier5)
    
    # Weekly performance table
    st.subheader("Weekly Performance Data")
    weekly_performance_data = weekly_df.drop(columns=['Tier 1', 'Tier 2'])
    st.dataframe(weekly_performance_data.set_index(weekly_performance_data.columns[0]).style.format({
        'Spend': '${:,.2f}',
        'Revenue': '${:,.2f}',
        'Conversions': '{:,.0f}',
        'ROAS': '{:.2f}',
        'CPA': '${:.2f}',
        'AO': '{:.2f}'
    }), use_container_width=True)

def main():
    st.set_page_config(layout="wide", page_title="Marketing Analysis Dashboard")
    st.title("Marketing Analysis Dashboard")
    st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Analyze your marketing performance across different tiers and time periods</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            #st.sidebar.header("Filters")
            filters = create_filters(df)
            filtered_df = filter_data(df, filters)
            
            tab1, tab2 = st.tabs(["Historical Overview", "Weekly Analysis"])
            
            with tab1:
                historical_overview(filtered_df)
            
            with tab2:
                weekly_analysis(filtered_df)

if __name__ == "__main__":
    main()
