import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta

@st.cache_data
def load_data(file):
    try:
        df = pd.read_excel(file, engine='openpyxl')
        df['Week'] = pd.to_datetime(df['Week'].str.split('-').str[0], format='%m/%d/%y')
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

def plot_metrics(df, x, y, title, color=None):
    fig = px.bar(df, x=x, y=y, title=title, color=color)
    fig.update_layout(xaxis_title=x, yaxis_title=y, height=400)
    return fig

def plot_time_series(df, metrics):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for metric in metrics:
        if metric not in ['ROAS', 'CPA', 'AO']:
            fig.add_trace(go.Scatter(x=df['Week'], y=df[metric], name=metric), secondary_y=False)
        else:
            fig.add_trace(go.Scatter(x=df['Week'], y=df[metric], name=metric), secondary_y=True)
    
    fig.update_layout(
        title='Performance Over Time',
        xaxis_title='Week',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    fig.update_yaxes(title_text="Amount ($) / Conversions", secondary_y=False)
    fig.update_yaxes(title_text="ROAS / CPA / AO", secondary_y=True)
    return fig

def plot_scatter(df, x, y, size, color, title):
    fig = px.scatter(df, x=x, y=y, size=size, color=color, hover_name=df.index, title=title)
    fig.update_layout(xaxis_title=x, yaxis_title=y, height=400)
    return fig

def plot_line_chart(df, x, y, title):
    fig = px.line(df, x=x, y=y, title=title)
    fig.update_layout(xaxis_title=x, yaxis_title=y, height=400)
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

def create_filters(df, prefix=''):
    filters = {}
    for col in ['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4', 'Tier 5']:
        if col in df.columns:
            options = ['All'] + sorted(df[col].dropna().astype(str).unique().tolist())
            filters[col] = st.selectbox(f"{prefix}Filter by {col}", options, key=f"{prefix}{col}")
    return filters

def create_date_filter(df, key_suffix=''):
    min_date = df['Week'].min().date()
    max_date = df['Week'].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(f"Start date", min_date, key=f"start_date_{key_suffix}")
    with col2:
        end_date = st.date_input(f"End date", max_date, key=f"end_date_{key_suffix}")
    
    start_date = start_date - timedelta(days=start_date.weekday())
    end_date = end_date - timedelta(days=end_date.weekday()) + timedelta(days=6)
    
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

def analyze_campaigns(df, tier):
    campaign_data = analyze_data(df, tier)
    
    metrics = ['Revenue', 'Spend', 'Conversions', 'ROAS', 'CPA', 'RB Conv', 'RB CPO', 'AO']
    top_campaigns = {}
    bottom_campaigns = {}
    
    for metric in metrics:
        top_campaigns[metric] = campaign_data.nlargest(5, metric).dropna(subset=[metric])
        bottom_campaigns[metric] = campaign_data.nsmallest(5, metric).dropna(subset=[metric])
    
    return top_campaigns, bottom_campaigns

def display_campaign_analysis(top_campaigns, bottom_campaigns):
    metrics = ['Revenue', 'Spend', 'Conversions', 'ROAS', 'CPA', 'RB Conv', 'RB CPO', 'AO']
    
    for metric in metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Top 5 by {metric}")
            st.dataframe(top_campaigns[metric], use_container_width=True)
        with col2:
            st.subheader(f"Bottom 5 by {metric}")
            st.dataframe(bottom_campaigns[metric], use_container_width=True)

def overall_analysis(df):
    st.header("Overall Analysis")
    
    start_date, end_date = create_date_filter(df, "overall")
    filtered_df = df[(df['Week'].dt.date >= start_date) & (df['Week'].dt.date <= end_date)]
    
    display_metrics(filtered_df)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Over Time")
        time_series = filtered_df.groupby('Week').agg({
            'Spend': 'sum',
            'Revenue': 'sum',
            'Conversions': 'sum'
        }).reset_index()
        time_series['ROAS'] = time_series['Revenue'] / time_series['Spend']
        time_series['CPA'] = time_series['Spend'] / time_series['Conversions']
        time_series['AO'] = time_series['Revenue'] / time_series['Spend']
        time_series = time_series.replace([np.inf, -np.inf], np.nan)
        st.plotly_chart(plot_time_series(time_series, ['Spend', 'Revenue', 'ROAS', 'CPA', 'AO', 'Conversions']), use_container_width=True)
    
    with col2:
        st.subheader("Weekly Trend of ROAS, CPA, and AO")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=time_series['Week'], y=time_series['ROAS'], name='ROAS'), secondary_y=False)
        fig.add_trace(go.Scatter(x=time_series['Week'], y=time_series['CPA'], name='CPA'), secondary_y=True)
        fig.add_trace(go.Scatter(x=time_series['Week'], y=time_series['AO'], name='AO'), secondary_y=False)
        fig.update_layout(title='Weekly Trend of ROAS, CPA, and AO', xaxis_title='Week', height=400)
        fig.update_yaxes(title_text="ROAS / AO", secondary_y=False)
        fig.update_yaxes(title_text="CPA", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Spend vs. Revenue")
    st.plotly_chart(plot_scatter(filtered_df, 'Spend', 'Revenue', 'Conversions', 'ROAS', 'Spend vs. Revenue'), use_container_width=True)

def tier_analysis(df, tier):
    st.header(f"{tier} Analysis")
    
    start_date, end_date = create_date_filter(df, tier)
    date_filtered_df = df[(df['Week'].dt.date >= start_date) & (df['Week'].dt.date <= end_date)]
    
    filters = create_filters(date_filtered_df, prefix=f"{tier}_")
    filtered_df = filter_data(date_filtered_df, filters)

    display_metrics(filtered_df)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"Top Performers ({tier})")
        top_revenue = filtered_df.groupby(tier)['Revenue'].sum().sort_values(ascending=False).head(10)
        st.plotly_chart(plot_metrics(top_revenue.reset_index(), tier, 'Revenue', f'Top 10 Revenue Generators ({tier})'), use_container_width=True)

    with col2:
        st.subheader(f"Performance Overview ({tier})")
        performance_data = analyze_data(filtered_df, tier)
        st.plotly_chart(plot_scatter(performance_data, 'Spend', 'Revenue', 'Conversions', 'ROAS', f'Performance by {tier}'), use_container_width=True)

    st.header(f"Detailed Analysis ({tier})")
    top_campaigns, bottom_campaigns = analyze_campaigns(filtered_df, tier)
    display_campaign_analysis(top_campaigns, bottom_campaigns)

def main():
    st.set_page_config(layout="wide", page_title="Marketing Analysis Dashboard")
    st.title("Comprehensive Marketing Analysis Dashboard")
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
            tabs = ["Overall"] + [f"Tier {i}" for i in range(1, 6)]
            selected_tab = st.tabs(tabs)

            with selected_tab[0]:
                overall_analysis(df)

            for i, tab in enumerate(selected_tab[1:], start=1):
                with tab:
                    tier_analysis(df, f"Tier {i}")

            st.header("Raw Data")
            st.dataframe(df, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download processed data as CSV",
                data=csv,
                file_name="processed_data.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()