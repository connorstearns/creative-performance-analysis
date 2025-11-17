import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


RATE_METRICS = [
    "CTR",
    "CVR",
    "purchase_rate",
    "add_to_cart_rate",
    "view_content_rate",
    "page_view_rate",
]


def classify_objective(objective: str) -> str:
    """
    Classify objective into Awareness, Conversion, or Other based on keywords.
    """
    if not isinstance(objective, str):
        return "Other"
    obj = objective.lower()
    awareness_keywords = ["awareness", "reach", "video view", "brand", "impression"]
    conversion_keywords = ["conversion", "purchase", "sale", "lead", "catalog", "app install"]
    
    if any(k in obj for k in awareness_keywords):
        return "Awareness"
    if any(k in obj for k in conversion_keywords):
        return "Conversion"
    return "Other"


def load_and_prepare_data(uploaded_file):
    """
    Load CSV file and prepare data with derived metrics.
    Returns processed dataframe or None if validation fails.
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        required_cols = ['date', 'platform', 'campaign_name', 'creative_name', 
                        'impressions', 'clicks', 'spend']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.info("Required columns: date, platform, campaign_name, creative_name, impressions, clicks, spend")
            return None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        if df['date'].isna().all():
            st.error("❌ Could not parse any dates in the 'date' column")
            return None
        
        numeric_cols = ['impressions', 'clicks', 'spend']
        optional_numeric = ['conversions', 'revenue', 'purchases', 'add_to_carts', 'view_content', 'page_views']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        for col in optional_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        df['CTR'] = np.where(df['impressions'] > 0, df['clicks'] / df['impressions'], 0)
        df['CPM'] = np.where(df['impressions'] > 0, df['spend'] / df['impressions'] * 1000, 0)
        df['CPC'] = np.where(df['clicks'] > 0, df['spend'] / df['clicks'], 0)
        
        if 'conversions' in df.columns:
            df['CVR'] = np.where(df['clicks'] > 0, df['conversions'] / df['clicks'], 0)
            df['CPA'] = np.where(df['conversions'] > 0, df['spend'] / df['conversions'], 0)
        
        if 'purchases' in df.columns:
            df['purchase_rate'] = np.where(df['clicks'] > 0, df['purchases'] / df['clicks'], 0)
            df['cost_per_purchase'] = np.where(df['purchases'] > 0, df['spend'] / df['purchases'], 0)
        
        if 'add_to_carts' in df.columns:
            df['add_to_cart_rate'] = np.where(df['clicks'] > 0, df['add_to_carts'] / df['clicks'], 0)
            df['cost_per_add_to_cart'] = np.where(df['add_to_carts'] > 0, df['spend'] / df['add_to_carts'], 0)
        
        if 'view_content' in df.columns:
            df['view_content_rate'] = np.where(df['clicks'] > 0, df['view_content'] / df['clicks'], 0)
            df['cost_per_view_content'] = np.where(df['view_content'] > 0, df['spend'] / df['view_content'], 0)
        
        if 'page_views' in df.columns:
            df['page_view_rate'] = np.where(df['clicks'] > 0, df['page_views'] / df['clicks'], 0)
            df['cost_per_page_view'] = np.where(df['page_views'] > 0, df['spend'] / df['page_views'], 0)
        
        if 'revenue' in df.columns:
            df['ROAS'] = np.where(df['spend'] > 0, df['revenue'] / df['spend'], 0)
        
        df = df.sort_values(['creative_name', 'date'])
        
        creative_first_dates = df.groupby('creative_name')['date'].transform('min')
        df['age_in_days'] = (df['date'] - creative_first_dates).dt.days
        
        df['cumulative_impressions'] = df.groupby('creative_name')['impressions'].cumsum()
        
        if 'objective' in df.columns:
            df['objective_type'] = df['objective'].apply(classify_objective)
        
        return df
    
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None


def apply_global_filters(df, filters):
    """
    Apply global filters to the dataframe.
    """
    filtered_df = df.copy()
    
    if filters['date_range']:
        start_date, end_date = filters['date_range']
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.to_datetime(start_date)) & 
            (filtered_df['date'] <= pd.to_datetime(end_date))
        ]
    
    if filters['platforms']:
        filtered_df = filtered_df[filtered_df['platform'].isin(filters['platforms'])]
    
    if filters['campaigns']:
        filtered_df = filtered_df[filtered_df['campaign_name'].isin(filters['campaigns'])]
    
    if filters.get('objectives') is not None:
        all_objectives_in_data = set(df['objective'].dropna().unique())
        if set(filters['objectives']) != all_objectives_in_data:
            filtered_df = filtered_df[filtered_df['objective'].isin(filters['objectives'])]
    
    if 'objective_type' in filtered_df.columns and filters.get('objective_type') not in (None, 'All'):
        filtered_df = filtered_df[filtered_df['objective_type'] == filters['objective_type']]
    
    agg_dict = {'impressions': 'sum'}
    if 'conversions' in filtered_df.columns:
        agg_dict['conversions'] = 'sum'
    
    creative_totals = filtered_df.groupby('creative_name').agg(agg_dict).reset_index()
    
    valid_creatives = creative_totals[
        creative_totals['impressions'] >= filters['min_impressions']
    ]['creative_name']
    
    if 'conversions' in filtered_df.columns and filters['min_conversions'] > 0:
        valid_creatives_conv = creative_totals[
            creative_totals['conversions'] >= filters['min_conversions']
        ]['creative_name']
        valid_creatives = set(valid_creatives) & set(valid_creatives_conv)
    
    filtered_df = filtered_df[filtered_df['creative_name'].isin(valid_creatives)]
    
    return filtered_df


@st.cache_data
def compute_aggregated_creative_metrics(df):
    """
    Aggregate metrics at the creative level.
    """
    agg_dict = {
        'impressions': 'sum',
        'clicks': 'sum',
        'spend': 'sum',
        'platform': 'first',
        'campaign_name': 'first',
        'age_in_days': 'max',
        'date': 'nunique'
    }
    
    if 'conversions' in df.columns:
        agg_dict['conversions'] = 'sum'
    
    if 'revenue' in df.columns:
        agg_dict['revenue'] = 'sum'
    
    if 'purchases' in df.columns:
        agg_dict['purchases'] = 'sum'
    
    if 'add_to_carts' in df.columns:
        agg_dict['add_to_carts'] = 'sum'
    
    if 'view_content' in df.columns:
        agg_dict['view_content'] = 'sum'
    
    if 'page_views' in df.columns:
        agg_dict['page_views'] = 'sum'
    
    if 'format' in df.columns:
        agg_dict['format'] = 'first'
    
    if 'objective' in df.columns:
        agg_dict['objective'] = 'first'
    
    if 'objective_type' in df.columns:
        agg_dict['objective_type'] = 'first'
    
    creative_metrics = df.groupby('creative_name').agg(agg_dict).reset_index()
    
    creative_metrics.rename(columns={
        'age_in_days': 'age_in_days_max',
        'date': 'total_days_active'
    }, inplace=True)
    
    creative_metrics['CTR'] = np.where(
        creative_metrics['impressions'] > 0,
        creative_metrics['clicks'] / creative_metrics['impressions'],
        0
    )
    creative_metrics['CPC'] = np.where(
        creative_metrics['clicks'] > 0,
        creative_metrics['spend'] / creative_metrics['clicks'],
        0
    )
    creative_metrics['CPM'] = np.where(
        creative_metrics['impressions'] > 0,
        creative_metrics['spend'] / creative_metrics['impressions'] * 1000,
        0
    )
    
    if 'conversions' in creative_metrics.columns:
        creative_metrics['CVR'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['conversions'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['CPA'] = np.where(
            creative_metrics['conversions'] > 0,
            creative_metrics['spend'] / creative_metrics['conversions'],
            0
        )
    
    if 'revenue' in creative_metrics.columns:
        creative_metrics['ROAS'] = np.where(
            creative_metrics['spend'] > 0,
            creative_metrics['revenue'] / creative_metrics['spend'],
            0
        )
    
    if 'purchases' in creative_metrics.columns:
        creative_metrics['purchase_rate'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['purchases'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['cost_per_purchase'] = np.where(
            creative_metrics['purchases'] > 0,
            creative_metrics['spend'] / creative_metrics['purchases'],
            0
        )
    
    if 'add_to_carts' in creative_metrics.columns:
        creative_metrics['add_to_cart_rate'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['add_to_carts'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['cost_per_add_to_cart'] = np.where(
            creative_metrics['add_to_carts'] > 0,
            creative_metrics['spend'] / creative_metrics['add_to_carts'],
            0
        )
    
    if 'view_content' in creative_metrics.columns:
        creative_metrics['view_content_rate'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['view_content'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['cost_per_view_content'] = np.where(
            creative_metrics['view_content'] > 0,
            creative_metrics['spend'] / creative_metrics['view_content'],
            0
        )
    
    if 'page_views' in creative_metrics.columns:
        creative_metrics['page_view_rate'] = np.where(
            creative_metrics['clicks'] > 0,
            creative_metrics['page_views'] / creative_metrics['clicks'],
            0
        )
        creative_metrics['cost_per_page_view'] = np.where(
            creative_metrics['page_views'] > 0,
            creative_metrics['spend'] / creative_metrics['page_views'],
            0
        )
    
    return creative_metrics


def build_leaderboard(creative_metrics):
    """
    Build leaderboard with performance scores.
    """
    leaderboard = creative_metrics.copy()
    
    has_conversions = 'CVR' in leaderboard.columns
    
    leaderboard['CTR_percentile'] = leaderboard['CTR'].rank(pct=True)
    leaderboard['CPC_percentile'] = leaderboard['CPC'].rank(pct=True)
    
    if has_conversions:
        leaderboard['CVR_percentile'] = leaderboard['CVR'].rank(pct=True)
        leaderboard['score'] = (
            0.4 * leaderboard['CTR_percentile'] +
            0.3 * (1 - leaderboard['CPC_percentile']) +
            0.3 * leaderboard['CVR_percentile']
        )
    else:
        leaderboard['score'] = (
            0.6 * leaderboard['CTR_percentile'] +
            0.4 * (1 - leaderboard['CPC_percentile'])
        )
    
    leaderboard = leaderboard.sort_values('score', ascending=False)
    
    return leaderboard


def compute_fatigue_metrics_for_creative(df, creative_name):
    """
    Compute fatigue metrics for a specific creative.
    """
    creative_data = df[df['creative_name'] == creative_name].copy()
    creative_data = creative_data.sort_values('date')
    
    return creative_data


def fit_simple_adjusted_model(df, outcome_metric):
    """
    Fit a simple linear regression model to adjust for context.
    Returns model results with adjusted scores per creative.
    """
    creative_agg = compute_aggregated_creative_metrics(df)
    
    if outcome_metric not in creative_agg.columns:
        return None, "Selected metric not available in the data"
    
    model_df = creative_agg[creative_agg[outcome_metric] > 0].copy()
    
    if len(model_df) < 10:
        return None, f"Insufficient data: only {len(model_df)} creatives with {outcome_metric} > 0. Need at least 10."
    
    model_df['log_impressions'] = np.log1p(model_df['impressions'])
    model_df['log_spend'] = np.log1p(model_df['spend'])
    
    feature_cols = ['log_impressions', 'log_spend']
    
    if 'placement' in df.columns:
        placement_dummies = pd.get_dummies(model_df['platform'], prefix='platform')
        model_df = pd.concat([model_df, placement_dummies], axis=1)
        feature_cols.extend(placement_dummies.columns.tolist())
    else:
        platform_dummies = pd.get_dummies(model_df['platform'], prefix='platform')
        model_df = pd.concat([model_df, platform_dummies], axis=1)
        feature_cols.extend(platform_dummies.columns.tolist())
    
    if 'format' in model_df.columns:
        format_dummies = pd.get_dummies(model_df['format'], prefix='format')
        model_df = pd.concat([model_df, format_dummies], axis=1)
        feature_cols.extend(format_dummies.columns.tolist())
    
    X = model_df[feature_cols].fillna(0)
    y = model_df[outcome_metric]
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    model_df['predicted'] = predictions
    model_df['residual'] = y - predictions
    model_df['adjusted_score'] = model_df['residual']
    
    results = model_df[['creative_name', 'platform', 'campaign_name', outcome_metric, 
                       'predicted', 'adjusted_score', 'impressions', 'spend']].copy()
    results = results.sort_values('adjusted_score', ascending=False)
    
    return results, None


def show_welcome_screen():
    """
    Display welcome screen with instructions.
    """
    st.title("📊 Creative Performance Analysis")
    st.markdown("---")
    
    st.markdown("""
    ### Welcome to Creative Performance Analysis
    
    This app helps you analyze ad creative performance across platforms with:
    - 📈 Performance leaderboards and benchmarking
    - 🎯 Creative fatigue detection
    - 🔍 Context-adjusted performance modeling
    - 📉 Time series analysis
    
    #### Getting Started
    
    Upload a CSV file with your creative performance data. The file should contain:
    """)
    
    st.markdown("""
    **Required columns:**
    - `date` - Date of the performance data
    - `platform` - Advertising platform (e.g., Meta, Google, TikTok)
    - `campaign_name` - Campaign name
    - `creative_name` - Creative name
    - `impressions` - Number of impressions
    - `clicks` - Number of clicks
    - `spend` - Ad spend
    
    **Optional columns:**
    - `purchases` - Number of purchases
    - `add_to_carts` - Number of add-to-cart events
    - `view_content` - Number of content view events
    - `page_views` - Number of page views
    - `conversions` - Number of conversions (generic)
    - `revenue` - Revenue generated
    - `placement` - Ad placement (Feed, Stories, etc.)
    - `format` - Creative format (Image, Video, etc.)
    """)
    
    st.info("💡 **Tip:** Use the sidebar to upload your CSV file and start analyzing!")


def main():
    st.set_page_config(
        page_title="Creative Performance Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.sidebar.title("📊 Creative Analytics")
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("📥 Download CSV Template"):
        st.markdown("""
        Download the template to see the expected CSV format with all required and optional columns.
        """)
        
        with open('creative_performance_template.csv', 'rb') as f:
            template_data = f.read()
        
        st.download_button(
            label="⬇️ Download Template",
            data=template_data,
            file_name="creative_performance_template.csv",
            mime="text/csv",
            help="Download a sample CSV file showing the expected format"
        )
    
    st.sidebar.markdown("---")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Creative Performance CSV",
        type=['csv'],
        help="Upload a CSV file with creative performance data"
    )
    
    if uploaded_file is None:
        show_welcome_screen()
        return
    
    df = load_and_prepare_data(uploaded_file)
    
    if df is None:
        return
    
    st.sidebar.success(f"✅ Loaded {len(df):,} rows")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("🔍 Filters")
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    if isinstance(date_range, tuple) and len(date_range) == 2:
        date_range_filter = date_range
    else:
        date_range_filter = (min_date, max_date)
    
    all_platforms = sorted(df['platform'].unique().tolist())
    selected_platforms = st.sidebar.multiselect(
        "Platform",
        options=all_platforms,
        default=all_platforms
    )
    
    all_campaigns = sorted(df['campaign_name'].unique().tolist())
    selected_campaigns = st.sidebar.multiselect(
        "Campaign",
        options=all_campaigns,
        default=all_campaigns
    )
    
    selected_objectives = None
    selected_objective_type = "All"
    
    if 'objective' in df.columns:
        all_objectives = sorted([o for o in df['objective'].dropna().unique().tolist()])
        selected_objectives = st.sidebar.multiselect(
            "Objective",
            options=all_objectives,
            default=all_objectives
        )
    
    if 'objective_type' in df.columns:
        objective_type_options = ["All", "Awareness", "Conversion", "Other"]
        selected_objective_type = st.sidebar.selectbox(
            "Objective Type (Awareness vs Conversion)",
            options=objective_type_options,
            index=0
        )
    
    min_impressions = st.sidebar.slider(
        "Min Impressions per Creative",
        min_value=0,
        max_value=50000,
        value=5000,
        step=1000
    )
    
    has_conversions = 'conversions' in df.columns
    has_any_conversion_metrics = (
        has_conversions or
        'purchases' in df.columns or
        'add_to_carts' in df.columns or
        'view_content' in df.columns or
        'page_views' in df.columns or
        'revenue' in df.columns
    )
    
    if has_conversions:
        min_conversions = st.sidebar.slider(
            "Min Conversions per Creative",
            min_value=0,
            max_value=100,
            value=10,
            step=5
        )
    else:
        min_conversions = 0
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Metrics")
    
    available_kpis = ['CTR', 'CPC', 'CPM']
    if has_conversions:
        available_kpis.extend(['CVR', 'CPA'])
    if 'purchases' in df.columns:
        available_kpis.extend(['purchase_rate', 'cost_per_purchase'])
    if 'add_to_carts' in df.columns:
        available_kpis.extend(['add_to_cart_rate', 'cost_per_add_to_cart'])
    if 'view_content' in df.columns:
        available_kpis.extend(['view_content_rate', 'cost_per_view_content'])
    if 'page_views' in df.columns:
        available_kpis.extend(['page_view_rate', 'cost_per_page_view'])
    if 'revenue' in df.columns:
        available_kpis.append('ROAS')
    
    selected_kpi = st.sidebar.selectbox(
        "Primary KPI",
        options=available_kpis,
        index=0
    )
    
    if has_any_conversion_metrics:
        show_conversion_metrics = st.sidebar.checkbox(
            "Show Conversion Metrics (Directional)",
            value=True
        )
    else:
        show_conversion_metrics = False
    
    filters = {
        'date_range': date_range_filter,
        'platforms': selected_platforms if selected_platforms else all_platforms,
        'campaigns': selected_campaigns if selected_campaigns else all_campaigns,
        'objectives': selected_objectives if selected_objectives else None,
        'objective_type': selected_objective_type,
        'min_impressions': min_impressions,
        'min_conversions': min_conversions
    }
    
    filtered_df = apply_global_filters(df, filters)
    
    if len(filtered_df) == 0:
        st.warning("⚠️ No data matches the current filters. Please adjust your filter settings.")
        return
    
    st.sidebar.info(f"📊 {len(filtered_df):,} rows after filtering")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🏆 Creative Leaderboard",
        "📉 Creative Detail & Fatigue",
        "🤖 Model & Insights"
    ])
    
    with tab1:
        st.header("Performance Overview")
        
        st.info("💡 **Note:** Conversion-based metrics (conversions, CPA, ROAS) are platform-attributed and should be treated as directional, not as perfect ground truth. CTR/CPC/CVR are generally more stable for creative comparison.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_spend = filtered_df['spend'].sum()
        total_impressions = filtered_df['impressions'].sum()
        total_clicks = filtered_df['clicks'].sum()
        total_conversions = filtered_df['conversions'].sum() if has_conversions else 0
        total_revenue = filtered_df['revenue'].sum() if 'revenue' in filtered_df.columns else 0
        
        with col1:
            st.metric("Total Spend", f"${total_spend:,.2f}")
        with col2:
            st.metric("Total Impressions", f"{total_impressions:,.0f}")
        with col3:
            st.metric("Total Clicks", f"{total_clicks:,.0f}")
        with col4:
            if has_conversions:
                st.metric("Total Conversions", f"{total_conversions:,.0f}")
            else:
                st.metric("Total Conversions", "N/A")
        with col5:
            if 'revenue' in filtered_df.columns:
                st.metric("Total Revenue", f"${total_revenue:,.2f}")
            else:
                st.metric("Total Revenue", "N/A")
        
        st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        overall_ctr = total_clicks / total_impressions if total_impressions > 0 else 0
        overall_cpc = total_spend / total_clicks if total_clicks > 0 else 0
        overall_cpm = total_spend / total_impressions * 1000 if total_impressions > 0 else 0
        overall_cvr = total_conversions / total_clicks if total_clicks > 0 and has_conversions else 0
        overall_cpa = total_spend / total_conversions if total_conversions > 0 and has_conversions else 0
        overall_roas = total_revenue / total_spend if total_spend > 0 and 'revenue' in filtered_df.columns else 0
        
        with col1:
            st.metric("Overall CTR", f"{overall_ctr:.3%}")
        with col2:
            st.metric("Overall CPC", f"${overall_cpc:.2f}")
        with col3:
            st.metric("Overall CPM", f"${overall_cpm:.2f}")
        with col4:
            if has_conversions:
                st.metric("Overall CVR", f"{overall_cvr:.3%}")
            else:
                st.metric("Overall CVR", "N/A")
        with col5:
            if has_conversions:
                st.metric("Overall CPA", f"${overall_cpa:.2f}")
            elif 'revenue' in filtered_df.columns:
                st.metric("Overall ROAS", f"{overall_roas:.2f}x")
            else:
                st.metric("Overall CPA", "N/A")
        
        st.markdown("---")
        
        st.subheader(f"{selected_kpi} Over Time")
        agg_dict_time = {
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum'
        }
        if has_conversions:
            agg_dict_time['conversions'] = 'sum'
        if 'revenue' in filtered_df.columns:
            agg_dict_time['revenue'] = 'sum'
        if 'purchases' in filtered_df.columns:
            agg_dict_time['purchases'] = 'sum'
        if 'add_to_carts' in filtered_df.columns:
            agg_dict_time['add_to_carts'] = 'sum'
        if 'view_content' in filtered_df.columns:
            agg_dict_time['view_content'] = 'sum'
        if 'page_views' in filtered_df.columns:
            agg_dict_time['page_views'] = 'sum'
        
        time_series = filtered_df.groupby('date').agg(agg_dict_time).reset_index()
        
        time_series['CTR'] = np.where(
            time_series['impressions'] > 0,
            time_series['clicks'] / time_series['impressions'],
            0
        )
        time_series['CPC'] = np.where(
            time_series['clicks'] > 0,
            time_series['spend'] / time_series['clicks'],
            0
        )
        time_series['CPM'] = np.where(
            time_series['impressions'] > 0,
            time_series['spend'] / time_series['impressions'] * 1000,
            0
        )
        
        if has_conversions:
            time_series['CVR'] = np.where(
                time_series['clicks'] > 0,
                time_series['conversions'] / time_series['clicks'],
                0
            )
            time_series['CPA'] = np.where(
                time_series['conversions'] > 0,
                time_series['spend'] / time_series['conversions'],
                0
            )
        
        if 'revenue' in filtered_df.columns:
            time_series['ROAS'] = np.where(
                time_series['spend'] > 0,
                time_series['revenue'] / time_series['spend'],
                0
            )
        
        if 'purchases' in filtered_df.columns:
            time_series['purchase_rate'] = np.where(
                time_series['clicks'] > 0,
                time_series['purchases'] / time_series['clicks'],
                0
            )
            time_series['cost_per_purchase'] = np.where(
                time_series['purchases'] > 0,
                time_series['spend'] / time_series['purchases'],
                0
            )
        
        if 'add_to_carts' in filtered_df.columns:
            time_series['add_to_cart_rate'] = np.where(
                time_series['clicks'] > 0,
                time_series['add_to_carts'] / time_series['clicks'],
                0
            )
            time_series['cost_per_add_to_cart'] = np.where(
                time_series['add_to_carts'] > 0,
                time_series['spend'] / time_series['add_to_carts'],
                0
            )
        
        if 'view_content' in filtered_df.columns:
            time_series['view_content_rate'] = np.where(
                time_series['clicks'] > 0,
                time_series['view_content'] / time_series['clicks'],
                0
            )
            time_series['cost_per_view_content'] = np.where(
                time_series['view_content'] > 0,
                time_series['spend'] / time_series['view_content'],
                0
            )
        
        if 'page_views' in filtered_df.columns:
            time_series['page_view_rate'] = np.where(
                time_series['clicks'] > 0,
                time_series['page_views'] / time_series['clicks'],
                0
            )
            time_series['cost_per_page_view'] = np.where(
                time_series['page_views'] > 0,
                time_series['spend'] / time_series['page_views'],
                0
            )
        
        fig = px.line(
            time_series,
            x='date',
            y=selected_kpi,
            title=f"{selected_kpi} Trend Over Time",
            labels={'date': 'Date', selected_kpi: selected_kpi}
        )
        if selected_kpi in RATE_METRICS:
            fig.update_yaxes(tickformat=".2%")
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{selected_kpi} by Platform")
            agg_dict_platform = {
                'impressions': 'sum',
                'clicks': 'sum',
                'spend': 'sum'
            }
            if has_conversions:
                agg_dict_platform['conversions'] = 'sum'
            if 'revenue' in filtered_df.columns:
                agg_dict_platform['revenue'] = 'sum'
            if 'purchases' in filtered_df.columns:
                agg_dict_platform['purchases'] = 'sum'
            if 'add_to_carts' in filtered_df.columns:
                agg_dict_platform['add_to_carts'] = 'sum'
            if 'view_content' in filtered_df.columns:
                agg_dict_platform['view_content'] = 'sum'
            if 'page_views' in filtered_df.columns:
                agg_dict_platform['page_views'] = 'sum'
            
            platform_metrics = filtered_df.groupby('platform').agg(agg_dict_platform).reset_index()
            
            platform_metrics['CTR'] = np.where(
                platform_metrics['impressions'] > 0,
                platform_metrics['clicks'] / platform_metrics['impressions'],
                0
            )
            platform_metrics['CPC'] = np.where(
                platform_metrics['clicks'] > 0,
                platform_metrics['spend'] / platform_metrics['clicks'],
                0
            )
            platform_metrics['CPM'] = np.where(
                platform_metrics['impressions'] > 0,
                platform_metrics['spend'] / platform_metrics['impressions'] * 1000,
                0
            )
            
            if has_conversions:
                platform_metrics['CVR'] = np.where(
                    platform_metrics['clicks'] > 0,
                    platform_metrics['conversions'] / platform_metrics['clicks'],
                    0
                )
                platform_metrics['CPA'] = np.where(
                    platform_metrics['conversions'] > 0,
                    platform_metrics['spend'] / platform_metrics['conversions'],
                    0
                )
            
            if 'revenue' in filtered_df.columns:
                platform_metrics['ROAS'] = np.where(
                    platform_metrics['spend'] > 0,
                    platform_metrics['revenue'] / platform_metrics['spend'],
                    0
                )
            
            if 'purchases' in filtered_df.columns:
                platform_metrics['purchase_rate'] = np.where(
                    platform_metrics['clicks'] > 0,
                    platform_metrics['purchases'] / platform_metrics['clicks'],
                    0
                )
                platform_metrics['cost_per_purchase'] = np.where(
                    platform_metrics['purchases'] > 0,
                    platform_metrics['spend'] / platform_metrics['purchases'],
                    0
                )
            
            if 'add_to_carts' in filtered_df.columns:
                platform_metrics['add_to_cart_rate'] = np.where(
                    platform_metrics['clicks'] > 0,
                    platform_metrics['add_to_carts'] / platform_metrics['clicks'],
                    0
                )
                platform_metrics['cost_per_add_to_cart'] = np.where(
                    platform_metrics['add_to_carts'] > 0,
                    platform_metrics['spend'] / platform_metrics['add_to_carts'],
                    0
                )
            
            if 'view_content' in filtered_df.columns:
                platform_metrics['view_content_rate'] = np.where(
                    platform_metrics['clicks'] > 0,
                    platform_metrics['view_content'] / platform_metrics['clicks'],
                    0
                )
                platform_metrics['cost_per_view_content'] = np.where(
                    platform_metrics['view_content'] > 0,
                    platform_metrics['spend'] / platform_metrics['view_content'],
                    0
                )
            
            if 'page_views' in filtered_df.columns:
                platform_metrics['page_view_rate'] = np.where(
                    platform_metrics['clicks'] > 0,
                    platform_metrics['page_views'] / platform_metrics['clicks'],
                    0
                )
                platform_metrics['cost_per_page_view'] = np.where(
                    platform_metrics['page_views'] > 0,
                    platform_metrics['spend'] / platform_metrics['page_views'],
                    0
                )
            
            fig = px.bar(
                platform_metrics,
                x='platform',
                y=selected_kpi,
                title=f"{selected_kpi} by Platform",
                labels={'platform': 'Platform', selected_kpi: selected_kpi},
                color=selected_kpi,
                color_continuous_scale='Blues'
            )
            if selected_kpi in RATE_METRICS:
                fig.update_yaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader(f"{selected_kpi} Distribution")
            creative_metrics = compute_aggregated_creative_metrics(filtered_df)
            
            fig = px.histogram(
                creative_metrics,
                x=selected_kpi,
                nbins=30,
                title=f"Distribution of {selected_kpi} Across Creatives",
                labels={selected_kpi: selected_kpi, 'count': 'Number of Creatives'}
            )
            if selected_kpi in RATE_METRICS:
                fig.update_xaxes(tickformat=".2%")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("🏆 Creative Leaderboard")
        
        creative_metrics = compute_aggregated_creative_metrics(filtered_df)
        leaderboard = build_leaderboard(creative_metrics)
        
        st.subheader(f"Top Performing Creatives ({len(leaderboard)} total)")
        
        display_cols = ['creative_name', 'platform', 'campaign_name']
        
        if 'objective' in leaderboard.columns:
            display_cols.append('objective')
        
        if 'format' in leaderboard.columns:
            display_cols.append('format')
        
        display_cols.extend(['impressions', 'clicks', 'spend', 'CTR', 'CPC', 'CPM'])
        
        if has_conversions and show_conversion_metrics:
            display_cols.extend(['conversions', 'CVR', 'CPA'])
        
        if 'revenue' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['ROAS'])
        
        if 'purchases' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['purchases', 'purchase_rate', 'cost_per_purchase'])
        
        if 'add_to_carts' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['add_to_carts', 'add_to_cart_rate', 'cost_per_add_to_cart'])
        
        if 'view_content' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['view_content', 'view_content_rate', 'cost_per_view_content'])
        
        if 'page_views' in leaderboard.columns and show_conversion_metrics:
            display_cols.extend(['page_views', 'page_view_rate', 'cost_per_page_view'])
        
        display_cols.extend(['age_in_days_max', 'total_days_active', 'score'])
        
        display_df = leaderboard[display_cols].copy()
        
        display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.3%}")
        display_df['CPC'] = display_df['CPC'].apply(lambda x: f"${x:.2f}")
        display_df['CPM'] = display_df['CPM'].apply(lambda x: f"${x:.2f}")
        
        if 'CVR' in display_df.columns:
            display_df['CVR'] = display_df['CVR'].apply(lambda x: f"{x:.3%}")
        if 'CPA' in display_df.columns:
            display_df['CPA'] = display_df['CPA'].apply(lambda x: f"${x:.2f}")
        if 'ROAS' in display_df.columns:
            display_df['ROAS'] = display_df['ROAS'].apply(lambda x: f"{x:.2f}x")
        
        display_df['score'] = display_df['score'].apply(lambda x: f"{x:.3f}")
        display_df['impressions'] = display_df['impressions'].apply(lambda x: f"{x:,.0f}")
        display_df['clicks'] = display_df['clicks'].apply(lambda x: f"{x:,.0f}")
        display_df['spend'] = display_df['spend'].apply(lambda x: f"${x:,.2f}")
        
        if 'conversions' in display_df.columns:
            display_df['conversions'] = display_df['conversions'].apply(lambda x: f"{x:,.0f}")
        
        if 'purchases' in display_df.columns:
            display_df['purchases'] = display_df['purchases'].apply(lambda x: f"{x:,.0f}")
        if 'purchase_rate' in display_df.columns:
            display_df['purchase_rate'] = display_df['purchase_rate'].apply(lambda x: f"{x:.3%}")
        if 'cost_per_purchase' in display_df.columns:
            display_df['cost_per_purchase'] = display_df['cost_per_purchase'].apply(lambda x: f"${x:.2f}")
        
        if 'add_to_carts' in display_df.columns:
            display_df['add_to_carts'] = display_df['add_to_carts'].apply(lambda x: f"{x:,.0f}")
        if 'add_to_cart_rate' in display_df.columns:
            display_df['add_to_cart_rate'] = display_df['add_to_cart_rate'].apply(lambda x: f"{x:.3%}")
        if 'cost_per_add_to_cart' in display_df.columns:
            display_df['cost_per_add_to_cart'] = display_df['cost_per_add_to_cart'].apply(lambda x: f"${x:.2f}")
        
        if 'view_content' in display_df.columns:
            display_df['view_content'] = display_df['view_content'].apply(lambda x: f"{x:,.0f}")
        if 'view_content_rate' in display_df.columns:
            display_df['view_content_rate'] = display_df['view_content_rate'].apply(lambda x: f"{x:.3%}")
        if 'cost_per_view_content' in display_df.columns:
            display_df['cost_per_view_content'] = display_df['cost_per_view_content'].apply(lambda x: f"${x:.2f}")
        
        if 'page_views' in display_df.columns:
            display_df['page_views'] = display_df['page_views'].apply(lambda x: f"{x:,.0f}")
        if 'page_view_rate' in display_df.columns:
            display_df['page_view_rate'] = display_df['page_view_rate'].apply(lambda x: f"{x:.3%}")
        if 'cost_per_page_view' in display_df.columns:
            display_df['cost_per_page_view'] = display_df['cost_per_page_view'].apply(lambda x: f"${x:.2f}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        st.markdown("---")
        st.subheader("Creative Performance Scatter Plot")
        
        y_axis_metric = 'CVR' if has_conversions and 'CVR' in leaderboard.columns else 'CTR'
        
        fig = px.scatter(
            leaderboard,
            x='CPC',
            y=y_axis_metric,
            size='spend',
            color='score',
            hover_data=['creative_name', 'campaign_name', 'platform', 'impressions', 'clicks'],
            title=f"Creative Performance: {y_axis_metric} vs CPC (size = spend)",
            labels={'CPC': 'Cost Per Click ($)', y_axis_metric: y_axis_metric},
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("📉 Creative Detail & Fatigue Analysis")
        
        creative_list = sorted(filtered_df['creative_name'].unique().tolist())
        
        if len(creative_list) == 0:
            st.warning("No creatives available with current filters.")
            return
        
        selected_creative = st.selectbox(
            "Select Creative to Analyze",
            options=creative_list,
            index=0
        )
        
        creative_data = compute_fatigue_metrics_for_creative(filtered_df, selected_creative)
        creative_summary = compute_aggregated_creative_metrics(
            filtered_df[filtered_df['creative_name'] == selected_creative]
        ).iloc[0]
        
        st.markdown("---")
        st.subheader("Creative Summary")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Platform", creative_summary['platform'])
        with col2:
            st.metric("Campaign", creative_summary['campaign_name'])
        with col3:
            st.metric("Total Spend", f"${creative_summary['spend']:,.2f}")
        with col4:
            st.metric("Impressions", f"{creative_summary['impressions']:,.0f}")
        with col5:
            st.metric("Days Active", f"{creative_summary['total_days_active']:.0f}")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("CTR", f"{creative_summary['CTR']:.3%}")
        with col2:
            st.metric("CPC", f"${creative_summary['CPC']:.2f}")
        with col3:
            if has_conversions:
                st.metric("CVR", f"{creative_summary['CVR']:.3%}")
            else:
                st.metric("CVR", "N/A")
        with col4:
            if has_conversions:
                st.metric("CPA", f"${creative_summary['CPA']:.2f}")
            else:
                st.metric("CPA", "N/A")
        with col5:
            if 'ROAS' in creative_summary:
                st.metric("ROAS", f"{creative_summary['ROAS']:.2f}x")
            else:
                st.metric("ROAS", "N/A")
        
        if 'objective' in creative_summary:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("Objective", creative_summary['objective'])
            if 'objective_type' in creative_summary:
                with col2:
                    st.metric("Objective Type", creative_summary['objective_type'])
        
        st.markdown("---")
        st.subheader("Fatigue Analysis")
        
        fatigue_kpi_options = ['CTR', 'CPC']
        if has_conversions:
            fatigue_kpi_options.append('CVR')
        if 'purchase_rate' in creative_summary:
            fatigue_kpi_options.append('purchase_rate')
        if 'add_to_cart_rate' in creative_summary:
            fatigue_kpi_options.append('add_to_cart_rate')
        if 'view_content_rate' in creative_summary:
            fatigue_kpi_options.append('view_content_rate')
        if 'page_view_rate' in creative_summary:
            fatigue_kpi_options.append('page_view_rate')
        
        fatigue_kpi = st.selectbox(
            "Select KPI for Fatigue Analysis",
            options=fatigue_kpi_options,
            index=0
        )
        
        if len(creative_data) >= 3:
            age_days = creative_data['age_in_days'].values
            kpi_values = creative_data[fatigue_kpi].values
            
            valid_indices = ~np.isnan(kpi_values) & ~np.isinf(kpi_values)
            age_days_clean = age_days[valid_indices]
            kpi_values_clean = kpi_values[valid_indices]
            
            if len(age_days_clean) >= 3:
                coeffs = np.polyfit(age_days_clean, kpi_values_clean, 1)
                slope = coeffs[0]
                trend_line = coeffs[0] * age_days_clean + coeffs[1]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=creative_data['date'],
                    y=creative_data[fatigue_kpi],
                    mode='lines+markers',
                    name=f'Actual {fatigue_kpi}',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=creative_data['date'].values[valid_indices],
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f"{fatigue_kpi} Over Time for {selected_creative}",
                    xaxis_title="Date",
                    yaxis_title=fatigue_kpi,
                    hovermode='x unified'
                )
                if fatigue_kpi in RATE_METRICS:
                    fig.update_yaxes(tickformat=".2%")
                
                st.plotly_chart(fig, use_container_width=True)
                
                rate_metrics = ['CTR', 'CVR', 'purchase_rate', 'add_to_cart_rate', 'view_content_rate', 'page_view_rate']
                fatigue_threshold = -0.0001 if fatigue_kpi in rate_metrics else 0.01
                min_days_for_fatigue = 7
                min_impressions_for_fatigue = 10000
                
                total_impressions = creative_summary['impressions']
                total_days = creative_summary['total_days_active']
                
                is_fatiguing = (
                    slope < fatigue_threshold and
                    total_days >= min_days_for_fatigue and
                    total_impressions >= min_impressions_for_fatigue
                )
                
                if is_fatiguing:
                    st.error(f"🔴 **Likely Fatigue Detected** - {fatigue_kpi} is declining over time (slope: {slope:.6f})")
                else:
                    st.success(f"🟢 **No Clear Fatigue Signal** - {fatigue_kpi} is stable or improving (slope: {slope:.6f})")
            else:
                st.warning("Not enough valid data points to compute trend.")
        else:
            st.warning("Not enough data points for fatigue analysis (minimum 3 days required).")
        
        st.markdown("---")
        st.subheader(f"{fatigue_kpi} vs Cumulative Impressions")
        
        if len(creative_data) >= 3:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=creative_data['cumulative_impressions'],
                y=creative_data[fatigue_kpi],
                mode='lines+markers',
                name=f'{fatigue_kpi}',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ))
            
            cum_impr = creative_data['cumulative_impressions'].values
            kpi_vals = creative_data[fatigue_kpi].values
            
            valid_idx = ~np.isnan(kpi_vals) & ~np.isinf(kpi_vals)
            if np.sum(valid_idx) >= 3:
                coeffs_cum = np.polyfit(cum_impr[valid_idx], kpi_vals[valid_idx], 1)
                trend_cum = coeffs_cum[0] * cum_impr[valid_idx] + coeffs_cum[1]
                
                fig.add_trace(go.Scatter(
                    x=cum_impr[valid_idx],
                    y=trend_cum,
                    mode='lines',
                    name='Trend Line',
                    line=dict(color='red', width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=f"{fatigue_kpi} vs Cumulative Impressions",
                xaxis_title="Cumulative Impressions",
                yaxis_title=fatigue_kpi,
                hovermode='x unified'
            )
            if fatigue_kpi in RATE_METRICS:
                fig.update_yaxes(tickformat=".2%")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data points for cumulative impression analysis.")
    
    with tab4:
        st.header("🤖 Model & Insights")
        
        st.info("💡 This model adjusts for context (platform, format, scale) to identify creatives that over- or under-perform relative to their expected performance.")
        
        model_outcome_options = ['CTR']
        if has_conversions:
            model_outcome_options.append('CVR')
        if 'purchases' in df.columns:
            model_outcome_options.append('purchase_rate')
        if 'add_to_carts' in df.columns:
            model_outcome_options.append('add_to_cart_rate')
        if 'view_content' in df.columns:
            model_outcome_options.append('view_content_rate')
        if 'page_views' in df.columns:
            model_outcome_options.append('page_view_rate')
        
        model_outcome = st.selectbox(
            "Select Outcome Metric to Model",
            options=model_outcome_options,
            index=0
        )
        
        if st.button("Run Context-Adjusted Model"):
            with st.spinner("Fitting model..."):
                results, error = fit_simple_adjusted_model(filtered_df, model_outcome)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.success(f"✅ Model fitted successfully on {len(results)} creatives")
                    
                    st.markdown("---")
                    st.subheader("Top Over-Performing Creatives (Adjusted)")
                    
                    top_n = min(20, len(results))
                    top_results = results.head(top_n).copy()
                    
                    if model_outcome in RATE_METRICS:
                        top_results[model_outcome] = top_results[model_outcome].apply(lambda x: f"{x:.3%}")
                        top_results['predicted'] = top_results['predicted'].apply(lambda x: f"{x:.3%}")
                    else:
                        top_results[model_outcome] = top_results[model_outcome].apply(lambda x: f"{x:.4f}")
                        top_results['predicted'] = top_results['predicted'].apply(lambda x: f"{x:.4f}")
                    top_results['adjusted_score'] = top_results['adjusted_score'].apply(lambda x: f"{x:.4f}")
                    top_results['impressions'] = top_results['impressions'].apply(lambda x: f"{x:,.0f}")
                    top_results['spend'] = top_results['spend'].apply(lambda x: f"${x:,.2f}")
                    
                    st.dataframe(top_results, use_container_width=True, height=400)
                    
                    st.markdown("---")
                    st.subheader("Adjusted Performance Chart")
                    
                    chart_data = results.head(15).copy()
                    
                    fig = px.bar(
                        chart_data,
                        x='creative_name',
                        y='adjusted_score',
                        color='platform',
                        title=f"Top 15 Creatives by Adjusted {model_outcome} Score",
                        labels={'creative_name': 'Creative Name', 'adjusted_score': 'Adjusted Score'},
                        hover_data=['campaign_name', 'platform']
                    )
                    fig.update_xaxes(tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("---")
                    st.subheader("Natural Language Insights")
                    
                    top_10 = results.head(10)
                    platform_counts = top_10['platform'].value_counts()
                    
                    insights = []
                    
                    if len(platform_counts) > 0:
                        top_platform = platform_counts.index[0]
                        top_platform_count = platform_counts.iloc[0]
                        if top_platform_count >= 5:
                            insights.append(f"• **{top_platform}** dominates the top performers with {top_platform_count} creatives in the top 10.")
                        else:
                            insights.append(f"• Performance is relatively balanced across platforms.")
                    
                    if 'format' in results.columns:
                        format_counts = top_10['format'].value_counts()
                        if len(format_counts) > 0:
                            top_format = format_counts.index[0]
                            top_format_count = format_counts.iloc[0]
                            if top_format_count >= 5:
                                insights.append(f"• **{top_format}** format is over-represented in top performers ({top_format_count}/10).")
                    
                    avg_adjusted = results['adjusted_score'].mean()
                    top_10_avg = top_10['adjusted_score'].mean()
                    insights.append(f"• Top 10 creatives have an average adjusted score of {top_10_avg:.4f} vs overall average of {avg_adjusted:.4f}.")
                    
                    for insight in insights:
                        st.markdown(insight)


if __name__ == "__main__":
    main()
