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


import numpy as np

def classify_journey_role(
    row,
    ctr_median=None,
    cpc_median=None,
    cvr_median=None,
    intent_median=None,
    purchase_rate_median=None,
    cpa_median=None,
    # tunable multipliers
    cvr_boost=1.2,
    intent_boost=1.1,
    ctr_boost=1.1,
    cpc_discount=0.9, 
    cpa_discount=0.8,
    purchase_boost=1.2,
):
    """
    Classify creative into Engagement, Intent, or Conversion.

    Priority:
      1) Conversion  – strong CVR / purchase_rate and/or efficient CPA
      2) Intent      – strong micro-conversion activity vs peers
      3) Engagement  – strong CTR vs peers

    Falls back sensibly when medians or metrics are missing.
    """

    ctr = row.get("CTR", 0.0)
    cvr = row.get("CVR", 0.0)  # may not exist in your data
    purchase_rate = row.get("purchase_rate", 0.0)
    cpa = row.get("cost_per_purchase", None) or row.get("CPA", None)

    # ---- Intent metrics (mid-funnel) ----
    intent_metrics = []
    for metric in ["add_to_cart_rate", "view_content_rate", "page_view_rate"]:
        val = row.get(metric, 0.0)
        if val and val > 0:
            intent_metrics.append(val)
    avg_intent = float(np.mean(intent_metrics)) if intent_metrics else 0.0
    has_intent_metrics = avg_intent > 0

    # ---------------------------------------------------------
    # 1) Conversion: strong closers
    #    - high CVR vs median (if present)
    #    - OR high purchase_rate vs non-zero median
    #    - OR efficient CPA vs median
    # ---------------------------------------------------------
    strong_conversion = False

    # CVR-based signal (if CVR exists in your data)
    if cvr_median and cvr_median > 0:
        if cvr >= cvr_boost * cvr_median:
            strong_conversion = True

    # purchase_rate-based signal (works for your current dataset)
    if purchase_rate_median and purchase_rate_median > 0:
        if purchase_rate >= purchase_boost * purchase_rate_median:
            strong_conversion = True
    elif purchase_rate > 0:
        # if median is 0, ANY non-zero purchase_rate is meaningful
        strong_conversion = True

    # CPA-based signal (lower is better)
    if cpa is not None and cpa_median and cpa_median > 0:
        if cpa <= cpa_discount * cpa_median:
            strong_conversion = True

    if strong_conversion:
        return "Conversion"

    # ---------------------------------------------------------
    # 2) Intent: strong mid-funnel signals
    # ---------------------------------------------------------
    strong_intent = False
    if has_intent_metrics and intent_median and intent_median > 0:
        if avg_intent >= intent_boost * intent_median:
            strong_intent = True

    if strong_intent:
        return "Intent"

    # ---------------------------------------------------------
    # 3) Engagement: strong cost-efficiency (CPC) + optional CTR
    # ---------------------------------------------------------
    strong_engagement = False

    # CPC-based signal (lower = better). e.g. <= 90% of median CPC
    if cpc_median and cpc_median > 0 and "CPC" in row:
        if row["CPC"] <= cpc_discount * cpc_median:
            strong_engagement = True

    # Optional backup: if CTR is also clearly strong, treat as Engagement
    if (not strong_engagement) and ctr_median and ctr_median > 0:
        if ctr >= ctr_boost * ctr_median:
            strong_engagement = True

    if strong_engagement:
        return "Engagement"

    # ---------------------------------------------------------
    # 4) Fallbacks when medians are missing or everything is meh
    # ---------------------------------------------------------
    if purchase_rate > 0:
        return "Conversion"
    if has_intent_metrics:
        return "Intent"
    if ctr > 0:
        return "Engagement"

    return "Engagement"



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

        def clean_numeric(series):
            # Convert to string, remove $ and commas, strip spaces
            return (
                series.astype(str)
                .str.replace(r'[\$,]', '', regex=True)
                .str.strip()
            )

        for col in numeric_cols:
            df[col] = clean_numeric(df[col])
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        for col in optional_numeric:
            if col in df.columns:
                df[col] = clean_numeric(df[col])
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

    if filters.get('topics') is not None and 'topic' in df.columns:
        all_topics_in_data = set(df['topic'].dropna().unique())
        if set(filters['topics']) != all_topics_in_data:
            filtered_df = filtered_df[filtered_df['topic'].isin(filters['topics'])]

    # --- NEW: format filter ---
    if filters.get('formats') is not None and 'format' in filtered_df.columns:
        all_formats_in_data = set(df['format'].dropna().unique())
        if set(filters['formats']) != all_formats_in_data:
            filtered_df = filtered_df[filtered_df['format'].isin(filters['formats'])]

    # --- NEW: placement filter ---
    if filters.get('placements') is not None and 'placement' in filtered_df.columns:
        all_place_in_data = set(df['placement'].dropna().unique())
        if set(filters['placements']) != all_place_in_data:
            filtered_df = filtered_df[filtered_df['placement'].isin(filters['placements'])]

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

    if 'topic' in df.columns:
        agg_dict['topic'] = 'first'

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

    # ---- Journey role thresholds ----
    # CTR median (only for creatives with impressions)
    ctr_median = creative_metrics.loc[
        creative_metrics['impressions'] > 0, 'CTR'
    ].median() if len(creative_metrics) > 0 else 0

    # CVR median (only > 0 so we don't get dragged down by zeros)
    cvr_median = None
    if 'CVR' in creative_metrics.columns:
        cvr_vals = creative_metrics.loc[creative_metrics['CVR'] > 0, 'CVR']
        cvr_median = cvr_vals.median() if len(cvr_vals) > 0 else None

    # CPC median (only > 0)
    cpc_median = None
    if 'CPC' in creative_metrics.columns:
        cpc_vals = creative_metrics.loc[creative_metrics['CPC'] > 0, 'CPC']
        cpc_median = cpc_vals.median() if len(cpc_vals) > 0 else None

    # Intent median based on micro-conversion rates
    intent_scores = []
    intent_cols = [
        col for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']
        if col in creative_metrics.columns
    ]

    for _, row in creative_metrics.iterrows():
        vals = [row[col] for col in intent_cols if row.get(col, 0) > 0]
        if vals:
            intent_scores.append(np.mean(vals))

    intent_median = np.median(intent_scores) if intent_scores else None

    # Apply journey role classification
    creative_metrics['journey_role'] = creative_metrics.apply(
        lambda row: classify_journey_role(
            row=row,
            ctr_median=ctr_median,
            cvr_median=cvr_median,
            intent_median=intent_median,
            cpc_median=cpc_median,
        ),
        axis=1
    )

    return creative_metrics


def build_leaderboard(creative_metrics):
    """
    Build leaderboard with journey-aware performance scores.
    Scoring weights are adjusted based on each creative's journey_role:
    - Engagement: Emphasize CTR/CPC
    - Intent: Emphasize micro-conversion rates
    - Conversion: Emphasize CVR/CPA
    """
    leaderboard = creative_metrics.copy()

    has_conversions = 'CVR' in leaderboard.columns
    has_intent_metrics = any(col in leaderboard.columns for col in 
                            ['add_to_cart_rate', 'view_content_rate', 'page_view_rate'])

    leaderboard['CTR_percentile'] = leaderboard['CTR'].rank(pct=True)
    leaderboard['CPC_percentile'] = leaderboard['CPC'].rank(pct=True)

    if has_conversions:
        leaderboard['CVR_percentile'] = leaderboard['CVR'].rank(pct=True)
    
    if has_intent_metrics:
        intent_cols = [col for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate'] 
                      if col in leaderboard.columns]
        leaderboard['intent_avg'] = leaderboard[intent_cols].mean(axis=1)
        leaderboard['intent_percentile'] = leaderboard['intent_avg'].rank(pct=True)
    
    def compute_journey_score(row):
        journey_role = row.get('journey_role', 'Engagement')
        ctr_p = row.get('CTR_percentile', 0.5)
        cpc_p = row.get('CPC_percentile', 0.5)
        cvr_p = row.get('CVR_percentile', 0.5) if has_conversions else 0.5
        intent_p = row.get('intent_percentile', 0.5) if has_intent_metrics else 0.5
        
        if journey_role == "Engagement":
            return 0.6 * ctr_p + 0.4 * (1 - cpc_p)
        elif journey_role == "Intent":
            if has_intent_metrics:
                return 0.3 * ctr_p + 0.2 * (1 - cpc_p) + 0.5 * intent_p
            else:
                return 0.5 * ctr_p + 0.5 * (1 - cpc_p)
        elif journey_role == "Conversion":
            if has_conversions:
                return 0.25 * ctr_p + 0.15 * (1 - cpc_p) + 0.6 * cvr_p
            else:
                return 0.6 * ctr_p + 0.4 * (1 - cpc_p)
        else:
            if has_conversions:
                return 0.4 * ctr_p + 0.3 * (1 - cpc_p) + 0.3 * cvr_p
            else:
                return 0.6 * ctr_p + 0.4 * (1 - cpc_p)
    
    leaderboard['score'] = leaderboard.apply(compute_journey_score, axis=1)
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
        placement_dummies = pd.get_dummies(model_df['placement'], prefix='placement')
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

    This app helps you analyze ad creative performance across platforms using a **3-layer journey framework**:
    
    - 📢 **Engagement Layer** (CTR, CPC) — Top-of-funnel attention drivers
    - 🛒 **Intent Layer** (micro-conversions) — Mid-funnel purchase intent builders
    - 💰 **Conversion Layer** (CVR, CPA, ROAS) — Bottom-of-funnel closers
    
    Each creative is automatically classified by its funnel role, helping you understand which assets drive awareness vs. which drive sales.

    #### Key Features
    - 📈 Journey-aware performance leaderboards
    - 🎯 Creative fatigue detection
    - 📊 Portfolio spend distribution by funnel layer
    - 🏷️ Topic insights with layer breakdowns

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

    selected_formats = None
    if 'format' in df.columns:
        all_formats = sorted([f for f in df['format'].dropna().unique().tolist()])
        if all_formats:
            selected_formats = st.sidebar.multiselect(
                "Format",
                options=all_formats,
                default=all_formats
            )

    selected_placements = None
    if 'placement' in df.columns:
        all_placements = sorted([p for p in df['placement'].dropna().unique().tolist()])
        if all_placements:
            selected_placements = st.sidebar.multiselect(
                "Placement",
                options=all_placements,
                default=all_placements
            )

    selected_topics = None
    if 'topic' in df.columns:
        all_topics = sorted([t for t in df['topic'].dropna().unique().tolist()])
        if len(all_topics) > 0:
            selected_topics = st.sidebar.multiselect(
                "Topic",
                options=all_topics,
                default=all_topics
            )
        else:
            st.sidebar.info("ℹ️ No topics found in data. Add a 'topic' column to enable topic filtering.")

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
        'topics': selected_topics if selected_topics else None,
        'formats': selected_formats if selected_formats else None,       # NEW
        'placements': selected_placements if selected_placements else None,  # NEW
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
        "🏷️ Topic Insights"
    ])

    with tab1:
        st.header("Performance Overview")

        st.info("💡 **3-Layer Analysis Framework:** We recommend analyzing creatives across three layers: **Engagement** (CTR/CPC), **Intent** (micro-conversions like add-to-carts), and **Conversion** (CVR/CPA/ROAS). Each creative is automatically classified by its strongest layer. Conversion metrics are platform-attributed and should be treated as directional.")

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
        
        st.subheader("🎯 Portfolio View by Journey Layer")
        st.caption("Creatives are classified into three layers: **Engagement** (top-of-funnel, drives clicks), **Intent** (mid-funnel, drives micro-conversions), and **Conversion** (bottom-of-funnel, drives sales).")
        
        creative_metrics = compute_aggregated_creative_metrics(filtered_df)
        
        journey_summary = creative_metrics.groupby('journey_role').agg({
            'spend': 'sum',
            'impressions': 'sum',
            'clicks': 'sum',
            'creative_name': 'count'
        }).reset_index()
        journey_summary.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
        
        for role in ['Engagement', 'Intent', 'Conversion']:
            if role not in journey_summary['journey_role'].values:
                journey_summary = pd.concat([journey_summary, pd.DataFrame({
                    'journey_role': [role], 'spend': [0], 'impressions': [0], 
                    'clicks': [0], 'num_creatives': [0]
                })], ignore_index=True)
        
        journey_summary = journey_summary.sort_values(
            'journey_role', 
            key=lambda x: x.map({'Engagement': 0, 'Intent': 1, 'Conversion': 2})
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            eng_data = journey_summary[journey_summary['journey_role'] == 'Engagement'].iloc[0] if len(journey_summary[journey_summary['journey_role'] == 'Engagement']) > 0 else None
            st.markdown("##### 📢 Engagement Layer")
            if eng_data is not None and eng_data['num_creatives'] > 0:
                eng_creatives = creative_metrics[creative_metrics['journey_role'] == 'Engagement']
                avg_ctr = eng_creatives['CTR'].mean()
                avg_cpc = eng_creatives['CPC'].mean()
                st.metric("Creatives", f"{int(eng_data['num_creatives'])}")
                st.metric("Avg CTR", f"{avg_ctr:.3%}")
                st.metric("Avg CPC", f"${avg_cpc:.2f}")
                st.metric("Spend", f"${eng_data['spend']:,.0f}")
            else:
                st.info("No engagement creatives")
        
        with col2:
            int_data = journey_summary[journey_summary['journey_role'] == 'Intent'].iloc[0] \
                if len(journey_summary[journey_summary['journey_role'] == 'Intent']) > 0 else None
        
            st.markdown("##### 🛒 Intent Layer")
        
            if int_data is not None and int_data['num_creatives'] > 0:
                int_creatives = creative_metrics[creative_metrics['journey_role'] == 'Intent']
        
                # top metric
                st.metric("Creatives", f"{int(int_data['num_creatives'])}")
        
                # KPI metrics
                if 'add_to_cart_rate' in int_creatives.columns:
                    avg_atc = int_creatives['add_to_cart_rate'].mean()
                    if avg_atc > 0:
                        st.metric("ATC Rate", f"{avg_atc:.3%}")
        
                if 'view_content_rate' in int_creatives.columns:
                    avg_vc = int_creatives['view_content_rate'].mean()
                    if avg_vc > 0:
                        st.metric("View Content", f"{avg_vc:.3%}")
        
                if 'page_view_rate' in int_creatives.columns:
                    avg_pv = int_creatives['page_view_rate'].mean()
                    if avg_pv > 0:
                        st.metric("Page View", f"{avg_pv:.3%}")
        
                # bottom metric
                st.metric("Spend", f"${int_data['spend']:,.0f}")
            else:
                st.info("No intent creatives")
        
        with col3:
            conv_data = journey_summary[journey_summary['journey_role'] == 'Conversion'].iloc[0] if len(journey_summary[journey_summary['journey_role'] == 'Conversion']) > 0 else None
            st.markdown("##### 💰 Conversion Layer")
            if conv_data is not None and conv_data['num_creatives'] > 0:
                conv_creatives = creative_metrics[creative_metrics['journey_role'] == 'Conversion']
                if 'CVR' in conv_creatives.columns:
                    avg_cvr = conv_creatives['CVR'].mean()
                    st.metric("Avg CVR", f"{avg_cvr:.3%}")
                if 'CPA' in conv_creatives.columns:
                    avg_cpa = conv_creatives['CPA'].mean()
                    st.metric("Avg CPA", f"${avg_cpa:.2f}")
                if 'ROAS' in conv_creatives.columns:
                    avg_roas = conv_creatives['ROAS'].mean()
                    st.metric("Avg ROAS", f"{avg_roas:.2f}x")
                st.metric("Creatives", f"{int(conv_data['num_creatives'])}")
                st.metric("Spend", f"${conv_data['spend']:,.0f}")
            else:
                st.info("No conversion creatives")
        
        st.markdown("---")
        st.subheader("💵 Spend Distribution by Journey Role")
        
        journey_spend = journey_summary[journey_summary['spend'] > 0].copy()
        if len(journey_spend) > 0:
            colors = {'Engagement': '#4CAF50', 'Intent': '#FF9800', 'Conversion': '#2196F3'}
            fig_spend = px.pie(
                journey_spend,
                values='spend',
                names='journey_role',
                title="How is your spend split across Engagement / Intent / Conversion?",
                color='journey_role',
                color_discrete_map=colors,
                hole=0.4
            )
            fig_spend.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_spend, use_container_width=True)
        else:
            st.info("No spend data available for journey role breakdown.")

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
                fig.update_coloraxes(colorbar_tickformat=".2%")
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

        st.info("💡 Scoring is journey-aware: **Engagement** creatives are scored primarily on CTR/CPC, **Intent** creatives on micro-conversion rates, and **Conversion** creatives on CVR/CPA.")
        
        journey_role_filter = st.selectbox(
            "Filter by Journey Role",
            options=["All", "Engagement", "Intent", "Conversion"],
            index=0,
            help="Filter creatives by their funnel position"
        )
        
        if journey_role_filter != "All":
            leaderboard = leaderboard[leaderboard['journey_role'] == journey_role_filter]

        st.subheader(f"Top Performing Creatives ({len(leaderboard)} total)")

        display_cols = ['creative_name', 'journey_role', 'platform', 'campaign_name']

        if 'topic' in leaderboard.columns:
            display_cols.append('topic')

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

        display_df['CTR'] = display_df['CTR'] * 100
        if 'CVR' in display_df.columns:
            display_df['CVR'] = display_df['CVR'] * 100
        if 'purchase_rate' in display_df.columns:
            display_df['purchase_rate'] = display_df['purchase_rate'] * 100
        if 'add_to_cart_rate' in display_df.columns:
            display_df['add_to_cart_rate'] = display_df['add_to_cart_rate'] * 100
        if 'view_content_rate' in display_df.columns:
            display_df['view_content_rate'] = display_df['view_content_rate'] * 100
        if 'page_view_rate' in display_df.columns:
            display_df['page_view_rate'] = display_df['page_view_rate'] * 100

        column_config = {
            'journey_role': st.column_config.TextColumn('Journey Role'),
            'CTR': st.column_config.NumberColumn('CTR', format="%.3f %%"),
            'CPC': st.column_config.NumberColumn('CPC', format="$ %.2f"),
            'CPM': st.column_config.NumberColumn('CPM', format="$ %.2f"),
            'score': st.column_config.NumberColumn('Score', format="%.3f"),
            'impressions': st.column_config.NumberColumn('Impressions', format="%,d"),
            'clicks': st.column_config.NumberColumn('Clicks', format="%,d"),
            'spend': st.column_config.NumberColumn('Spend', format="$ %,.2f"),
        }

        if 'CVR' in display_df.columns:
            column_config['CVR'] = st.column_config.NumberColumn('CVR', format="%.3f %%")
        if 'CPA' in display_df.columns:
            column_config['CPA'] = st.column_config.NumberColumn('CPA', format="$ %.2f")
        if 'ROAS' in display_df.columns:
            column_config['ROAS'] = st.column_config.NumberColumn('ROAS', format="%.2f x")

        if 'conversions' in display_df.columns:
            column_config['conversions'] = st.column_config.NumberColumn('Conversions', format="%,d")

        if 'purchases' in display_df.columns:
            column_config['purchases'] = st.column_config.NumberColumn('Purchases', format="%,d")
        if 'purchase_rate' in display_df.columns:
            column_config['purchase_rate'] = st.column_config.NumberColumn('Purchase Rate', format="%.3f %%")
        if 'cost_per_purchase' in display_df.columns:
            column_config['cost_per_purchase'] = st.column_config.NumberColumn('Cost/Purchase', format="$ %.2f")

        if 'add_to_carts' in display_df.columns:
            column_config['add_to_carts'] = st.column_config.NumberColumn('Add to Carts', format="%,d")
        if 'add_to_cart_rate' in display_df.columns:
            column_config['add_to_cart_rate'] = st.column_config.NumberColumn('Add to Cart Rate', format="%.3f %%")
        if 'cost_per_add_to_cart' in display_df.columns:
            column_config['cost_per_add_to_cart'] = st.column_config.NumberColumn('Cost/Add to Cart', format="$ %.2f")

        if 'view_content' in display_df.columns:
            column_config['view_content'] = st.column_config.NumberColumn('View Content', format="%,d")
        if 'view_content_rate' in display_df.columns:
            column_config['view_content_rate'] = st.column_config.NumberColumn('View Content Rate', format="%.3f %%")
        if 'cost_per_view_content' in display_df.columns:
            column_config['cost_per_view_content'] = st.column_config.NumberColumn('Cost/View Content', format="$ %.2f")

        if 'page_views' in display_df.columns:
            column_config['page_views'] = st.column_config.NumberColumn('Page Views', format="%,d")
        if 'page_view_rate' in display_df.columns:
            column_config['page_view_rate'] = st.column_config.NumberColumn('Page View Rate', format="%.3f %%")
        if 'cost_per_page_view' in display_df.columns:
            column_config['cost_per_page_view'] = st.column_config.NumberColumn('Cost/Page View', format="$ %.2f")

        # --- NEW: pick a creative to sync with Detail tab ---
        if 'selected_creative' not in st.session_state:
            st.session_state['selected_creative'] = leaderboard.iloc[0]['creative_name']

        selected_from_leaderboard = st.selectbox(
            "Select a creative to analyze in the Detail & Fatigue tab",
            options=leaderboard['creative_name'].tolist(),
            index=0 if st.session_state['selected_creative'] not in leaderboard['creative_name'].tolist()
                   else leaderboard['creative_name'].tolist().index(st.session_state['selected_creative']),
            key="leaderboard_creative_select"
        )
        st.session_state['selected_creative'] = selected_from_leaderboard

        st.dataframe(display_df, use_container_width=True, height=400, column_config=column_config)

        csv_leaderboard = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Leaderboard CSV",
            data=csv_leaderboard,
            file_name="creative_leaderboard.csv",
            mime="text/csv"
        )

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

        # default to last selected creative if available
        default_index = 0
        if 'selected_creative' in st.session_state and st.session_state['selected_creative'] in creative_list:
            default_index = creative_list.index(st.session_state['selected_creative'])

        selected_creative = st.selectbox(
            "Select Creative to Analyze",
            options=creative_list,
            index=default_index,
            key="selected_creative"  # share key with session_state
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

        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            journey_role = creative_summary.get('journey_role', 'Engagement')
            role_emoji = {'Engagement': '📢', 'Intent': '🛒', 'Conversion': '💰'}.get(journey_role, '📊')
            st.metric("Journey Role", f"{role_emoji} {journey_role}")
        
        if 'objective' in creative_summary:
            with col2:
                st.metric("Objective", creative_summary['objective'])
        if 'objective_type' in creative_summary:
            with col3:
                st.metric("Objective Type", creative_summary['objective_type'])
        
        if journey_role == "Engagement":
            st.info("💡 **Engagement Creative**: This creative excels at driving clicks and attention. Evaluate it primarily on CTR/CPC performance. Don't over-judge it on final CPA - its role is to generate interest.")
        elif journey_role == "Intent":
            st.info("💡 **Intent Creative**: This creative drives mid-funnel actions like add-to-carts and page views. Focus on micro-conversion rates. It's building purchase intent, not necessarily closing sales.")
        elif journey_role == "Conversion":
            st.info("💡 **Conversion Creative**: This creative is a closer - it drives final purchases. Evaluate it on CVR, CPA, and ROAS. Strong conversion performance justifies the spend.")

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

        secondary_kpi = st.selectbox(
            "Optional secondary KPI (overlay)",
            options=["None"] + fatigue_kpi_options,
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

                # primary KPI
                fig.add_trace(go.Scatter(
                    x=creative_data['date'],
                    y=creative_data[fatigue_kpi],
                    mode='lines+markers',
                    name=f'Actual {fatigue_kpi}',
                    line=dict(width=2),
                    marker=dict(size=6),
                    yaxis="y1"
                ))

                # trend line for primary
                fig.add_trace(go.Scatter(
                    x=creative_data['date'].values[valid_indices],
                    y=trend_line,
                    mode='lines',
                    name='Trend Line',
                    line=dict(width=2, dash='dash'),
                    yaxis="y1"
                ))

                # optional secondary KPI
                if secondary_kpi != "None":
                    fig.add_trace(go.Scatter(
                        x=creative_data['date'],
                        y=creative_data[secondary_kpi],
                        mode='lines+markers',
                        name=f'{secondary_kpi}',
                        line=dict(width=2, dash='dot'),
                        marker=dict(size=5),
                        yaxis="y2"
                    ))

                    fig.update_layout(
                        yaxis=dict(title=fatigue_kpi),
                        yaxis2=dict(
                            title=secondary_kpi,
                            overlaying='y',
                            side='right'
                        )
                    )
                else:
                    fig.update_layout(
                        yaxis=dict(title=fatigue_kpi)
                    )

                fig.update_layout(
                    title=f"{fatigue_kpi} Over Time for {selected_creative}",
                    xaxis_title="Date",
                    hovermode='x unified'
                )

                if fatigue_kpi in RATE_METRICS:
                    fig.update_yaxes(tickformat=".2%")

                if secondary_kpi in RATE_METRICS and secondary_kpi != "None":
                    # Explicitly update yaxis2 layout
                    fig.update_layout(
                        yaxis2=dict(fig.layout.yaxis2, tickformat=".2%")
                    )

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
        st.header("🏷️ Topic Insights")

        creative_metrics = compute_aggregated_creative_metrics(filtered_df)

        if 'topic' not in creative_metrics.columns or creative_metrics['topic'].isna().all():
            st.warning("⚠️ No topic data available. Add a 'topic' column to your CSV to enable topic-based analysis.")
            st.info("💡 **Tip:** Topics help you group creatives by theme or content type (e.g., 'Product Demo', 'UGC Content', 'Brand Messaging').")
            return

        st.info("💡 Analyze creative performance by topic to identify which content themes drive the best results.")

        st.markdown("---")
        st.subheader("CTR vs CPC Performance by Topic")

        plot_data = creative_metrics[creative_metrics['topic'].notna()].copy()

        if len(plot_data) == 0:
            st.warning("No data available with topics after filtering.")
        else:
            fig = px.scatter(
                plot_data,
                x='CPC',
                y='CTR',
                size='spend',
                color='topic',
                hover_data=['creative_name', 'platform', 'impressions', 'clicks'],
                title="Creative Performance: CTR vs CPC by Topic (bubble size = spend)",
                labels={'CPC': 'Cost Per Click ($)', 'CTR': 'Click-Through Rate', 'topic': 'Topic'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(hovermode='closest', height=500)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("📊 Spend by Topic & Journey Role")
        st.caption("See which topics are skewed toward top-of-funnel (Engagement), mid-funnel (Intent), or bottom-funnel (Conversion) creatives.")
        
        topic_journey_data = creative_metrics[creative_metrics['topic'].notna()].copy()
        
        if len(topic_journey_data) > 0:
            topic_journey_spend = topic_journey_data.groupby(['topic', 'journey_role']).agg({
                'spend': 'sum',
                'creative_name': 'count'
            }).reset_index()
            topic_journey_spend.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
            
            colors = {'Engagement': '#4CAF50', 'Intent': '#FF9800', 'Conversion': '#2196F3'}
            fig_stacked = px.bar(
                topic_journey_spend,
                x='topic',
                y='spend',
                color='journey_role',
                title="Spend Distribution by Topic and Journey Role",
                labels={'topic': 'Topic', 'spend': 'Spend ($)', 'journey_role': 'Journey Role'},
                color_discrete_map=colors,
                barmode='stack'
            )
            fig_stacked.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_stacked, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Topic Performance by Layer")
            
            layer_tabs = st.tabs(["📢 Engagement", "🛒 Intent", "💰 Conversion"])
            
            with layer_tabs[0]:
                eng_topics = topic_journey_data[topic_journey_data['journey_role'] == 'Engagement']
                if len(eng_topics) > 0:
                    eng_by_topic = eng_topics.groupby('topic').agg({
                        'spend': 'sum',
                        'impressions': 'sum',
                        'clicks': 'sum',
                        'CTR': 'mean',
                        'CPC': 'mean',
                        'creative_name': 'count'
                    }).reset_index()
                    eng_by_topic.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
                    eng_by_topic = eng_by_topic.sort_values('CTR', ascending=False)
                    eng_by_topic['CTR'] = eng_by_topic['CTR'] * 100
                    st.dataframe(eng_by_topic, use_container_width=True, column_config={
                        'CTR': st.column_config.NumberColumn('Avg CTR', format="%.3f %%"),
                        'CPC': st.column_config.NumberColumn('Avg CPC', format="$ %.2f"),
                        'spend': st.column_config.NumberColumn('Spend', format="$ %,.0f"),
                    })
                else:
                    st.info("No engagement creatives with topics.")
            
            with layer_tabs[1]:
                int_topics = topic_journey_data[topic_journey_data['journey_role'] == 'Intent']
                if len(int_topics) > 0:
                    int_agg = {'spend': 'sum', 'impressions': 'sum', 'clicks': 'sum', 'creative_name': 'count'}
                    for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']:
                        if col in int_topics.columns:
                            int_agg[col] = 'mean'
                    int_by_topic = int_topics.groupby('topic').agg(int_agg).reset_index()
                    int_by_topic.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
                    col_config = {
                        'spend': st.column_config.NumberColumn('Spend', format="$ %,.0f"),
                    }
                    for col in ['add_to_cart_rate', 'view_content_rate', 'page_view_rate']:
                        if col in int_by_topic.columns:
                            int_by_topic[col] = int_by_topic[col] * 100
                            col_config[col] = st.column_config.NumberColumn(col.replace('_', ' ').title(), format="%.3f %%")
                    st.dataframe(int_by_topic, use_container_width=True, column_config=col_config)
                else:
                    st.info("No intent creatives with topics.")
            
            with layer_tabs[2]:
                conv_topics = topic_journey_data[topic_journey_data['journey_role'] == 'Conversion']
                if len(conv_topics) > 0:
                    conv_agg = {'spend': 'sum', 'impressions': 'sum', 'clicks': 'sum', 'creative_name': 'count'}
                    for col in ['CVR', 'CPA', 'ROAS']:
                        if col in conv_topics.columns:
                            conv_agg[col] = 'mean'
                    if 'conversions' in conv_topics.columns:
                        conv_agg['conversions'] = 'sum'
                    conv_by_topic = conv_topics.groupby('topic').agg(conv_agg).reset_index()
                    conv_by_topic.rename(columns={'creative_name': 'num_creatives'}, inplace=True)
                    conv_by_topic = conv_by_topic.sort_values('CVR' if 'CVR' in conv_by_topic.columns else 'spend', ascending=False)
                    col_config = {
                        'spend': st.column_config.NumberColumn('Spend', format="$ %,.0f"),
                    }
                    if 'CVR' in conv_by_topic.columns:
                        conv_by_topic['CVR'] = conv_by_topic['CVR'] * 100
                        col_config['CVR'] = st.column_config.NumberColumn('Avg CVR', format="%.3f %%")
                    if 'CPA' in conv_by_topic.columns:
                        col_config['CPA'] = st.column_config.NumberColumn('Avg CPA', format="$ %.2f")
                    if 'ROAS' in conv_by_topic.columns:
                        col_config['ROAS'] = st.column_config.NumberColumn('Avg ROAS', format="%.2f x")
                    if 'conversions' in conv_by_topic.columns:
                        col_config['conversions'] = st.column_config.NumberColumn('Conversions', format="%,d")
                    st.dataframe(conv_by_topic, use_container_width=True, column_config=col_config)
                else:
                    st.info("No conversion creatives with topics.")
        else:
            st.info("No topic data available for journey role analysis.")

        st.markdown("---")
        st.subheader("Topic Performance Summary")

        topic_agg_dict = {
            'impressions': 'sum',
            'clicks': 'sum',
            'spend': 'sum',
            'creative_name': 'nunique'
        }

        if has_conversions:
            topic_agg_dict['conversions'] = 'sum'

        topic_metrics = filtered_df[filtered_df['topic'].notna()].groupby('topic').agg(topic_agg_dict).reset_index()
        topic_metrics.rename(columns={'creative_name': 'num_creatives'}, inplace=True)

        topic_metrics['CTR'] = np.where(
            topic_metrics['impressions'] > 0,
            topic_metrics['clicks'] / topic_metrics['impressions'],
            0
        )
        topic_metrics['CPC'] = np.where(
            topic_metrics['clicks'] > 0,
            topic_metrics['spend'] / topic_metrics['clicks'],
            0
        )

        if has_conversions:
            topic_metrics['CVR'] = np.where(
                topic_metrics['clicks'] > 0,
                topic_metrics['conversions'] / topic_metrics['clicks'],
                0
            )
            topic_metrics['CPA'] = np.where(
                topic_metrics['conversions'] > 0,
                topic_metrics['spend'] / topic_metrics['conversions'],
                0
            )

        topic_metrics = topic_metrics.sort_values('CTR', ascending=False)

        # --- NEW: Quadrant chart: CTR vs Spend share by topic ---
        st.markdown("---")
        st.subheader("Spend vs CTR by Topic (Quadrant View)")

        # Work with raw CTR (0-1) for chart; keep a copy to avoid fighting with % scaling
        topic_quadrant = topic_metrics.copy()

        total_spend_topics = topic_quadrant['spend'].sum()
        topic_quadrant['spend_share'] = np.where(
            total_spend_topics > 0,
            topic_quadrant['spend'] / total_spend_topics,
            0
        )

        avg_ctr_raw = topic_quadrant['CTR'].mean()
        avg_spend_share = topic_quadrant['spend_share'].mean()

        fig_q = px.scatter(
            topic_quadrant,
            x='spend_share',
            y='CTR',
            size='spend',
            text='topic',
            labels={
                'spend_share': 'Share of Spend',
                'CTR': 'CTR'
            },
            title="Topic Efficiency: CTR vs Share of Spend (size = spend)"
        )
        fig_q.update_traces(textposition="top center")

        # Add quadrant lines
        fig_q.add_vline(x=avg_spend_share, line_dash="dash", line_color="grey")
        fig_q.add_hline(y=avg_ctr_raw, line_dash="dash", line_color="grey")

        fig_q.update_yaxes(tickformat=".2%")
        fig_q.update_xaxes(tickformat=".1%")
        st.plotly_chart(fig_q, use_container_width=True)


        topic_metrics['CTR'] = topic_metrics['CTR'] * 100
        if 'CVR' in topic_metrics.columns:
            topic_metrics['CVR'] = topic_metrics['CVR'] * 100

        topic_column_config = {
            'topic': st.column_config.TextColumn('Topic'),
            'num_creatives': st.column_config.NumberColumn('# Creatives', format="%d"),
            'impressions': st.column_config.NumberColumn('Impressions', format="%,d"),
            'clicks': st.column_config.NumberColumn('Clicks', format="%,d"),
            'spend': st.column_config.NumberColumn('Spend', format="$ %,.2f"),
            'CTR': st.column_config.NumberColumn('CTR', format="%.3f %%"),
            'CPC': st.column_config.NumberColumn('CPC', format="$ %.2f"),
        }

        if 'CVR' in topic_metrics.columns:
            topic_column_config['CVR'] = st.column_config.NumberColumn('CVR', format="%.3f %%")
            topic_column_config['conversions'] = st.column_config.NumberColumn('Conversions', format="%,d")
            topic_column_config['CPA'] = st.column_config.NumberColumn('CPA', format="$ %.2f")

        st.dataframe(topic_metrics, use_container_width=True, height=400, column_config=topic_column_config)

        csv_topics = topic_metrics.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Topic Performance CSV",
            data=csv_topics,
            file_name="topic_performance.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("📊 Key Topic Insights")

        if len(topic_metrics) >= 1:
            top_topic = topic_metrics.iloc[0]
            bottom_topic = topic_metrics.iloc[-1]

            col1, col2 = st.columns(2)

            with col1:
                st.success(f"**🏆 Best Performing Topic**")
                st.write(f"**{top_topic['topic']}**")
                st.write(f"- CTR: **{top_topic['CTR']:.3f}%**")
                st.write(f"- CPC: **${top_topic['CPC']:.2f}**")
                st.write(f"- {int(top_topic['num_creatives'])} creatives")
                st.write(f"- ${top_topic['spend']:,.0f} total spend")

            with col2:
                st.error(f"**⚠️ Lowest Performing Topic**")
                st.write(f"**{bottom_topic['topic']}**")
                st.write(f"- CTR: **{bottom_topic['CTR']:.3f}%**")
                st.write(f"- CPC: **${bottom_topic['CPC']:.2f}**")
                st.write(f"- {int(bottom_topic['num_creatives'])} creatives")
                st.write(f"- ${bottom_topic['spend']:,.0f} total spend")

            st.markdown("---")

            insights = []

            ctr_range = topic_metrics['CTR'].max() - topic_metrics['CTR'].min()
            if ctr_range > 2.0:
                insights.append(f"• **High CTR variance** across topics ({ctr_range:.2f}% spread) - some topics significantly outperform others")

            high_spend_topics = topic_metrics.nlargest(3, 'spend')
            high_ctr_topics = topic_metrics.nlargest(3, 'CTR')

            overlap = set(high_spend_topics['topic']) & set(high_ctr_topics['topic'])
            if len(overlap) > 0:
                insights.append(f"• **Efficient spend allocation** - High-spend topics ({', '.join(overlap)}) also have high CTR")
            else:
                insights.append(f"• **Opportunity for reallocation** - Your highest-spend topics aren't your best performers")

            avg_ctr = topic_metrics['CTR'].mean()
            above_avg_count = len(topic_metrics[topic_metrics['CTR'] > avg_ctr])
            insights.append(f"• {above_avg_count}/{len(topic_metrics)} topics perform above average CTR ({avg_ctr:.2f}%)")

            if len(insights) > 0:
                st.markdown("**Summary:**")
                for insight in insights:
                    st.markdown(insight)


if __name__ == "__main__":
    main()
