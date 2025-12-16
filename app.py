import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

RATE_METRICS = [
    "CTR",
    "CVR",
    "purchase_rate",
    "add_to_cart_rate",
    "view_content_rate",
    "page_view_rate",
]

RATE_METRICS_SET = set(RATE_METRICS)

CURRENCY_METRICS_SET = {
    "CPC",
    "CPA",
    "CPM",
    "spend",
    "cost_per_purchase",
    "cost_per_add_to_cart",
}

def load_google_sheet_to_df():
    try:
        from google.oauth2.service_account import Credentials
        import gspread
        
        sa = st.secrets["gcp_service_account"]
        sheet_url = st.secrets["google"]["sheet_url"]
        
        creds = Credentials.from_service_account_info(sa).with_scopes(SCOPES)
        gc = gspread.authorize(creds)
        
        sh = gc.open_by_url(sheet_url)
        ws = sh.sheet1
        data = ws.get_all_records()
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Error loading Google Sheet: {e}")
        return None

def classify_objective(objective: str) -> str:
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

def classify_campaign_type(creative_name: str) -> str:
    if not isinstance(creative_name, str):
        return "Other"
    name = creative_name.lower()
    
    awareness_keywords = ["awareness", "reach", "video view", "brand", "impression", "awr"]
    traffic_keywords = ["traffic", "engagement", "clicks"]
    conversion_keywords = ["conversion", "purchase", "sale", "lead", "catalog", "app install"]
    
    if any(k in name for k in conversion_keywords):
        return "Conversion"
    if any(k in name for k in traffic_keywords):
        return "Traffic"
    if any(k in name for k in awareness_keywords):
        return "Awareness"
    return "Other"

def classify_business_objective(campaign_name: str) -> str:
    if not isinstance(campaign_name, str):
        return "Other"
    name = campaign_name.lower()
    
    if "purchase" in name or "sale" in name or "shop" in name:
        return "Direct Purchase"
    if "lead" in name or "signup" in name:
        return "Lead Gen"
    if "catalog" in name or "dpa" in name:
        return "Catalog Sales"
    if "traffic" in name or "click" in name:
        return "Traffic"
    if "awareness" in name or "reach" in name or "brand" in name:
        return "Awareness"
    if "retarget" in name or "remarket" in name or "retention" in name:
        return "Retargeting"
    return "Other"

def classify_campaign_format(campaign_name: str) -> str:
    if not isinstance(campaign_name, str):
        return "Other"
    name = campaign_name.upper()
    
    if "PMAX" in name:
        return "Performance Max"
    if "NB" in name:
        return "Search"
    if "YT" in name or "YOUTUBE" in name:
        return "YouTube"
    if "DISPLAY" in name:
        return "Display"
    if "META" in name or " TT" in name or name.startswith("TT") or "TIKTOK" in name:
        return "Social"
    if "SHOPPING" in name:
        return "Shopping"
    return "Other"

def classify_journey_role(
    row,
    ctr_median=None,
    cpc_median=None,
    cvr_median=None,
    intent_median=None,
    purchase_rate_median=None,
    cpa_median=None,
    cvr_boost=1.2,
    intent_boost=1.1,
    ctr_boost=1.1,
    cpc_discount=0.9,
    cpa_discount=0.8,
    purchase_boost=1.2,
):
    ctr = row.get("CTR", 0.0)
    cvr = row.get("CVR", 0.0)
    purchase_rate = row.get("purchase_rate", 0.0)
    cpa = row.get("cost_per_purchase", None) or row.get("CPA", None)
    
    intent_metrics = []
    for metric in ["add_to_cart_rate", "view_content_rate", "page_view_rate"]:
        val = row.get(metric, 0.0)
        if val and val > 0:
            intent_metrics.append(val)
    avg_intent = float(np.mean(intent_metrics)) if intent_metrics else 0.0
    has_intent_metrics = avg_intent > 0
    
    strong_conversion = False
    if cvr_median and cvr_median > 0:
        if cvr >= cvr_boost * cvr_median:
            strong_conversion = True
    if purchase_rate_median and purchase_rate_median > 0:
        if purchase_rate >= purchase_boost * purchase_rate_median:
            strong_conversion = True
    elif purchase_rate > 0:
        strong_conversion = True
    if cpa is not None and cpa_median and cpa_median > 0:
        if cpa <= cpa_discount * cpa_median:
            strong_conversion = True
    if strong_conversion:
        return "Conversion"
    
    strong_intent = False
    if has_intent_metrics and intent_median and intent_median > 0:
        if avg_intent >= intent_boost * intent_median:
            strong_intent = True
    if strong_intent:
        return "Intent"
    
    strong_engagement = False
    if cpc_median and cpc_median > 0 and "CPC" in row:
        if row["CPC"] <= cpc_discount * cpc_median:
            strong_engagement = True
    if (not strong_engagement) and ctr_median and ctr_median > 0:
        if ctr >= ctr_boost * ctr_median:
            strong_engagement = True
    if strong_engagement:
        return "Engagement"
    
    if purchase_rate > 0:
        return "Conversion"
    if has_intent_metrics:
        return "Intent"
    if ctr > 0:
        return "Engagement"
    return "Engagement"

def load_and_prepare_data(df_raw: pd.DataFrame):
    try:
        df = df_raw.copy()
        
        rename_map = {
            "Date": "date",
            "Week Start": "week_start",
            "Period": "period",
            "Fiscal Year": "fiscal_year",
            "Channel": "platform",
            "Platform": "platform",
            "Campaign": "campaign_name",
            "Campaign Name": "campaign_name",
            "Content Topic": "topic",
            "Topic": "topic",
            "Ad name": "creative_name",
            "Ad Name": "creative_name",
            "Creative Name": "creative_name",
            "Creative Size": "format",
            "Format": "format",
            "Impressions": "impressions",
            "Clicks": "clicks",
            "Spend": "spend",
            "Purchases": "purchases",
            "Revenue": "revenue",
            "Conversions": "conversions",
            "Add To Carts": "add_to_carts",
            "Content Views": "view_content",
            "Page Views": "page_views",
        }
        df = df.rename(columns=rename_map)
        
        required_cols = ["date", "platform", "campaign_name", "creative_name", "impressions", "clicks", "spend"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if df["date"].isna().all():
            st.error("Could not parse any dates in the 'date' column")
            return None
        
        if "week_start" in df.columns:
            df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        
        numeric_cols = ["impressions", "clicks", "spend"]
        optional_numeric = ["conversions", "revenue", "purchases", "add_to_carts", "view_content", "page_views"]
        
        def clean_numeric(series):
            return series.astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
        
        for col in numeric_cols:
            df[col] = clean_numeric(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        for col in optional_numeric:
            if col in df.columns:
                df[col] = clean_numeric(df[col])
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        
        df["CTR"] = np.where(df["impressions"] > 0, df["clicks"] / df["impressions"], 0)
        df["CPM"] = np.where(df["impressions"] > 0, df["spend"] / df["impressions"] * 1000, 0)
        df["CPC"] = np.where(df["clicks"] > 0, df["spend"] / df["clicks"], 0)
        
        if "conversions" in df.columns:
            df["CVR"] = np.where(df["clicks"] > 0, df["conversions"] / df["clicks"], 0)
            df["CPA"] = np.where(df["conversions"] > 0, df["spend"] / df["conversions"], 0)
        
        if "purchases" in df.columns:
            df["purchase_rate"] = np.where(df["clicks"] > 0, df["purchases"] / df["clicks"], 0)
            df["cost_per_purchase"] = np.where(df["purchases"] > 0, df["spend"] / df["purchases"], 0)
        
        if "add_to_carts" in df.columns:
            df["add_to_cart_rate"] = np.where(df["clicks"] > 0, df["add_to_carts"] / df["clicks"], 0)
            df["cost_per_add_to_cart"] = np.where(df["add_to_carts"] > 0, df["spend"] / df["add_to_carts"], 0)
        
        if "view_content" in df.columns:
            df["view_content_rate"] = np.where(df["clicks"] > 0, df["view_content"] / df["clicks"], 0)
            df["cost_per_view_content"] = np.where(df["view_content"] > 0, df["spend"] / df["view_content"], 0)
        
        if "page_views" in df.columns:
            df["page_view_rate"] = np.where(df["clicks"] > 0, df["page_views"] / df["clicks"], 0)
            df["cost_per_page_view"] = np.where(df["page_views"] > 0, df["spend"] / df["page_views"], 0)
        
        if "revenue" in df.columns:
            df["ROAS"] = np.where(df["spend"] > 0, df["revenue"] / df["spend"], 0)
        
        df = df.sort_values(["creative_name", "date"])
        
        df["campaign_format"] = df["campaign_name"].apply(classify_campaign_format)
        df["campaign_type"] = df["creative_name"].apply(classify_campaign_type)
        df["business_objective"] = df["campaign_name"].apply(classify_business_objective)
        
        creative_first_dates = df.groupby("creative_name")["date"].transform("min")
        df["age_in_days"] = (df["date"] - creative_first_dates).dt.days
        df["cumulative_impressions"] = df.groupby("creative_name")["impressions"].cumsum()
        
        if "objective" in df.columns:
            df["objective_type"] = df["objective"].apply(classify_objective)
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def get_date_ranges(choice: str, df, min_date, max_date):
    today = datetime.now().date()
    
    week_starts = None
    if "week_start" in df.columns:
        week_starts = sorted(df["week_start"].dropna().unique())
    
    current_fy = None
    df_fy = None
    period_order = []
    current_period = None
    
    if "fiscal_year" in df.columns:
        df_sorted = df.sort_values("date")
        latest_row = df_sorted.iloc[-1]
        current_fy = latest_row["fiscal_year"]
        df_fy = df_sorted[df_sorted["fiscal_year"] == current_fy]
        
        if "period" in df_fy.columns and not df_fy.empty:
            period_order = (
                df_fy.groupby("period")["date"]
                .min()
                .sort_values()
                .index
                .tolist()
            )
            current_period = df_fy.iloc[-1]["period"]
    
    if choice == "Custom Range":
        return None, None, None, None
    elif choice == "This Week":
        if week_starts is not None and len(week_starts) >= 1:
            this_week_start = week_starts[-1]
            start = this_week_start
            end = this_week_start + pd.Timedelta(days=6)
            prior_start = start - pd.Timedelta(days=7)
            prior_end = start - pd.Timedelta(days=1)
            return pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(prior_start), pd.Timestamp(prior_end)
        else:
            start = today - timedelta(days=today.weekday())
            end = today
            prior_start = start - timedelta(days=7)
            prior_end = start - timedelta(days=1)
    elif choice == "Last Week":
        if week_starts is not None and len(week_starts) >= 2:
            last_week_start = week_starts[-2]
            start = last_week_start
            end = last_week_start + pd.Timedelta(days=6)
            prior_start = start - pd.Timedelta(days=7)
            prior_end = start - pd.Timedelta(days=1)
            return pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(prior_start), pd.Timestamp(prior_end)
        else:
            end = today - timedelta(days=today.weekday() + 1)
            start = end - timedelta(days=6)
            prior_end = start - timedelta(days=1)
            prior_start = prior_end - timedelta(days=6)
    elif choice == "This Period" and df_fy is not None and current_period is not None:
        period_mask = (df["fiscal_year"] == current_fy) & (df["period"] == current_period)
        period_dates = df.loc[period_mask, "date"]
        if not period_dates.empty:
            start = period_dates.min()
            end = period_dates.max()
            idx = period_order.index(current_period) if current_period in period_order else 0
            if idx > 0:
                last_period = period_order[idx - 1]
                last_period_mask = (df["fiscal_year"] == current_fy) & (df["period"] == last_period)
                last_period_dates = df.loc[last_period_mask, "date"]
                prior_start = last_period_dates.min() if not last_period_dates.empty else None
                prior_end = last_period_dates.max() if not last_period_dates.empty else None
            else:
                prior_start, prior_end = None, None
            return pd.Timestamp(start), pd.Timestamp(end), prior_start, prior_end
        return min_date, max_date, None, None
    elif choice == "Last Period" and df_fy is not None and current_period is not None and period_order:
        idx = period_order.index(current_period) if current_period in period_order else 0
        if idx > 0:
            last_period = period_order[idx - 1]
            last_period_mask = (df["fiscal_year"] == current_fy) & (df["period"] == last_period)
            last_period_dates = df.loc[last_period_mask, "date"]
            if not last_period_dates.empty:
                start = last_period_dates.min()
                end = last_period_dates.max()
                if idx > 1:
                    prior_period = period_order[idx - 2]
                    prior_mask = (df["fiscal_year"] == current_fy) & (df["period"] == prior_period)
                    prior_dates = df.loc[prior_mask, "date"]
                    prior_start = prior_dates.min() if not prior_dates.empty else None
                    prior_end = prior_dates.max() if not prior_dates.empty else None
                else:
                    prior_start, prior_end = None, None
                return pd.Timestamp(start), pd.Timestamp(end), prior_start, prior_end
        return min_date, max_date, None, None
    elif choice == "This Fiscal Year" and df_fy is not None and not df_fy.empty:
        fy_dates = df_fy["date"]
        start = fy_dates.min()
        end = fy_dates.max()
        prior_start = start.replace(year=start.year - 1) if hasattr(start, 'replace') else None
        prior_end = end.replace(year=end.year - 1) if hasattr(end, 'replace') else None
        return pd.Timestamp(start), pd.Timestamp(end), prior_start, prior_end
    elif choice == "This Month":
        start = today.replace(day=1)
        end = today
        prior_end = start - timedelta(days=1)
        prior_start = prior_end.replace(day=1)
    elif choice == "Last Month":
        first_of_this_month = today.replace(day=1)
        end = first_of_this_month - timedelta(days=1)
        start = end.replace(day=1)
        prior_end = start - timedelta(days=1)
        prior_start = prior_end.replace(day=1)
    elif choice == "This Year":
        start = today.replace(month=1, day=1)
        end = today
        prior_start = start.replace(year=start.year - 1)
        prior_end = end.replace(year=end.year - 1)
    elif choice == "Last Year":
        end = today.replace(month=1, day=1) - timedelta(days=1)
        start = end.replace(month=1, day=1)
        prior_start = start.replace(year=start.year - 1)
        prior_end = end.replace(year=end.year - 1)
    else:
        return min_date, max_date, None, None
    
    return pd.Timestamp(start), pd.Timestamp(end), pd.Timestamp(prior_start) if prior_start else None, pd.Timestamp(prior_end) if prior_end else None

def apply_global_filters(df, filters):
    filtered_df = df.copy()
    
    if filters["date_range"]:
        start_date, end_date = filters["date_range"]
        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.to_datetime(start_date)) &
            (filtered_df["date"] <= pd.to_datetime(end_date))
        ]
    
    if filters.get("platforms"):
        filtered_df = filtered_df[filtered_df["platform"].isin(filters["platforms"])]
    
    if filters.get("campaign_formats") and "campaign_format" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["campaign_format"].isin(filters["campaign_formats"])]
    
    if filters.get("business_objectives") and "business_objective" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["business_objective"].isin(filters["business_objectives"])]
    
    if filters.get("campaign_types") and "campaign_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["campaign_type"].isin(filters["campaign_types"])]
    
    if filters.get("topics") and "topic" in df.columns:
        filtered_df = filtered_df[filtered_df["topic"].isin(filters["topics"])]
    
    if filters.get("formats") and "format" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["format"].isin(filters["formats"])]
    
    agg_dict = {"impressions": "sum"}
    if "conversions" in filtered_df.columns:
        agg_dict["conversions"] = "sum"
    
    creative_totals = filtered_df.groupby("creative_name").agg(agg_dict).reset_index()
    valid_creatives = creative_totals[creative_totals["impressions"] >= filters["min_impressions"]]["creative_name"]
    
    if "conversions" in filtered_df.columns and filters["min_conversions"] > 0:
        valid_creatives_conv = creative_totals[creative_totals["conversions"] >= filters["min_conversions"]]["creative_name"]
        valid_creatives = set(valid_creatives) & set(valid_creatives_conv)
    
    filtered_df = filtered_df[filtered_df["creative_name"].isin(valid_creatives)]
    return filtered_df

@st.cache_data
def compute_aggregated_creative_metrics(df):
    agg_dict = {
        "impressions": "sum",
        "clicks": "sum",
        "spend": "sum",
        "platform": "first",
        "campaign_name": "first",
        "age_in_days": "max",
        "date": "nunique",
    }
    
    for col in ["conversions", "revenue", "purchases", "add_to_carts", "view_content", "page_views"]:
        if col in df.columns:
            agg_dict[col] = "sum"
    
    for col in ["format", "objective", "objective_type", "topic", "campaign_type", "campaign_format", "business_objective"]:
        if col in df.columns:
            agg_dict[col] = "first"
    
    creative_metrics = df.groupby("creative_name").agg(agg_dict).reset_index()
    creative_metrics.rename(columns={"age_in_days": "age_in_days_max", "date": "total_days_active"}, inplace=True)
    
    creative_metrics["CTR"] = np.where(creative_metrics["impressions"] > 0, creative_metrics["clicks"] / creative_metrics["impressions"], 0)
    creative_metrics["CPC"] = np.where(creative_metrics["clicks"] > 0, creative_metrics["spend"] / creative_metrics["clicks"], 0)
    creative_metrics["CPM"] = np.where(creative_metrics["impressions"] > 0, creative_metrics["spend"] / creative_metrics["impressions"] * 1000, 0)
    
    if "conversions" in creative_metrics.columns:
        creative_metrics["CVR"] = np.where(creative_metrics["clicks"] > 0, creative_metrics["conversions"] / creative_metrics["clicks"], 0)
        creative_metrics["CPA"] = np.where(creative_metrics["conversions"] > 0, creative_metrics["spend"] / creative_metrics["conversions"], 0)
    
    if "revenue" in creative_metrics.columns:
        creative_metrics["ROAS"] = np.where(creative_metrics["spend"] > 0, creative_metrics["revenue"] / creative_metrics["spend"], 0)
    
    if "purchases" in creative_metrics.columns:
        creative_metrics["purchase_rate"] = np.where(creative_metrics["clicks"] > 0, creative_metrics["purchases"] / creative_metrics["clicks"], 0)
        creative_metrics["cost_per_purchase"] = np.where(creative_metrics["purchases"] > 0, creative_metrics["spend"] / creative_metrics["purchases"], 0)
    
    if "add_to_carts" in creative_metrics.columns:
        creative_metrics["add_to_cart_rate"] = np.where(creative_metrics["clicks"] > 0, creative_metrics["add_to_carts"] / creative_metrics["clicks"], 0)
        creative_metrics["cost_per_add_to_cart"] = np.where(creative_metrics["add_to_carts"] > 0, creative_metrics["spend"] / creative_metrics["add_to_carts"], 0)
    
    if "view_content" in creative_metrics.columns:
        creative_metrics["view_content_rate"] = np.where(creative_metrics["clicks"] > 0, creative_metrics["view_content"] / creative_metrics["clicks"], 0)
    
    if "page_views" in creative_metrics.columns:
        creative_metrics["page_view_rate"] = np.where(creative_metrics["clicks"] > 0, creative_metrics["page_views"] / creative_metrics["clicks"], 0)
    
    ctr_median = creative_metrics.loc[creative_metrics["impressions"] > 0, "CTR"].median() if len(creative_metrics) > 0 else 0
    cvr_median = None
    if "CVR" in creative_metrics.columns:
        cvr_vals = creative_metrics.loc[creative_metrics["CVR"] > 0, "CVR"]
        cvr_median = cvr_vals.median() if len(cvr_vals) > 0 else None
    
    cpc_median = None
    if "CPC" in creative_metrics.columns:
        cpc_vals = creative_metrics.loc[creative_metrics["CPC"] > 0, "CPC"]
        cpc_median = cpc_vals.median() if len(cpc_vals) > 0 else None
    
    intent_scores = []
    intent_cols = [col for col in ["add_to_cart_rate", "view_content_rate", "page_view_rate"] if col in creative_metrics.columns]
    for _, row in creative_metrics.iterrows():
        vals = [row[col] for col in intent_cols if row.get(col, 0) > 0]
        if vals:
            intent_scores.append(np.mean(vals))
    intent_median = np.median(intent_scores) if intent_scores else None
    
    creative_metrics["journey_role"] = creative_metrics.apply(
        lambda row: classify_journey_role(row=row, ctr_median=ctr_median, cvr_median=cvr_median, intent_median=intent_median, cpc_median=cpc_median),
        axis=1
    )
    
    return creative_metrics

def compute_platform_metrics(df: pd.DataFrame):
    agg_dict = {
        "impressions": "sum",
        "clicks": "sum",
        "spend": "sum",
    }
    
    for col in ["purchases", "revenue", "add_to_carts", "conversions"]:
        if col in df.columns:
            agg_dict[col] = "sum"
    
    plat = df.groupby("platform").agg(agg_dict).reset_index()
    
    plat["CTR"] = np.where(plat["impressions"] > 0, plat["clicks"] / plat["impressions"], 0)
    plat["CPC"] = np.where(plat["clicks"] > 0, plat["spend"] / plat["clicks"], 0)
    plat["CPM"] = np.where(plat["impressions"] > 0, plat["spend"] / plat["impressions"] * 1000, 0)
    
    if "purchases" in plat.columns:
        plat["purchase_rate"] = np.where(plat["clicks"] > 0, plat["purchases"] / plat["clicks"], 0)
        plat["cost_per_purchase"] = np.where(plat["purchases"] > 0, plat["spend"] / plat["purchases"], 0)
    
    if "add_to_carts" in plat.columns:
        plat["add_to_cart_rate"] = np.where(plat["clicks"] > 0, plat["add_to_carts"] / plat["clicks"], 0)
        plat["cost_per_add_to_cart"] = np.where(plat["add_to_carts"] > 0, plat["spend"] / plat["add_to_carts"], 0)
    
    if "revenue" in plat.columns:
        plat["ROAS"] = np.where(plat["spend"] > 0, plat["revenue"] / plat["spend"], 0)
    
    if "conversions" in plat.columns:
        plat["CVR"] = np.where(plat["clicks"] > 0, plat["conversions"] / plat["clicks"], 0)
        plat["CPA"] = np.where(plat["conversions"] > 0, plat["spend"] / plat["conversions"], 0)
    
    return plat

def compute_topic_summary(creative_metrics, has_conversions):
    if "topic" not in creative_metrics.columns:
        return None
    
    topic_level = creative_metrics[creative_metrics["topic"].notna()].copy()
    if len(topic_level) == 0:
        return None
    
    agg_dict = {
        "impressions": "sum",
        "clicks": "sum",
        "spend": "sum",
        "creative_name": "nunique",
    }
    
    for col in ["purchases", "revenue", "add_to_carts"]:
        if col in topic_level.columns:
            agg_dict[col] = "sum"
    
    topic_summary = topic_level.groupby("topic").agg(agg_dict).reset_index()
    topic_summary.rename(columns={"creative_name": "num_creatives"}, inplace=True)
    
    topic_summary["CTR"] = np.where(topic_summary["impressions"] > 0, topic_summary["clicks"] / topic_summary["impressions"], 0)
    topic_summary["CPC"] = np.where(topic_summary["clicks"] > 0, topic_summary["spend"] / topic_summary["clicks"], 0)
    
    if "purchases" in topic_summary.columns:
        topic_summary["purchase_rate"] = np.where(topic_summary["clicks"] > 0, topic_summary["purchases"] / topic_summary["clicks"], 0)
    
    if "revenue" in topic_summary.columns:
        topic_summary["ROAS"] = np.where(topic_summary["spend"] > 0, topic_summary["revenue"] / topic_summary["spend"], 0)
    
    return topic_summary

def compute_period_metrics(df):
    total_spend = df["spend"].sum()
    total_impressions = df["impressions"].sum()
    total_clicks = df["clicks"].sum()
    total_purchases = df["purchases"].sum() if "purchases" in df.columns else 0
    total_add_to_carts = df["add_to_carts"].sum() if "add_to_carts" in df.columns else 0
    total_revenue = df["revenue"].sum() if "revenue" in df.columns else 0
    total_conversions = df["conversions"].sum() if "conversions" in df.columns else 0
    
    ctr = total_clicks / total_impressions if total_impressions > 0 else 0
    cpc = total_spend / total_clicks if total_clicks > 0 else 0
    cpm = total_spend / total_impressions * 1000 if total_impressions > 0 else 0
    roas = total_revenue / total_spend if total_spend > 0 else 0
    cvr = total_conversions / total_clicks if total_clicks > 0 else 0
    
    return {
        "spend": total_spend,
        "impressions": total_impressions,
        "clicks": total_clicks,
        "purchases": total_purchases,
        "add_to_carts": total_add_to_carts,
        "revenue": total_revenue,
        "conversions": total_conversions,
        "CTR": ctr,
        "CPC": cpc,
        "CPM": cpm,
        "ROAS": roas,
        "CVR": cvr,
    }

def build_leaderboard(creative_metrics):
    leaderboard = creative_metrics.copy()
    has_conversions = "CVR" in leaderboard.columns
    has_intent_metrics = any(col in leaderboard.columns for col in ["add_to_cart_rate", "view_content_rate", "page_view_rate"])
    
    leaderboard["CTR_percentile"] = leaderboard["CTR"].rank(pct=True)
    leaderboard["CPC_percentile"] = leaderboard["CPC"].rank(pct=True)
    
    if has_conversions:
        leaderboard["CVR_percentile"] = leaderboard["CVR"].rank(pct=True)
    
    if has_intent_metrics:
        intent_cols = [col for col in ["add_to_cart_rate", "view_content_rate", "page_view_rate"] if col in leaderboard.columns]
        leaderboard["intent_avg"] = leaderboard[intent_cols].mean(axis=1)
        leaderboard["intent_percentile"] = leaderboard["intent_avg"].rank(pct=True)
    
    def compute_journey_score(row):
        journey_role = row.get("journey_role", "Engagement")
        ctr_p = row.get("CTR_percentile", 0.5)
        cpc_p = row.get("CPC_percentile", 0.5)
        cvr_p = row.get("CVR_percentile", 0.5) if has_conversions else 0.5
        intent_p = row.get("intent_percentile", 0.5) if has_intent_metrics else 0.5
        
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
    
    leaderboard["score"] = leaderboard.apply(compute_journey_score, axis=1)
    leaderboard = leaderboard.sort_values("score", ascending=False)
    return leaderboard

def show_welcome_screen():
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
    - 📈 Executive Summary with period-over-period comparisons
    - 🟦 Platform comparison with efficiency metrics
    - 📊 Campaign-level performance overview
    - 🎯 Creative fatigue detection by topic
    - 🏷️ Topic insights with layer breakdowns
    
    #### Getting Started
    
    Choose a data source from the sidebar:
    - **CSV Upload**: Upload your own creative performance CSV
    - **Google Sheets**: Connect to a live Google Sheet (requires setup)
    """)
    
    st.markdown("""
    **Required columns:**
    - `date`, `platform`, `campaign_name`, `creative_name`, `impressions`, `clicks`, `spend`
    
    **Optional columns:**
    - `purchases`, `add_to_carts`, `view_content`, `page_views`, `conversions`, `revenue`, `format`, `topic`
    - `week_start`, `period`, `fiscal_year` (for period-based filtering)
    """)

def format_delta(current, prior, is_rate=False, is_currency=False, lower_is_better=False):
    if prior == 0:
        return ""
    delta = ((current - prior) / prior) * 100
    arrow = "↑" if delta > 0 else "↓"
    color = "green" if (delta > 0 and not lower_is_better) or (delta < 0 and lower_is_better) else "red"
    return f":{color}[{arrow} {abs(delta):.1f}%]"

def main():
    st.set_page_config(
        page_title="Creative Performance Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.sidebar.title("📊 Creative Analytics")
    st.sidebar.markdown("---")
    
    data_source = st.sidebar.radio(
        "Data Source",
        options=["CSV Upload", "Google Sheets"],
        index=0,
        help="Choose how to load your data"
    )
    
    df = None
    
    if data_source == "CSV Upload":
        with st.sidebar.expander("📥 Download CSV Template"):
            st.markdown("Download the template to see the expected CSV format.")
            try:
                with open("creative_performance_template.csv", "rb") as f:
                    template_data = f.read()
                st.download_button(
                    label="⬇️ Download Template",
                    data=template_data,
                    file_name="creative_performance_template.csv",
                    mime="text/csv",
                )
            except FileNotFoundError:
                st.info("Template file not found.")
        
        uploaded_file = st.sidebar.file_uploader("Upload Creative Performance CSV", type=["csv"])
        
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            df = load_and_prepare_data(raw_df)
    else:
        st.sidebar.info("📡 Loading from Google Sheets...")
        try:
            raw_df = load_google_sheet_to_df()
            if raw_df is not None:
                df = load_and_prepare_data(raw_df)
                st.sidebar.success(f"✅ Loaded {len(df):,} rows from Google Sheets")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
            st.sidebar.markdown("""
            **Setup Required:**
            Create `.streamlit/secrets.toml` with your Google credentials.
            """)
    
    if df is None:
        show_welcome_screen()
        return
    
    st.sidebar.success(f"✅ Loaded {len(df):,} rows")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📅 Date Filter")
    
    min_date = df["date"].min()
    max_date = df["date"].max()
    
    date_filter_options = ["Custom Range", "This Week", "Last Week", "This Month", "Last Month", "This Year", "Last Year"]
    
    if "fiscal_year" in df.columns and "period" in df.columns:
        date_filter_options = ["Custom Range", "This Week", "Last Week", "This Period", "Last Period", "This Fiscal Year", "This Month", "Last Month"]
    
    quick_choice = st.sidebar.radio(
        "Quick Date Filter",
        options=date_filter_options,
        index=0,
    )
    
    start_date, end_date, prior_start, prior_end = get_date_ranges(quick_choice, df, min_date, max_date)
    
    if quick_choice == "Custom Range":
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        else:
            start_date, end_date = min_date, max_date
        prior_start, prior_end = None, None
    else:
        if start_date and end_date:
            st.sidebar.info(f"📅 {start_date.date()} to {end_date.date()}")
    
    date_range_filter = (start_date, end_date)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Filters")
    
    all_platforms = sorted(df["platform"].unique().tolist())
    selected_platforms = st.sidebar.multiselect("Platform", options=all_platforms, default=all_platforms)
    
    selected_campaign_formats = None
    if "campaign_format" in df.columns:
        all_formats = sorted(df["campaign_format"].dropna().unique().tolist())
        selected_campaign_formats = st.sidebar.multiselect("Campaign Format", options=all_formats, default=all_formats)
    
    selected_business_objectives = None
    if "business_objective" in df.columns:
        all_biz_obj = sorted([o for o in df["business_objective"].dropna().unique().tolist()])
        if all_biz_obj:
            selected_business_objectives = st.sidebar.multiselect(
                "Business Objective",
                options=all_biz_obj,
                default=all_biz_obj,
                help="Derived from campaign names (Direct Purchase, Lead Gen, Catalog Sales, Traffic, Awareness, Retargeting, Other)"
            )
    
    selected_campaign_types = None
    if "campaign_type" in df.columns:
        all_types = sorted(df["campaign_type"].dropna().unique().tolist())
        selected_campaign_types = st.sidebar.multiselect("Campaign Type", options=all_types, default=all_types)
    
    selected_topics = None
    if "topic" in df.columns:
        all_topics = sorted([t for t in df["topic"].dropna().unique().tolist()])
        if all_topics:
            selected_topics = st.sidebar.multiselect("Topic", options=all_topics, default=all_topics)
    
    min_impressions = st.sidebar.slider("Min Impressions per Creative", min_value=0, max_value=50000, value=1000, step=500)
    
    has_conversions = "conversions" in df.columns
    min_conversions = 0
    if has_conversions:
        min_conversions = st.sidebar.slider("Min Conversions per Creative", min_value=0, max_value=100, value=0, step=5)
    
    filters = {
        "date_range": date_range_filter,
        "platforms": selected_platforms,
        "campaign_formats": selected_campaign_formats,
        "business_objectives": selected_business_objectives,
        "campaign_types": selected_campaign_types,
        "topics": selected_topics,
        "formats": None,
        "min_impressions": min_impressions,
        "min_conversions": min_conversions,
    }
    
    filtered_df = apply_global_filters(df, filters)
    
    if len(filtered_df) == 0:
        st.warning("No data matches the current filters. Please adjust your filter settings.")
        return
    
    st.sidebar.info(f"📊 {len(filtered_df):,} rows after filtering")
    
    prior_df = None
    if prior_start and prior_end:
        prior_filters = filters.copy()
        prior_filters["date_range"] = (prior_start, prior_end)
        prior_df = apply_global_filters(df, prior_filters)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 Executive Summary",
        "🟦 Platform Comparison",
        "📊 Campaign Overview",
        "📉 Creative Detail & Fatigue",
        "🏷️ Topic Insights"
    ])
    
    with tab1:
        st.header("📌 Executive Summary")
        st.caption("High-level view: which platforms and creative themes are driving purchases, add-to-carts, and revenue—and where to shift budget next.")
        
        if prior_df is not None and len(prior_df) > 0:
            st.caption(f"Comparing {start_date.date()} - {end_date.date()} vs. {prior_start.date()} - {prior_end.date()}")
        
        current_metrics = compute_period_metrics(filtered_df)
        prior_metrics = compute_period_metrics(prior_df) if prior_df is not None and len(prior_df) > 0 else None
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            delta = format_delta(current_metrics["spend"], prior_metrics["spend"]) if prior_metrics else ""
            st.metric("Total Spend", f"${current_metrics['spend']:,.0f}", delta if delta else None)
        with col2:
            delta = format_delta(current_metrics["impressions"], prior_metrics["impressions"]) if prior_metrics else ""
            st.metric("Impressions", f"{current_metrics['impressions']:,.0f}", delta if delta else None)
        with col3:
            delta = format_delta(current_metrics["clicks"], prior_metrics["clicks"]) if prior_metrics else ""
            st.metric("Clicks", f"{current_metrics['clicks']:,.0f}", delta if delta else None)
        with col4:
            if current_metrics["purchases"] > 0:
                delta = format_delta(current_metrics["purchases"], prior_metrics["purchases"]) if prior_metrics else ""
                st.metric("Purchases", f"{int(current_metrics['purchases']):,}", delta if delta else None)
            elif current_metrics["conversions"] > 0:
                delta = format_delta(current_metrics["conversions"], prior_metrics["conversions"]) if prior_metrics else ""
                st.metric("Conversions", f"{int(current_metrics['conversions']):,}", delta if delta else None)
            else:
                st.metric("Purchases", "N/A")
        with col5:
            if current_metrics["add_to_carts"] > 0:
                delta = format_delta(current_metrics["add_to_carts"], prior_metrics["add_to_carts"]) if prior_metrics else ""
                st.metric("Add to Carts", f"{int(current_metrics['add_to_carts']):,}", delta if delta else None)
            else:
                st.metric("Add to Carts", "N/A")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            delta = format_delta(current_metrics["CTR"], prior_metrics["CTR"]) if prior_metrics else ""
            st.metric("CTR", f"{current_metrics['CTR']:.2%}", delta if delta else None)
        with col2:
            delta = format_delta(current_metrics["CPC"], prior_metrics["CPC"], lower_is_better=True) if prior_metrics else ""
            st.metric("CPC", f"${current_metrics['CPC']:.2f}", delta if delta else None)
        with col3:
            delta = format_delta(current_metrics["CPM"], prior_metrics["CPM"], lower_is_better=True) if prior_metrics else ""
            st.metric("CPM", f"${current_metrics['CPM']:.2f}", delta if delta else None)
        with col4:
            if current_metrics["revenue"] > 0:
                delta = format_delta(current_metrics["revenue"], prior_metrics["revenue"]) if prior_metrics else ""
                st.metric("Revenue", f"${current_metrics['revenue']:,.0f}", delta if delta else None)
            else:
                st.metric("Revenue", "N/A")
        with col5:
            if current_metrics["ROAS"] > 0:
                delta = format_delta(current_metrics["ROAS"], prior_metrics["ROAS"]) if prior_metrics else ""
                st.metric("ROAS", f"{current_metrics['ROAS']:.2f}x", delta if delta else None)
            else:
                st.metric("ROAS", "N/A")
        
        st.markdown("---")
        
        creative_metrics = compute_aggregated_creative_metrics(filtered_df)
        platform_metrics = compute_platform_metrics(filtered_df)
        
        st.subheader("🔍 Platform Snapshot")
        
        if len(platform_metrics) == 0:
            st.info("No platform data available after filters.")
        else:
            total_platform_spend = platform_metrics["spend"].sum()
            cols = st.columns(min(4, len(platform_metrics)))
            
            for idx, row in platform_metrics.iterrows():
                col = cols[idx % len(cols)]
                with col:
                    spend_share = row["spend"] / total_platform_spend if total_platform_spend > 0 else 0
                    st.markdown(f"#### {row['platform']}")
                    st.metric("Share of Spend", f"{spend_share:.1%}")
                    st.metric("CPC", f"${row['CPC']:.2f}")
                    if "purchases" in row and row["purchases"] > 0:
                        st.metric("Purchases (Cost)", f"{int(row['purchases'])} | ${row['cost_per_purchase']:.2f}")
                    if "add_to_carts" in row and row["add_to_carts"] > 0:
                        st.metric("Add to Carts (Cost)", f"{int(row['add_to_carts'])} | ${row['cost_per_add_to_cart']:.2f}")
                    if "ROAS" in row and row["ROAS"] > 0:
                        st.metric("ROAS", f"{row['ROAS']:.2f}x")
        
        st.markdown("---")
        
        st.subheader("🏷️ Top Creative Themes")
        
        topic_summary = compute_topic_summary(creative_metrics, has_conversions)
        if topic_summary is not None and len(topic_summary) > 0:
            topic_summary = topic_summary.sort_values("spend", ascending=False).head(5)
            
            for _, row in topic_summary.iterrows():
                topic_creatives = creative_metrics[creative_metrics["topic"] == row["topic"]]
                journey_dist = topic_creatives.groupby("journey_role")["spend"].sum()
                
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                with col1:
                    st.markdown(f"**{row['topic']}** ({row['num_creatives']} creatives)")
                with col2:
                    st.metric("Spend", f"${row['spend']:,.0f}")
                with col3:
                    st.metric("CTR", f"{row['CTR']:.2%}")
                with col4:
                    if "ROAS" in row and row["ROAS"] > 0:
                        st.metric("ROAS", f"{row['ROAS']:.2f}x")
                    else:
                        st.metric("ROAS", "N/A")
        else:
            st.info("No topic data available. Add a 'topic' column to enable theme analysis.")
        
        st.markdown("---")
        
        st.subheader("🎯 Portfolio View by Journey Layer")
        
        journey_summary = creative_metrics.groupby("journey_role").agg({
            "spend": "sum",
            "impressions": "sum",
            "clicks": "sum",
            "creative_name": "count"
        }).reset_index()
        journey_summary.rename(columns={"creative_name": "num_creatives"}, inplace=True)
        
        col1, col2, col3 = st.columns(3)
        
        for i, role in enumerate(["Engagement", "Intent", "Conversion"]):
            role_data = journey_summary[journey_summary["journey_role"] == role]
            with [col1, col2, col3][i]:
                emoji = {"Engagement": "📢", "Intent": "🛒", "Conversion": "💰"}[role]
                st.markdown(f"### {emoji} {role}")
                if len(role_data) > 0:
                    row = role_data.iloc[0]
                    st.metric("Creatives", f"{int(row['num_creatives']):,}")
                    st.metric("Spend", f"${row['spend']:,.0f}")
                    ctr = row["clicks"] / row["impressions"] if row["impressions"] > 0 else 0
                    st.metric("CTR", f"{ctr:.2%}")
                else:
                    st.info("No creatives in this layer")
    
    with tab2:
        st.header("🟦 Platform Comparison")
        st.caption("Compare platform-level efficiency: CPC, conversions, and ROAS across your media mix.")
        
        platform_metrics = compute_platform_metrics(filtered_df)
        
        if len(platform_metrics) == 0:
            st.warning("No platform data available after filters.")
        else:
            total_spend_pf = platform_metrics["spend"].sum()
            cols = st.columns(min(4, len(platform_metrics)))
            
            for idx, row in platform_metrics.iterrows():
                col = cols[idx % len(cols)]
                with col:
                    spend_share = row["spend"] / total_spend_pf if total_spend_pf > 0 else 0
                    st.markdown(f"### {row['platform']}")
                    st.metric("Share of Spend", f"{spend_share:.1%}")
                    st.metric("Spend", f"${row['spend']:,.0f}")
                    st.metric("CPC", f"${row['CPC']:.2f}")
                    st.metric("CTR", f"{row['CTR']:.2%}")
                    
                    if "purchases" in row and row["purchases"] > 0:
                        st.metric("Purchases", f"{int(row['purchases']):,}")
                        st.metric("Cost per Purchase", f"${row['cost_per_purchase']:.2f}")
                    
                    if "add_to_carts" in row and row["add_to_carts"] > 0:
                        st.metric("Add to Carts", f"{int(row['add_to_carts']):,}")
                        st.metric("Cost per Add to Cart", f"${row['cost_per_add_to_cart']:.2f}")
                    
                    if "ROAS" in row and row["ROAS"] > 0:
                        st.metric("ROAS", f"{row['ROAS']:.2f}x")
            
            st.markdown("---")
            st.subheader("Spend vs CPC (color = ROAS)")
            
            pf_chart = platform_metrics.copy()
            pf_chart["spend_share"] = np.where(total_spend_pf > 0, pf_chart["spend"] / total_spend_pf, 0)
            
            hover_data = ["impressions", "clicks"]
            if "purchases" in pf_chart.columns:
                hover_data.append("purchases")
            if "add_to_carts" in pf_chart.columns:
                hover_data.append("add_to_carts")
            
            color_col = "ROAS" if "ROAS" in pf_chart.columns and pf_chart["ROAS"].sum() > 0 else "CTR"
            
            fig_pf = px.scatter(
                pf_chart,
                x="spend_share",
                y="CPC",
                size="spend",
                color=color_col,
                text="platform",
                hover_data=hover_data,
                labels={
                    "spend_share": "Share of Spend",
                    "CPC": "CPC ($)",
                    color_col: color_col,
                },
                title=f"Platform Efficiency: CPC vs Spend (color = {color_col})",
                color_continuous_scale="RdYlGn",
            )
            fig_pf.update_xaxes(tickformat=".1%")
            fig_pf.update_traces(textposition="top center")
            st.plotly_chart(fig_pf, use_container_width=True)
    
    with tab3:
        st.header("📊 Campaign Overview")
        
        campaign_agg = filtered_df.groupby("campaign_name").agg({
            "impressions": "sum",
            "clicks": "sum",
            "spend": "sum",
            "platform": "first",
        }).reset_index()
        
        if "purchases" in filtered_df.columns:
            campaign_agg["purchases"] = filtered_df.groupby("campaign_name")["purchases"].sum().values
        if "revenue" in filtered_df.columns:
            campaign_agg["revenue"] = filtered_df.groupby("campaign_name")["revenue"].sum().values
        if "add_to_carts" in filtered_df.columns:
            campaign_agg["add_to_carts"] = filtered_df.groupby("campaign_name")["add_to_carts"].sum().values
        
        campaign_agg["CTR"] = np.where(campaign_agg["impressions"] > 0, campaign_agg["clicks"] / campaign_agg["impressions"], 0)
        campaign_agg["CPC"] = np.where(campaign_agg["clicks"] > 0, campaign_agg["spend"] / campaign_agg["clicks"], 0)
        
        if "purchases" in campaign_agg.columns:
            campaign_agg["CPA"] = np.where(campaign_agg["purchases"] > 0, campaign_agg["spend"] / campaign_agg["purchases"], 0)
        if "revenue" in campaign_agg.columns:
            campaign_agg["ROAS"] = np.where(campaign_agg["spend"] > 0, campaign_agg["revenue"] / campaign_agg["spend"], 0)
        
        campaign_agg = campaign_agg.sort_values("spend", ascending=False)
        
        display_cols = ["campaign_name", "platform", "spend", "impressions", "clicks", "CTR", "CPC"]
        if "purchases" in campaign_agg.columns:
            display_cols.extend(["purchases", "CPA"])
        if "revenue" in campaign_agg.columns:
            display_cols.extend(["revenue", "ROAS"])
        
        display_df = campaign_agg[display_cols].copy()
        display_df["CTR"] = display_df["CTR"] * 100
        
        column_config = {
            "campaign_name": st.column_config.TextColumn("Campaign"),
            "platform": st.column_config.TextColumn("Platform"),
            "spend": st.column_config.NumberColumn("Spend", format="$ %,.0f"),
            "impressions": st.column_config.NumberColumn("Impressions", format="%,d"),
            "clicks": st.column_config.NumberColumn("Clicks", format="%,d"),
            "CTR": st.column_config.NumberColumn("CTR", format="%.2f %%"),
            "CPC": st.column_config.NumberColumn("CPC", format="$ %.2f"),
        }
        if "purchases" in display_df.columns:
            column_config["purchases"] = st.column_config.NumberColumn("Purchases", format="%,d")
            column_config["CPA"] = st.column_config.NumberColumn("CPA", format="$ %.2f")
        if "revenue" in display_df.columns:
            column_config["revenue"] = st.column_config.NumberColumn("Revenue", format="$ %,.0f")
            column_config["ROAS"] = st.column_config.NumberColumn("ROAS", format="%.2f x")
        
        st.dataframe(display_df, use_container_width=True, column_config=column_config)
        
        st.markdown("---")
        st.subheader("📊 Campaign Spend Distribution")
        top_campaigns = campaign_agg.head(10)
        fig = px.bar(top_campaigns, x="campaign_name", y="spend", title="Top 10 Campaigns by Spend")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("📉 Creative Detail & Fatigue Analysis")
        
        topic_list = sorted(t for t in filtered_df["topic"].dropna().unique().tolist() if t and str(t).strip() != "")
        
        if len(topic_list) == 0:
            st.warning("No topics available with current filters. Add a 'topic' column to enable topic-based fatigue analysis.")
            st.stop()
        
        if "selected_topic" not in st.session_state or st.session_state["selected_topic"] not in topic_list:
            st.session_state["selected_topic"] = topic_list[0]
        
        selected_topic = st.selectbox(
            "Select Topic to Analyze",
            options=topic_list,
            index=topic_list.index(st.session_state["selected_topic"]),
            key="topic_detail_select",
        )
        st.session_state["selected_topic"] = selected_topic
        
        topic_data = filtered_df[filtered_df["topic"] == selected_topic].copy()
        topic_data = topic_data.sort_values("date")
        
        if topic_data.empty:
            st.warning("No data for this topic with current filters.")
            st.stop()
        
        agg_dict = {"impressions": "sum", "clicks": "sum", "spend": "sum"}
        for col in ["purchases", "revenue", "add_to_carts"]:
            if col in topic_data.columns:
                agg_dict[col] = "sum"
        
        topic_summary_row = topic_data.groupby("topic").agg(agg_dict).reset_index().iloc[0]
        
        topic_summary_row["platforms"] = ", ".join(sorted(topic_data["platform"].dropna().unique()))
        topic_summary_row["days_active"] = topic_data["date"].nunique()
        
        impr = topic_summary_row["impressions"]
        clicks = topic_summary_row["clicks"]
        spend = topic_summary_row["spend"]
        
        topic_summary_row["CTR"] = clicks / impr if impr > 0 else 0
        topic_summary_row["CPC"] = spend / clicks if clicks > 0 else 0
        
        purchases = topic_summary_row.get("purchases", 0)
        add_to_carts = topic_summary_row.get("add_to_carts", 0)
        revenue = topic_summary_row.get("revenue", 0)
        
        topic_summary_row["purchase_rate"] = purchases / clicks if clicks > 0 and purchases > 0 else 0
        topic_summary_row["add_to_cart_rate"] = add_to_carts / clicks if clicks > 0 and add_to_carts > 0 else 0
        topic_summary_row["ROAS"] = revenue / spend if spend > 0 and revenue > 0 else 0
        
        st.markdown("---")
        st.subheader("Topic Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Platform(s)", topic_summary_row["platforms"])
        with col2:
            st.metric("Total Spend", f"${topic_summary_row['spend']:,.2f}")
        with col3:
            st.metric("Impressions", f"{topic_summary_row['impressions']:,.0f}")
        with col4:
            st.metric("Days Active", f"{topic_summary_row['days_active']:.0f}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CTR", f"{topic_summary_row['CTR']:.3%}")
        with col2:
            st.metric("CPC", f"${topic_summary_row['CPC']:.2f}")
        with col3:
            if topic_summary_row.get("purchase_rate", 0) > 0:
                st.metric("CVR (Purchase)", f"{topic_summary_row['purchase_rate']:.3%}")
            else:
                st.metric("CVR (Purchase)", "N/A")
        with col4:
            if topic_summary_row.get("add_to_cart_rate", 0) > 0:
                st.metric("CVR (Add to Cart)", f"{topic_summary_row['add_to_cart_rate']:.3%}")
            else:
                st.metric("CVR (Add to Cart)", "N/A")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROAS", f"{topic_summary_row['ROAS']:.2f}x" if topic_summary_row["ROAS"] > 0 else "N/A")
        with col2:
            st.metric("Total Revenue", f"${revenue:,.0f}" if revenue > 0 else "N/A")
        
        daily_agg_dict = {"impressions": "sum", "clicks": "sum", "spend": "sum"}
        for col in ["purchases", "add_to_carts"]:
            if col in topic_data.columns:
                daily_agg_dict[col] = "sum"
        
        topic_daily = topic_data.groupby("date").agg(daily_agg_dict).reset_index().sort_values("date")
        
        topic_daily["CTR"] = np.where(topic_daily["impressions"] > 0, topic_daily["clicks"] / topic_daily["impressions"], 0)
        topic_daily["CPC"] = np.where(topic_daily["clicks"] > 0, topic_daily["spend"] / topic_daily["clicks"], 0)
        
        if "purchases" in topic_daily.columns:
            topic_daily["purchase_rate"] = np.where(topic_daily["clicks"] > 0, topic_daily["purchases"] / topic_daily["clicks"], 0)
        
        if "add_to_carts" in topic_daily.columns:
            topic_daily["add_to_cart_rate"] = np.where(topic_daily["clicks"] > 0, topic_daily["add_to_carts"] / topic_daily["clicks"], 0)
        
        topic_daily["cumulative_impressions"] = topic_daily["impressions"].cumsum()
        topic_daily["age_in_days"] = (topic_daily["date"] - topic_daily["date"].min()).dt.days
        
        st.markdown("---")
        st.subheader("Fatigue Analysis")
        
        fatigue_kpi_options = ["CTR", "CPC"]
        if "purchase_rate" in topic_daily.columns:
            fatigue_kpi_options.append("purchase_rate")
        if "add_to_cart_rate" in topic_daily.columns:
            fatigue_kpi_options.append("add_to_cart_rate")
        
        fatigue_kpi = st.selectbox("Select KPI for Fatigue Analysis", options=fatigue_kpi_options, index=0)
        secondary_kpi = st.selectbox("Optional secondary KPI (overlay)", options=["None"] + fatigue_kpi_options, index=0)
        
        if len(topic_daily) >= 3:
            age_days = topic_daily["age_in_days"].values
            kpi_values = topic_daily[fatigue_kpi].values
            
            valid_indices = ~np.isnan(kpi_values) & ~np.isinf(kpi_values)
            age_days_clean = age_days[valid_indices]
            kpi_values_clean = kpi_values[valid_indices]
            
            if len(age_days_clean) >= 3:
                coeffs = np.polyfit(age_days_clean, kpi_values_clean, 1)
                slope = coeffs[0]
                trend_line = coeffs[0] * age_days_clean + coeffs[1]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=topic_daily["date"],
                    y=topic_daily[fatigue_kpi],
                    mode="lines+markers",
                    name=f"Actual {fatigue_kpi}",
                    line=dict(width=2),
                    marker=dict(size=6),
                    yaxis="y1"
                ))
                
                fig.add_trace(go.Scatter(
                    x=topic_daily["date"].values[valid_indices],
                    y=trend_line,
                    mode="lines",
                    name="Trend Line",
                    line=dict(width=2, dash="dash"),
                    yaxis="y1"
                ))
                
                if secondary_kpi != "None":
                    fig.add_trace(go.Scatter(
                        x=topic_daily["date"],
                        y=topic_daily[secondary_kpi],
                        mode="lines+markers",
                        name=f"{secondary_kpi}",
                        line=dict(width=2, dash="dot"),
                        marker=dict(size=5),
                        yaxis="y2"
                    ))
                    
                    fig.update_layout(
                        yaxis=dict(title=fatigue_kpi),
                        yaxis2=dict(title=secondary_kpi, overlaying="y", side="right")
                    )
                else:
                    fig.update_layout(yaxis=dict(title=fatigue_kpi))
                
                fig.update_layout(
                    title=f"{fatigue_kpi} Over Time for {selected_topic}",
                    xaxis_title="Date",
                    hovermode="x unified"
                )
                
                if fatigue_kpi in RATE_METRICS_SET:
                    fig.update_layout(yaxis=dict(tickformat=".2%"))
                elif fatigue_kpi in CURRENCY_METRICS_SET:
                    fig.update_layout(yaxis=dict(tickprefix="$"))
                
                if secondary_kpi != "None":
                    if secondary_kpi in RATE_METRICS_SET:
                        fig.update_layout(yaxis2=dict(tickformat=".2%"))
                    elif secondary_kpi in CURRENCY_METRICS_SET:
                        fig.update_layout(yaxis2=dict(tickprefix="$"))
                
                st.plotly_chart(fig, use_container_width=True)
                
                rate_metrics = ["CTR", "purchase_rate", "add_to_cart_rate", "view_content_rate", "page_view_rate"]
                fatigue_threshold = -0.0001 if fatigue_kpi in rate_metrics else 0.01
                min_days_for_fatigue = 7
                min_impressions_for_fatigue = 10000
                
                total_impressions = topic_summary_row["impressions"]
                total_days = topic_summary_row["days_active"]
                
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
        
        if len(topic_daily) >= 3:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=topic_daily["cumulative_impressions"],
                y=topic_daily[fatigue_kpi],
                mode="lines+markers",
                name=f"{fatigue_kpi}",
                line=dict(width=2),
                marker=dict(size=6)
            ))
            
            cum_impr = topic_daily["cumulative_impressions"].values
            kpi_vals = topic_daily[fatigue_kpi].values
            
            valid_idx = ~np.isnan(kpi_vals) & ~np.isinf(kpi_vals)
            if np.sum(valid_idx) >= 3:
                coeffs_cum = np.polyfit(cum_impr[valid_idx], kpi_vals[valid_idx], 1)
                trend_cum = coeffs_cum[0] * cum_impr[valid_idx] + coeffs_cum[1]
                
                fig.add_trace(go.Scatter(
                    x=cum_impr[valid_idx],
                    y=trend_cum,
                    mode="lines",
                    name="Trend Line",
                    line=dict(width=2, dash="dash")
                ))
            
            fig.update_layout(
                title=f"{fatigue_kpi} vs Cumulative Impressions (Topic: {selected_topic})",
                xaxis_title="Cumulative Impressions",
                yaxis_title=fatigue_kpi,
                hovermode="x unified"
            )
            if fatigue_kpi in RATE_METRICS:
                fig.update_yaxes(tickformat=".2%")
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data points for cumulative impression analysis.")
    
    with tab5:
        st.header("🏷️ Topic Insights")
        
        creative_metrics = compute_aggregated_creative_metrics(filtered_df)
        
        if "topic" not in creative_metrics.columns or creative_metrics["topic"].isna().all():
            st.warning("No topic data available. Add a 'topic' column to your CSV to enable topic-based analysis.")
            st.info("💡 **Tip:** Topics help you group creatives by theme or content type (e.g., 'Product Demo', 'UGC Content', 'Brand Messaging').")
            st.stop()
        
        st.info("💡 Analyze creative performance by topic to identify which content themes drive the best results.")
        
        st.markdown("---")
        st.subheader("CTR vs CPC Performance by Topic")
        
        topic_level = creative_metrics[creative_metrics["topic"].notna()].copy()
        
        if len(topic_level) == 0:
            st.warning("No data available with topics after filtering.")
        else:
            topic_summary = (
                topic_level
                .groupby("topic")
                .agg(
                    impressions=("impressions", "sum"),
                    clicks=("clicks", "sum"),
                    spend=("spend", "sum"),
                    num_creatives=("creative_name", "nunique"),
                )
                .reset_index()
            )
            
            topic_summary["CTR"] = np.where(topic_summary["impressions"] > 0, topic_summary["clicks"] / topic_summary["impressions"], 0)
            topic_summary["CPC"] = np.where(topic_summary["clicks"] > 0, topic_summary["spend"] / topic_summary["clicks"], 0)
            
            fig = px.scatter(
                topic_summary,
                x="CPC",
                y="CTR",
                size="spend",
                color="topic",
                hover_data=["impressions", "clicks", "spend", "num_creatives"],
                title="CTR vs CPC Performance by Topic (bubble size = spend)",
                labels={
                    "CPC": "Cost Per Click ($)",
                    "CTR": "Click-Through Rate",
                    "topic": "Topic",
                    "num_creatives": "# of Creatives",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            
            fig.update_yaxes(tickformat=".2%")
            fig.update_layout(hovermode="closest", height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("📊 Spend by Topic & Journey Role")
        st.caption("See which topics are skewed toward top-of-funnel (Engagement), mid-funnel (Intent), or bottom-funnel (Conversion) creatives.")
        
        topic_journey_data = creative_metrics[creative_metrics["topic"].notna()].copy()
        
        if len(topic_journey_data) > 0:
            topic_journey_spend = topic_journey_data.groupby(["topic", "journey_role"]).agg({
                "spend": "sum",
                "creative_name": "count"
            }).reset_index()
            topic_journey_spend.rename(columns={"creative_name": "num_creatives"}, inplace=True)
            
            colors = {"Engagement": "#4CAF50", "Intent": "#FF9800", "Conversion": "#2196F3"}
            fig_stacked = px.bar(
                topic_journey_spend,
                x="topic",
                y="spend",
                color="journey_role",
                title="Spend Distribution by Topic and Journey Role",
                labels={"topic": "Topic", "spend": "Spend ($)", "journey_role": "Journey Role"},
                color_discrete_map=colors,
                barmode="stack"
            )
            fig_stacked.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_stacked, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🎯 Topic Performance by Layer")
            
            layer_tabs = st.tabs(["📢 Engagement", "🛒 Intent", "💰 Conversion"])
            
            with layer_tabs[0]:
                eng_topics = topic_journey_data[topic_journey_data["journey_role"] == "Engagement"]
                if len(eng_topics) > 0:
                    eng_by_topic = eng_topics.groupby("topic").agg({
                        "spend": "sum",
                        "impressions": "sum",
                        "clicks": "sum",
                        "CTR": "mean",
                        "CPC": "mean",
                        "creative_name": "count"
                    }).reset_index()
                    eng_by_topic.rename(columns={"creative_name": "num_creatives"}, inplace=True)
                    eng_by_topic = eng_by_topic.sort_values("CTR", ascending=False)
                    eng_by_topic["CTR"] = eng_by_topic["CTR"] * 100
                    st.dataframe(eng_by_topic, use_container_width=True, column_config={
                        "CTR": st.column_config.NumberColumn("Avg CTR", format="%.3f %%"),
                        "CPC": st.column_config.NumberColumn("Avg CPC", format="$ %.2f"),
                        "spend": st.column_config.NumberColumn("Spend", format="$ %,.0f"),
                    })
                else:
                    st.info("No engagement creatives with topics.")
            
            with layer_tabs[1]:
                int_topics = topic_journey_data[topic_journey_data["journey_role"] == "Intent"]
                if len(int_topics) > 0:
                    int_agg = {"spend": "sum", "impressions": "sum", "clicks": "sum", "creative_name": "count"}
                    for col in ["add_to_cart_rate", "view_content_rate", "page_view_rate"]:
                        if col in int_topics.columns:
                            int_agg[col] = "mean"
                    int_by_topic = int_topics.groupby("topic").agg(int_agg).reset_index()
                    int_by_topic.rename(columns={"creative_name": "num_creatives"}, inplace=True)
                    col_config = {"spend": st.column_config.NumberColumn("Spend", format="$ %,.0f")}
                    for col in ["add_to_cart_rate", "view_content_rate", "page_view_rate"]:
                        if col in int_by_topic.columns:
                            int_by_topic[col] = int_by_topic[col] * 100
                            col_config[col] = st.column_config.NumberColumn(col.replace("_", " ").title(), format="%.3f %%")
                    st.dataframe(int_by_topic, use_container_width=True, column_config=col_config)
                else:
                    st.info("No intent creatives with topics.")
            
            with layer_tabs[2]:
                conv_topics = topic_journey_data[topic_journey_data["journey_role"] == "Conversion"]
                if len(conv_topics) > 0:
                    conv_agg = {"spend": "sum", "impressions": "sum", "clicks": "sum", "creative_name": "count"}
                    for col in ["CVR", "CPA", "ROAS"]:
                        if col in conv_topics.columns:
                            conv_agg[col] = "mean"
                    if "purchases" in conv_topics.columns:
                        conv_agg["purchases"] = "sum"
                    conv_by_topic = conv_topics.groupby("topic").agg(conv_agg).reset_index()
                    conv_by_topic.rename(columns={"creative_name": "num_creatives"}, inplace=True)
                    col_config = {"spend": st.column_config.NumberColumn("Spend", format="$ %,.0f")}
                    if "CVR" in conv_by_topic.columns:
                        conv_by_topic["CVR"] = conv_by_topic["CVR"] * 100
                        col_config["CVR"] = st.column_config.NumberColumn("Avg CVR", format="%.3f %%")
                    if "CPA" in conv_by_topic.columns:
                        col_config["CPA"] = st.column_config.NumberColumn("Avg CPA", format="$ %.2f")
                    if "ROAS" in conv_by_topic.columns:
                        col_config["ROAS"] = st.column_config.NumberColumn("Avg ROAS", format="%.2f x")
                    if "purchases" in conv_by_topic.columns:
                        col_config["purchases"] = st.column_config.NumberColumn("Purchases", format="%,d")
                    st.dataframe(conv_by_topic, use_container_width=True, column_config=col_config)
                else:
                    st.info("No conversion creatives with topics.")

if __name__ == "__main__":
    main()
