from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "indicators.json"
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.worldbank.org/v2"

def minmax(series: pd.Series, invert: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s_min = s.min()
    s_max = s.max()

    if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
        out = pd.Series([0.5] * len(s), index=s.index)
    else:
        out = (s - s_min) / (s_max - s_min)

    if invert:
        out = 1 - out
    return out

def fetch_indicator(country_code: str, indicator_code: str) -> pd.DataFrame:
    url = f"{API_BASE}/country/{country_code}/indicator/{indicator_code}?format=json&per_page=200"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()

    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"Unexpected API response for {indicator_code}")

    rows = payload[1]
    records = []

    for item in rows:
        year = item.get("date")
        value = item.get("value")
        if year is None or value is None:
            continue

        records.append({
            "year": int(year),
            "indicator_code": indicator_code,
            "value": float(value)
        })

    return pd.DataFrame(records)

def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    country_code = config["country_code"]
    min_year = int(config["min_year"])
    indicators = config["indicators"]

    raw_frames = []
    for name, code in indicators.items():
        df = fetch_indicator(country_code, code)
        df["indicator_name"] = name
        raw_frames.append(df)

    raw_df = pd.concat(raw_frames, ignore_index=True)
    raw_df.to_csv(RAW_DIR / "worldbank_long.csv", index=False)

    wide_df = (
        raw_df
        .pivot_table(index="year", columns="indicator_name", values="value", aggfunc="first")
        .reset_index()
        .sort_values("year")
    )

    wide_df = wide_df[wide_df["year"] >= min_year].copy()

    required_cols = list(indicators.keys())
    wide_df = wide_df.dropna(subset=required_cols).reset_index(drop=True)

    wide_df["digital_access_index"] = 100 * (
        0.5 * minmax(wide_df["internet_users_pct"]) +
        0.5 * minmax(wide_df["mobile_subscriptions_per_100"])
    )

    wide_df["financial_inclusion_index"] = 100 * (
        minmax(wide_df["bank_branches_per_100k_adults"]) +
        minmax(wide_df["atms_per_100k_adults"]) +
        minmax(wide_df["private_credit_pct_gdp"])
    ) / 3

    wide_df["macro_resilience_index"] = 100 * (
        minmax(wide_df["gdp_growth_pct"]) +
        minmax(wide_df["inflation_pct"], invert=True) +
        minmax(wide_df["unemployment_pct"], invert=True)
    ) / 3

    wide_df["overall_resilience_score"] = (
        0.35 * wide_df["digital_access_index"] +
        0.35 * wide_df["financial_inclusion_index"] +
        0.30 * wide_df["macro_resilience_index"]
    )

    wide_df["intervention_priority_score"] = 100 - wide_df["overall_resilience_score"]

    def priority_tier(score: float) -> str:
        if score >= 66:
            return "High"
        if score >= 33:
            return "Medium"
        return "Low"

    wide_df["priority_tier"] = wide_df["intervention_priority_score"].apply(priority_tier)

    def weakest_pillar(row) -> str:
        scores = {
            "Digital Access": row["digital_access_index"],
            "Financial Inclusion": row["financial_inclusion_index"],
            "Macro Resilience": row["macro_resilience_index"],
        }
        return min(scores, key=scores.get)

    wide_df["weakest_pillar"] = wide_df.apply(weakest_pillar, axis=1)
    wide_df["country"] = "Indonesia"

    final_cols = [
        "country",
        "year",
        "internet_users_pct",
        "mobile_subscriptions_per_100",
        "bank_branches_per_100k_adults",
        "atms_per_100k_adults",
        "private_credit_pct_gdp",
        "gdp_growth_pct",
        "inflation_pct",
        "unemployment_pct",
        "digital_access_index",
        "financial_inclusion_index",
        "macro_resilience_index",
        "overall_resilience_score",
        "intervention_priority_score",
        "priority_tier",
        "weakest_pillar",
    ]

    final_df = wide_df[final_cols].copy()
    final_df.to_csv(PROC_DIR / "final_dataset.csv", index=False)

    print("Saved raw long file to data/raw/worldbank_long.csv")
    print("Saved final dataset to data/processed/final_dataset.csv")
    print(final_df.tail())

if __name__ == "__main__":
    main()
