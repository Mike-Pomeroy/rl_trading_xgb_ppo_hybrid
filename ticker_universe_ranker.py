"""
Standalone ticker universe ranking report.

Purpose:
- Take a finite list of tickers.
- Fetch split-adjusted daily bars from Alpaca.
- Rank every ticker using objective screening metrics.
- Include SPY in the ranking by default.
- Save a sorted CSV and a sorted PDF report.

Run from your project folder:
    python -u ticker_universe_ranker.py

Required .env:
    APCA_API_KEY_ID
    APCA_API_SECRET_KEY

Outputs:
    ticker_ranking_results/ticker_ranking.csv
    ticker_ranking_results/ticker_ranking_report.pdf
    ticker_ranking_results/ticker_ranking_metadata.csv

Notes:
- This script does NOT trade.
- This script does NOT submit Alpaca orders.
- This script is intentionally separate from the XGBoost trading system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from data_module import fetch_alpaca_daily_bars

# PDF generation imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
    )
except ImportError as exc:
    raise ImportError(
        "ticker_universe_ranker.py requires reportlab to create the PDF report. "
        "Install it with: pip install reportlab"
    ) from exc


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

OUTPUT_DIR = Path("ticker_ranking_results")
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PATH = OUTPUT_DIR / "ticker_ranking.csv"
PDF_PATH = OUTPUT_DIR / "ticker_ranking_report.pdf"
METADATA_PATH = OUTPUT_DIR / "ticker_ranking_metadata.csv"

DATA_ADJUSTMENT = "split"
FEED = "sip"  # If your Alpaca plan rejects this, change to "delayed_sip".

START_DATE = "2019-01-01"
END_DATE: Optional[str] = None  # None means fetch through latest available.

# Include SPY in the ranking by default.
EXCLUDE_FROM_RANKING: set[str] = set()

# Replace or extend this finite list as needed.
TICKERS: List[str] = [
    "AAPL", "MSFT", "SPY", "GOOGL", "AMZN",
    "META", "TSLA", "NVDA", "JPM", "JNJ",
    "AVGO", "LLY", "UNH", "COST", "V",
    "MA", "HD", "PG", "XOM", "AMD",
    "CAT", "WMT", "CVX", "AMAT", "MU",
]

# Eligibility filters.
MIN_HISTORY_DAYS = 750
MIN_PRICE = 10.0
MIN_AVG_DOLLAR_VOLUME_60D = 50_000_000

# Metric windows.
MOMENTUM_6M_DAYS = 126
MOMENTUM_12M_DAYS = 252
VOL_DAYS = 60
DRAWDOWN_DAYS = 252

# Score weights.
WEIGHT_MOMENTUM_6M = 0.30
WEIGHT_MOMENTUM_12M = 0.25
WEIGHT_LIQUIDITY = 0.20
WEIGHT_LOW_VOLATILITY = 0.15
WEIGHT_LOW_DRAWDOWN = 0.10


# ============================================================
# METRIC HELPERS
# ============================================================

@dataclass
class RankingConfig:
    data_adjustment: str
    feed: str
    start_date: str
    end_date: Optional[str]
    min_history_days: int
    min_price: float
    min_avg_dollar_volume_60d: float
    momentum_6m_days: int
    momentum_12m_days: int
    vol_days: int
    drawdown_days: int


def normalize_ticker_list(tickers: Iterable[str]) -> List[str]:
    clean = []

    for ticker in tickers:
        ticker = str(ticker).strip().upper()
        if ticker:
            clean.append(ticker)

    return sorted(set(clean))


def max_drawdown(values: pd.Series) -> float:
    values = values.dropna().astype(float)

    if values.empty:
        return np.nan

    peak = values.cummax()
    dd = values / peak - 1.0

    return float(dd.min())


def safe_pct_change(series: pd.Series, periods: int) -> float:
    series = series.dropna().astype(float)

    if len(series) <= periods:
        return np.nan

    start = series.iloc[-periods - 1]
    end = series.iloc[-1]

    if not np.isfinite(start) or start <= 0:
        return np.nan

    return float(end / start - 1.0)


def fetch_data(tickers: List[str]) -> pd.DataFrame:
    fetch_kwargs = {
        "tickers": tickers,
        "start_date": START_DATE,
        "adjustment": DATA_ADJUSTMENT,
        "feed": FEED,
    }

    if END_DATE is not None:
        fetch_kwargs["end_date"] = END_DATE

    raw_df = fetch_alpaca_daily_bars(**fetch_kwargs)
    raw_df["date"] = pd.to_datetime(raw_df["date"]).dt.normalize()

    return raw_df


def compute_candidate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for tic, g in df.groupby("tic"):
        g = g.sort_values("date").copy()

        close = g["close"].astype(float)
        volume = g["volume"].astype(float)
        daily_returns = close.pct_change()

        latest_date = g["date"].iloc[-1] if len(g) else pd.NaT
        latest_close = close.iloc[-1] if len(close) else np.nan
        history_days = int(close.notna().sum())

        avg_dollar_volume_60d = (
            (close * volume)
            .tail(60)
            .replace([np.inf, -np.inf], np.nan)
            .mean()
        )

        momentum_6m = safe_pct_change(close, MOMENTUM_6M_DAYS)
        momentum_12m = safe_pct_change(close, MOMENTUM_12M_DAYS)

        vol_60d = (
            daily_returns
            .tail(VOL_DAYS)
            .replace([np.inf, -np.inf], np.nan)
            .std()
        )

        drawdown_1y = max_drawdown(close.tail(DRAWDOWN_DAYS))

        rows.append({
            "ticker": tic,
            "latest_date": latest_date,
            "latest_close": float(latest_close) if pd.notna(latest_close) else np.nan,
            "history_days": history_days,
            "avg_dollar_volume_60d": (
                float(avg_dollar_volume_60d)
                if pd.notna(avg_dollar_volume_60d)
                else np.nan
            ),
            "momentum_6m": momentum_6m,
            "momentum_12m": momentum_12m,
            "volatility_60d": float(vol_60d) if pd.notna(vol_60d) else np.nan,
            "drawdown_1y": drawdown_1y,
        })

    return pd.DataFrame(rows)


def add_ranks_and_score(metrics: pd.DataFrame) -> pd.DataFrame:
    out = metrics.copy()

    out["eligible"] = True
    out.loc[out["ticker"].isin(EXCLUDE_FROM_RANKING), "eligible"] = False
    out.loc[out["history_days"] < MIN_HISTORY_DAYS, "eligible"] = False
    out.loc[out["latest_close"] < MIN_PRICE, "eligible"] = False
    out.loc[out["avg_dollar_volume_60d"] < MIN_AVG_DOLLAR_VOLUME_60D, "eligible"] = False

    required_cols = [
        "momentum_6m",
        "momentum_12m",
        "volatility_60d",
        "drawdown_1y",
    ]

    for col in required_cols:
        out.loc[out[col].isna(), "eligible"] = False

    eligible = out["eligible"]

    out["rank_momentum_6m"] = np.nan
    out["rank_momentum_12m"] = np.nan
    out["rank_liquidity"] = np.nan
    out["rank_low_volatility"] = np.nan
    out["rank_low_drawdown"] = np.nan

    out.loc[eligible, "rank_momentum_6m"] = out.loc[eligible, "momentum_6m"].rank(pct=True)
    out.loc[eligible, "rank_momentum_12m"] = out.loc[eligible, "momentum_12m"].rank(pct=True)
    out.loc[eligible, "rank_liquidity"] = out.loc[eligible, "avg_dollar_volume_60d"].rank(pct=True)

    # Lower volatility is better.
    out.loc[eligible, "rank_low_volatility"] = (-out.loc[eligible, "volatility_60d"]).rank(pct=True)

    # drawdown_1y is negative; less negative is better.
    out.loc[eligible, "rank_low_drawdown"] = out.loc[eligible, "drawdown_1y"].rank(pct=True)

    out["universe_score"] = (
        WEIGHT_MOMENTUM_6M * out["rank_momentum_6m"]
        + WEIGHT_MOMENTUM_12M * out["rank_momentum_12m"]
        + WEIGHT_LIQUIDITY * out["rank_liquidity"]
        + WEIGHT_LOW_VOLATILITY * out["rank_low_volatility"]
        + WEIGHT_LOW_DRAWDOWN * out["rank_low_drawdown"]
    )

    out.loc[~eligible, "universe_score"] = np.nan

    out = out.sort_values(
        ["eligible", "universe_score", "avg_dollar_volume_60d"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    out["rank"] = np.nan
    if out["eligible"].any():
        out.loc[out["eligible"], "rank"] = np.arange(1, int(out["eligible"].sum()) + 1)

    # Friendly column order.
    ordered_cols = [
        "rank",
        "ticker",
        "eligible",
        "universe_score",
        "latest_date",
        "latest_close",
        "momentum_6m",
        "momentum_12m",
        "avg_dollar_volume_60d",
        "volatility_60d",
        "drawdown_1y",
        "history_days",
        "rank_momentum_6m",
        "rank_momentum_12m",
        "rank_liquidity",
        "rank_low_volatility",
        "rank_low_drawdown",
    ]

    return out[ordered_cols]


# ============================================================
# REPORT HELPERS
# ============================================================


def fmt_pct(x: float, decimals: int = 1) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:.{decimals}f}%"


def fmt_num(x: float, decimals: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{decimals}f}"


def fmt_money(x: float, decimals: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"${x:,.{decimals}f}"


def fmt_dollar_volume(x: float) -> str:
    if pd.isna(x):
        return ""
    if abs(x) >= 1_000_000_000:
        return f"${x / 1_000_000_000:,.2f}B"
    if abs(x) >= 1_000_000:
        return f"${x / 1_000_000:,.1f}M"
    return f"${x:,.0f}"


def make_metadata(tickers: List[str], ranking_df: pd.DataFrame) -> pd.DataFrame:
    config = RankingConfig(
        data_adjustment=DATA_ADJUSTMENT,
        feed=FEED,
        start_date=START_DATE,
        end_date=END_DATE,
        min_history_days=MIN_HISTORY_DAYS,
        min_price=MIN_PRICE,
        min_avg_dollar_volume_60d=MIN_AVG_DOLLAR_VOLUME_60D,
        momentum_6m_days=MOMENTUM_6M_DAYS,
        momentum_12m_days=MOMENTUM_12M_DAYS,
        vol_days=VOL_DAYS,
        drawdown_days=DRAWDOWN_DAYS,
    )

    rows = [
        {"field": "run_timestamp", "value": pd.Timestamp.now().isoformat(timespec="seconds")},
        {"field": "ticker_count_requested", "value": len(tickers)},
        {"field": "ticker_count_ranked", "value": int(len(ranking_df))},
        {"field": "eligible_count", "value": int(ranking_df["eligible"].sum())},
        {"field": "data_adjustment", "value": config.data_adjustment},
        {"field": "feed", "value": config.feed},
        {"field": "start_date", "value": config.start_date},
        {"field": "end_date", "value": config.end_date or "latest_available"},
        {"field": "min_history_days", "value": config.min_history_days},
        {"field": "min_price", "value": config.min_price},
        {"field": "min_avg_dollar_volume_60d", "value": config.min_avg_dollar_volume_60d},
        {"field": "momentum_6m_days", "value": config.momentum_6m_days},
        {"field": "momentum_12m_days", "value": config.momentum_12m_days},
        {"field": "vol_days", "value": config.vol_days},
        {"field": "drawdown_days", "value": config.drawdown_days},
        {"field": "weight_momentum_6m", "value": WEIGHT_MOMENTUM_6M},
        {"field": "weight_momentum_12m", "value": WEIGHT_MOMENTUM_12M},
        {"field": "weight_liquidity", "value": WEIGHT_LIQUIDITY},
        {"field": "weight_low_volatility", "value": WEIGHT_LOW_VOLATILITY},
        {"field": "weight_low_drawdown", "value": WEIGHT_LOW_DRAWDOWN},
        {"field": "excluded_from_ranking", "value": ",".join(sorted(EXCLUDE_FROM_RANKING)) or "none"},
    ]

    return pd.DataFrame(rows)


def build_pdf_report(ranking_df: pd.DataFrame, metadata_df: pd.DataFrame, output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        rightMargin=0.35 * inch,
        leftMargin=0.35 * inch,
        topMargin=0.35 * inch,
        bottomMargin=0.35 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "HeadingCustom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10.5,
        spaceAfter=4,
    )
    small_style = ParagraphStyle(
        "SmallCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=8,
    )

    story = []
    story.append(Paragraph("Ticker Universe Ranking Report", title_style))
    story.append(Paragraph(
        "Standalone objective ranking of a finite ticker universe. SPY is included in the ranking unless you add it to EXCLUDE_FROM_RANKING.",
        body_style,
    ))
    story.append(Spacer(1, 0.08 * inch))

    # Summary metadata.
    meta_lookup = dict(zip(metadata_df["field"], metadata_df["value"]))
    summary_data = [
        ["Run timestamp", str(meta_lookup.get("run_timestamp", "")), "Data adjustment", str(meta_lookup.get("data_adjustment", ""))],
        ["Tickers requested", str(meta_lookup.get("ticker_count_requested", "")), "Feed", str(meta_lookup.get("feed", ""))],
        ["Tickers ranked", str(meta_lookup.get("ticker_count_ranked", "")), "Start date", str(meta_lookup.get("start_date", ""))],
        ["Eligible count", str(meta_lookup.get("eligible_count", "")), "End date", str(meta_lookup.get("end_date", ""))],
    ]
    summary_table = Table(summary_data, colWidths=[1.5 * inch, 2.4 * inch, 1.4 * inch, 2.2 * inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)

    story.append(Paragraph("Ranking Formula", heading_style))
    formula_text = (
        "Universe score = 30% 6M momentum rank + 25% 12M momentum rank + "
        "20% liquidity rank + 15% low-volatility rank + 10% low-drawdown rank. "
        "All component ranks are percentile ranks within this ticker list. Higher scores are better."
    )
    story.append(Paragraph(formula_text, body_style))

    story.append(Paragraph("Eligibility Filters", heading_style))
    filters_text = (
        f"Eligible tickers require at least {MIN_HISTORY_DAYS} history days, "
        f"latest close >= ${MIN_PRICE:,.2f}, and 60-day average dollar volume >= "
        f"{fmt_dollar_volume(MIN_AVG_DOLLAR_VOLUME_60D)}. Tickers with missing momentum, volatility, "
        "or drawdown metrics are marked ineligible."
    )
    story.append(Paragraph(filters_text, body_style))

    story.append(Paragraph("Ranked Tickers", heading_style))

    display_df = ranking_df.copy()
    display_df = display_df.sort_values(["eligible", "rank", "ticker"], ascending=[False, True, True])

    table_data = [[
        "Rank", "Ticker", "Eligible", "Score", "Close", "6M Mom", "12M Mom", "60D Vol", "1Y DD", "60D $Vol"
    ]]

    for _, row in display_df.iterrows():
        table_data.append([
            "" if pd.isna(row["rank"]) else str(int(row["rank"])),
            row["ticker"],
            "Y" if bool(row["eligible"]) else "N",
            fmt_num(row["universe_score"], 3),
            fmt_money(row["latest_close"], 2),
            fmt_pct(row["momentum_6m"], 1),
            fmt_pct(row["momentum_12m"], 1),
            fmt_pct(row["volatility_60d"], 2),
            fmt_pct(row["drawdown_1y"], 1),
            fmt_dollar_volume(row["avg_dollar_volume_60d"]),
        ])

    # Split into multiple tables if list is long to avoid very small type.
    header = table_data[0]
    rows = table_data[1:]
    chunk_size = 34

    for i in range(0, len(rows), chunk_size):
        if i > 0:
            story.append(PageBreak())
            story.append(Paragraph("Ranked Tickers - continued", heading_style))

        chunk = [header] + rows[i:i + chunk_size]
        table = Table(
            chunk,
            repeatRows=1,
            colWidths=[0.45 * inch, 0.62 * inch, 0.55 * inch, 0.58 * inch, 0.78 * inch,
                       0.68 * inch, 0.72 * inch, 0.68 * inch, 0.62 * inch, 0.92 * inch],
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("ALIGN", (0, 0), (0, -1), "RIGHT"),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
            ("ALIGN", (1, 0), (1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FA")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.1 * inch))

    story.append(PageBreak())
    story.append(Paragraph("Configuration Details", heading_style))
    meta_table_data = [["Field", "Value"]] + metadata_df.astype(str).values.tolist()
    meta_table = Table(meta_table_data, repeatRows=1, colWidths=[2.4 * inch, 5.6 * inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263238")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(meta_table)

    doc.build(story)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    tickers = normalize_ticker_list(TICKERS)

    print("\n===== STANDALONE TICKER UNIVERSE RANKER =====")
    print(f"Tickers requested: {len(tickers)}")
    print(f"Data adjustment:   {DATA_ADJUSTMENT}")
    print(f"Feed:              {FEED}")
    print(f"Start date:        {START_DATE}")
    print(f"End date:          {END_DATE or 'latest_available'}")
    print(f"SPY included:      {'SPY' in tickers and 'SPY' not in EXCLUDE_FROM_RANKING}")

    print("\nFetching Alpaca daily bars...")
    raw_df = fetch_data(tickers)

    print(f"Rows fetched:      {len(raw_df):,}")
    print(f"Tickers fetched:   {raw_df['tic'].nunique()}")

    missing_fetch = sorted(set(tickers) - set(raw_df["tic"].unique()))
    if missing_fetch:
        print("\nWARNING: These tickers were requested but not fetched:")
        print(", ".join(missing_fetch))

    print("\nComputing ranking metrics...")
    metrics = compute_candidate_metrics(raw_df)
    ranking = add_ranks_and_score(metrics)
    metadata = make_metadata(tickers, ranking)

    ranking.to_csv(CSV_PATH, index=False)
    metadata.to_csv(METADATA_PATH, index=False)

    print("\nBuilding PDF report...")
    build_pdf_report(ranking, metadata, PDF_PATH)

    print("\n===== TOP RANKED TICKERS =====")
    cols = [
        "rank", "ticker", "universe_score", "momentum_6m", "momentum_12m",
        "volatility_60d", "drawdown_1y", "avg_dollar_volume_60d", "latest_close"
    ]
    print(
        ranking[ranking["eligible"]]
        .head(30)[cols]
        .to_string(index=False, float_format=lambda x: f"{x:,.4f}")
    )

    print("\n===== SAVED FILES =====")
    print(f"CSV ranking: {CSV_PATH}")
    print(f"PDF report:  {PDF_PATH}")
    print(f"Metadata:    {METADATA_PATH}")


if __name__ == "__main__":
    main()
