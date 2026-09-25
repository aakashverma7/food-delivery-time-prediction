import argparse
import platform
import random
import re
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

SEED = 42
TEST_SIZE = 0.2
DT_MIN_SAMPLES_LEAF = 20
ELASTIC_ALPHA = 0.1
ELASTIC_L1 = 0.5
DEFAULT_N_ESTIMATORS = 300
EARTH_RADIUS_KM = 6371.0
RESTAURANT_CODE_RE = re.compile(r"^([A-Z]+RES\d+)")
TARGET = "Time_taken(min)"
PACKAGES = ["pandas", "numpy", "scikit-learn", "matplotlib"]

NUMERIC_B = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "distance_km",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "order_hour",
]
CATEGORICAL_B = [
    "weather",
    "traffic",
    "festival",
    "City",
    "Type_of_order",
    "Type_of_vehicle",
]
NUMERIC_A = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "distance_km",
]
CATEGORICAL_A = ["weather", "traffic", "festival"]
FEATURES_B = NUMERIC_B + CATEGORICAL_B
FEATURES_A = NUMERIC_A + CATEGORICAL_A


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.asarray(v, dtype=float) for v in (lat1, lon1, lat2, lon2))
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def blank_to_nan(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.mask(text.str.lower().isin(["nan", "none", ""]))


def parse_target(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def parse_weather(series: pd.Series) -> pd.Series:
    text = blank_to_nan(series)
    cleaned = text.str.replace(r"(?i)^conditions\s+", "", regex=True).str.strip()
    return cleaned.mask(cleaned.str.lower().isin(["nan", "none", ""]))


def parse_hour(series: pd.Series) -> pd.Series:
    text = blank_to_nan(series)
    parsed = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    return parsed.dt.hour.astype("float64")


def restaurant_code(series: pd.Series) -> pd.Series:
    return blank_to_nan(series).str.upper().str.extract(RESTAURANT_CODE_RE, expand=False)


def load_and_clean(path: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(path)
    counts = {
        "file": path.name,
        "n_raw": len(raw),
        "n_cols": raw.shape[1],
    }
    print(f"loaded {path.name}: {counts['n_raw']} rows x {counts['n_cols']} columns")

    df = pd.DataFrame(
        {
            "restaurant_code": restaurant_code(raw["Delivery_person_ID"]),
            "Delivery_person_Age": pd.to_numeric(
                blank_to_nan(raw["Delivery_person_Age"]), errors="coerce"
            ),
            "Delivery_person_Ratings": pd.to_numeric(
                blank_to_nan(raw["Delivery_person_Ratings"]), errors="coerce"
            ),
            "Restaurant_latitude": pd.to_numeric(raw["Restaurant_latitude"], errors="coerce"),
            "Restaurant_longitude": pd.to_numeric(raw["Restaurant_longitude"], errors="coerce"),
            "Delivery_location_latitude": pd.to_numeric(
                raw["Delivery_location_latitude"], errors="coerce"
            ),
            "Delivery_location_longitude": pd.to_numeric(
                raw["Delivery_location_longitude"], errors="coerce"
            ),
            "order_date": pd.to_datetime(
                blank_to_nan(raw["Order_Date"]), format="%d-%m-%Y", errors="coerce"
            ),
            "order_hour": parse_hour(raw["Time_Orderd"]),
            "weather": parse_weather(raw["Weatherconditions"]),
            "traffic": blank_to_nan(raw["Road_traffic_density"]),
            "Vehicle_condition": pd.to_numeric(raw["Vehicle_condition"], errors="coerce"),
            "Type_of_order": blank_to_nan(raw["Type_of_order"]),
            "Type_of_vehicle": blank_to_nan(raw["Type_of_vehicle"]),
            "multiple_deliveries": pd.to_numeric(
                blank_to_nan(raw["multiple_deliveries"]), errors="coerce"
            ),
            "festival": blank_to_nan(raw["Festival"]),
            "City": blank_to_nan(raw["City"]),
            TARGET: parse_target(raw[TARGET]),
        }
    )
    counts["n_parsed"] = len(df)
    counts["n_missing_target"] = int(df[TARGET].isna().sum())
    counts["n_code_unparsed"] = int(df["restaurant_code"].isna().sum())
    print(f"parsed fields: {counts['n_parsed']} rows, missing target {counts['n_missing_target']}")

    before = len(df)
    df = df.dropna(subset=[TARGET]).copy()
    counts["n_after_target"] = len(df)
    if len(df) != before:
        print(f"dropped missing target: {before - len(df)} rows, {len(df)} remain")

    zero = (df["Restaurant_latitude"] == 0) & (df["Restaurant_longitude"] == 0)
    counts["n_zero_restaurant"] = int(zero.sum())
    df = df.loc[~zero].copy()
    counts["n_after_zero"] = len(df)
    print(
        f"dropped restaurant (0, 0): {counts['n_zero_restaurant']} rows, "
        f"{counts['n_after_zero']} remain"
    )

    neg_lat = df["Restaurant_latitude"] < 0
    neg_lon = df["Restaurant_longitude"] < 0
    counts["n_neg_lat"] = int(neg_lat.sum())
    counts["n_neg_lon"] = int(neg_lon.sum())
    counts["n_neg_both"] = int((neg_lat & neg_lon).sum())
    df["Restaurant_latitude"] = df["Restaurant_latitude"].abs()
    df["Restaurant_longitude"] = df["Restaurant_longitude"].abs()
    print(f"abs() negative restaurant latitude: {counts['n_neg_lat']} rows")
    print(f"abs() negative restaurant longitude: {counts['n_neg_lon']} rows")

    lat_off = df["Delivery_location_latitude"] - df["Restaurant_latitude"]
    lon_off = df["Delivery_location_longitude"] - df["Restaurant_longitude"]
    equal = np.isclose(lat_off, lon_off, atol=1e-5, equal_nan=False)
    counts["offset_equal_share"] = float(equal.mean()) if len(df) else float("nan")
    counts["offset_min_deg"] = float(np.nanmin(lat_off)) if len(df) else float("nan")
    counts["offset_max_deg"] = float(np.nanmax(lat_off)) if len(df) else float("nan")
    counts["lon_offset_min_deg"] = float(np.nanmin(lon_off)) if len(df) else float("nan")
    counts["lon_offset_max_deg"] = float(np.nanmax(lon_off)) if len(df) else float("nan")
    counts["n_positive_offsets"] = int(((lat_off > 0) & (lon_off > 0)).sum())

    df["distance_km"] = haversine_km(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    counts["n_model"] = len(df)
    counts["distance_min_km"] = float(df["distance_km"].min()) if len(df) else float("nan")
    counts["distance_max_km"] = float(df["distance_km"].max()) if len(df) else float("nan")
    print(
        f"haversine distance_km on {counts['n_model']} rows; max {counts['distance_max_km']:.3f} km"
    )
    print(
        f"delivery minus restaurant offset: lat "
        f"{counts['offset_min_deg']:.6f} to {counts['offset_max_deg']:.6f} deg, "
        f"lon {counts['lon_offset_min_deg']:.6f} to {counts['lon_offset_max_deg']:.6f} deg; "
        f"lat==lon on {counts['offset_equal_share']:.1%} of rows"
    )
    for column in CATEGORICAL_B:
        values = df[column].astype(object)
        df[column] = values.where(pd.notna(values), np.nan)
    return df.reset_index(drop=True), counts


def make_regressor(name: str, n_estimators: int, n_jobs: int):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "DecisionTree":
        return DecisionTreeRegressor(min_samples_leaf=DT_MIN_SAMPLES_LEAF, random_state=SEED)
    if name == "ElasticNet":
        return ElasticNet(
            alpha=ELASTIC_ALPHA,
            l1_ratio=ELASTIC_L1,
            random_state=SEED,
            max_iter=10000,
        )
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=SEED,
            n_jobs=n_jobs,
        )
    raise ValueError(name)


def build_pipeline(numeric_cols, categorical_cols, model, scale: bool) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), list(numeric_cols)),
            ("cat", categorical, list(categorical_cols)),
        ]
    )
    return Pipeline([("prep", prep), ("model", model)])


def score_arrays(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate(
    name: str,
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_cols,
    categorical_cols,
    scale: bool = False,
) -> tuple[dict, np.ndarray]:
    if name == "Mean baseline":
        estimator = DummyRegressor(strategy="mean")
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
    else:
        pipe = build_pipeline(numeric_cols, categorical_cols, model, scale=scale)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
    metrics = score_arrays(y_test, pred)
    metrics["model"] = name
    return metrics, np.asarray(pred)


def split_random(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(), test_size=TEST_SIZE, random_state=SEED
    )
    return train_idx, test_idx


def split_grouped(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df, groups=df["restaurant_code"]))
    return df.index.to_numpy()[train_idx], df.index.to_numpy()[test_idx]


def split_time(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    dates = np.sort(df["order_date"].dropna().unique())
    n_test_dates = max(1, round(len(dates) * TEST_SIZE))
    cutoff = pd.Timestamp(dates[-n_test_dates])
    train_mask = df["order_date"] < cutoff
    test_mask = df["order_date"] >= cutoff
    info = {
        "n_unique_dates": len(dates),
        "n_test_dates": n_test_dates,
        "cutoff": cutoff.date().isoformat(),
        "first_date": pd.Timestamp(dates[0]).date().isoformat(),
        "last_date": pd.Timestamp(dates[-1]).date().isoformat(),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    return df.index.to_numpy()[train_mask], df.index.to_numpy()[test_mask], info


def xy(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    return df.loc[:, features], df[TARGET]


def row_result(model, features, split, n_train, n_test, scores) -> dict:
    return {
        "Model": model,
        "Features": features,
        "Split": split,
        "n_train": n_train,
        "n_test": n_test,
        "MAE (min)": scores["MAE"],
        "RMSE (min)": scores["RMSE"],
        "R2": scores["R2"],
    }


def metrics_table(rows: list[dict]) -> str:
    header = "| Model | Features | Split | MAE (min) | RMSE (min) | R2 |"
    sep = "|---|---|---|---:|---:|---:|"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['Model']} | {row['Features']} | {row['Split']} | "
            f"{row['MAE (min)']:.2f} | {row['RMSE (min)']:.2f} | {row['R2']:.3f} |"
        )
    return "\n".join(lines)


def plot_pred_vs_actual(y_true, y_pred, path: Path) -> None:
    fig = Figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111)
    ax.scatter(y_true, y_pred, s=8, alpha=0.2, c="tab:blue", linewidths=0)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("Actual time (min)")
    ax.set_ylabel("Predicted time (min)")
    ax.set_title("RandomForest, feature set B, random 80/20")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_mae_by_traffic(traffic, y_true, y_pred, path: Path) -> None:
    err = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    frame = pd.DataFrame({"traffic": traffic.to_numpy(), "ae": err})
    frame = frame.dropna(subset=["traffic"])
    order = [label for label in ["Low", "Medium", "High", "Jam"] if label in set(frame["traffic"])]
    extra = [label for label in frame["traffic"].unique() if label not in order]
    order.extend(sorted(extra))
    grouped = frame.groupby("traffic", observed=False)["ae"].agg(["mean", "size"]).reindex(order)
    fig = Figure(figsize=(6.8, 4.4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(grouped)), grouped["mean"], color="tab:orange")
    ax.set_xticks(range(len(grouped)), list(grouped.index))
    ax.set_ylabel("MAE (min)")
    ax.set_xlabel("Road traffic density")
    ax.set_title("RandomForest test MAE by traffic, feature set B")
    for i, (mae, n) in enumerate(zip(grouped["mean"], grouped["size"])):
        if pd.notna(mae):
            ax.text(i, mae, f" {mae:.2f}\nn={int(n)}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def write_metrics(path: Path, payload: dict) -> None:
    counts = payload["counts"]
    time_info = payload["time_info"]
    grouped = payload["grouped_info"]
    versions = ", ".join(f"{pkg} {version(pkg)}" for pkg in PACKAGES)
    text = f"""# Results

Run on {payload["run_date"]}.

## Metrics

Errors are in minutes on held-out orders. The mean baseline predicts the training-set mean of `{TARGET}`.
Imputation and encoding are fitted on the training rows of that split only.

{payload["table"]}

RandomForest uses {payload["n_estimators"]} trees (`random_state` {SEED}).
DecisionTree uses `min_samples_leaf={DT_MIN_SAMPLES_LEAF}`.
ElasticNet uses `alpha={ELASTIC_ALPHA}`, `l1_ratio={ELASTIC_L1}`, and StandardScaler on the numeric columns.

## Data

- File: `{counts["file"]}` ({counts["n_raw"]} rows x {counts["n_cols"]} columns)
- Parsed all fields; missing target after parse: {counts["n_missing_target"]}
- Rows after requiring a numeric target: {counts["n_after_target"]}
- Dropped restaurant location (0, 0): {counts["n_zero_restaurant"]} rows; {counts["n_after_zero"]} remain
- `abs()` on negative restaurant latitude: {counts["n_neg_lat"]} rows
- `abs()` on negative restaurant longitude: {counts["n_neg_lon"]} rows ({counts["n_neg_both"]} of these also had a negative latitude)
- Rows used for modelling (no feature-only dedupe): {counts["n_model"]}
- Restaurant codes that did not match `[A-Z]+RES` + digits: {counts["n_code_unparsed"]}

## Synthetic delivery geometry

After dropping (0, 0) and taking absolute restaurant coordinates, every remaining row has
`Delivery_location_latitude = Restaurant_latitude + offset` and
`Delivery_location_longitude = Restaurant_longitude + offset` with the same positive offset
in both coordinates.

- Offset range (degrees): {counts["offset_min_deg"]:.6f} to {counts["offset_max_deg"]:.6f}
- Latitude and longitude offsets match on {counts["offset_equal_share"]:.1%} of rows ({counts["n_positive_offsets"]} of {counts["n_model"]})
- Haversine distance: min {counts["distance_min_km"]:.3f} km, max {counts["distance_max_km"]:.3f} km

## Splits

- Seed: {SEED}
- Main random split: 80/20 on rows (`train_test_split`, `random_state={SEED}`). Train {payload["n_random_train"]}, test {payload["n_random_test"]}. Feature set A reuses these same rows.
- Grouped split: `GroupShuffleSplit` 80/20 on restaurant code, `random_state={SEED}`. Restaurant code is the leading `[A-Z]+RES` + digits token of `Delivery_person_ID` (example: `INDORES13` from `INDORES13DEL02`; the remainder is a `DEL` + person suffix). Train {grouped["n_train"]} rows / {grouped["n_train_groups"]} codes, test {grouped["n_test"]} rows / {grouped["n_test_groups"]} codes. Codes in both sides: {grouped["n_overlap"]}.
- Time split: unique `Order_Date` values sorted; last {time_info["n_test_dates"]} of {time_info["n_unique_dates"]} dates are test (about 20%). Cutoff date {time_info["cutoff"]} (test dates on or after this day). Date range {time_info["first_date"]} to {time_info["last_date"]}. Train {time_info["n_train"]}, test {time_info["n_test"]}.

Feature set B (dispatch-time): restaurant and delivery latitude/longitude, `distance_km`, weather, traffic, festival, City, courier age and rating, `Vehicle_condition`, `Type_of_order`, `Type_of_vehicle`, `multiple_deliveries`, order hour from `Time_Orderd`.
Feature set A (course set): restaurant and delivery latitude/longitude, `distance_km`, weather, traffic, festival.
Excluded from both: `Time_Order_picked`, `ID`, `Delivery_person_ID`, restaurant code, `Order_Date`.

## Environment

- Python {platform.python_version()}
- {versions}
- Runtime: {payload["runtime_s"]:.1f} s
"""
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path, help="Path to train.csv")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
        help=f"RandomForest trees (default {DEFAULT_N_ESTIMATORS})",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.n_estimators < 1:
        parser.error("--n-estimators must be at least 1")
    return args


def main(argv: list[str] | None = None) -> list[dict]:
    args = parse_args(argv)
    started = time.perf_counter()
    random.seed(SEED)
    np.random.seed(SEED)

    df, counts = load_and_clean(args.csv_path)
    if df.empty:
        raise ValueError("No rows left after cleaning")

    random_train, random_test = split_random(df)
    grouped_train, grouped_test = split_grouped(df)
    time_train, time_test, time_info = split_time(df)

    grouped_info = {
        "n_train": len(grouped_train),
        "n_test": len(grouped_test),
        "n_train_groups": int(df.loc[grouped_train, "restaurant_code"].nunique()),
        "n_test_groups": int(df.loc[grouped_test, "restaurant_code"].nunique()),
        "n_overlap": len(
            set(df.loc[grouped_train, "restaurant_code"])
            & set(df.loc[grouped_test, "restaurant_code"])
        ),
    }
    print(f"random split: train {len(random_train)}, test {len(random_test)}")
    print(
        f"grouped split: train {grouped_info['n_train']} "
        f"({grouped_info['n_train_groups']} codes), "
        f"test {grouped_info['n_test']} ({grouped_info['n_test_groups']} codes)"
    )
    print(
        f"time split: cutoff {time_info['cutoff']}, "
        f"train {time_info['n_train']}, test {time_info['n_test']}"
    )

    y = df[TARGET]
    xb, yb = xy(df, FEATURES_B)
    xa, ya = xy(df, FEATURES_A)

    rows: list[dict] = []
    rf_pred = None

    main_models = [
        ("Mean baseline", None, False),
        (
            "LinearRegression",
            make_regressor("LinearRegression", args.n_estimators, args.n_jobs),
            False,
        ),
        ("DecisionTree", make_regressor("DecisionTree", args.n_estimators, args.n_jobs), False),
        ("ElasticNet", make_regressor("ElasticNet", args.n_estimators, args.n_jobs), True),
        ("RandomForest", make_regressor("RandomForest", args.n_estimators, args.n_jobs), False),
    ]
    for name, model, scale in main_models:
        scores, pred = evaluate(
            name,
            model,
            xb.loc[random_train],
            xb.loc[random_test],
            yb.loc[random_train],
            yb.loc[random_test],
            NUMERIC_B,
            CATEGORICAL_B,
            scale=scale,
        )
        rows.append(
            row_result(
                name,
                "B",
                "random 80/20",
                len(random_train),
                len(random_test),
                scores,
            )
        )
        if name == "RandomForest":
            rf_pred = pred

    scores_a, _ = evaluate(
        "RandomForest",
        make_regressor("RandomForest", args.n_estimators, args.n_jobs),
        xa.loc[random_train],
        xa.loc[random_test],
        ya.loc[random_train],
        ya.loc[random_test],
        NUMERIC_A,
        CATEGORICAL_A,
    )
    rows.append(
        row_result(
            "RandomForest",
            "A",
            "random 80/20",
            len(random_train),
            len(random_test),
            scores_a,
        )
    )

    scores_g, _ = evaluate(
        "RandomForest",
        make_regressor("RandomForest", args.n_estimators, args.n_jobs),
        xb.loc[grouped_train],
        xb.loc[grouped_test],
        yb.loc[grouped_train],
        yb.loc[grouped_test],
        NUMERIC_B,
        CATEGORICAL_B,
    )
    rows.append(
        row_result(
            "RandomForest",
            "B",
            "grouped by restaurant code",
            len(grouped_train),
            len(grouped_test),
            scores_g,
        )
    )

    scores_t, _ = evaluate(
        "RandomForest",
        make_regressor("RandomForest", args.n_estimators, args.n_jobs),
        xb.loc[time_train],
        xb.loc[time_test],
        yb.loc[time_train],
        yb.loc[time_test],
        NUMERIC_B,
        CATEGORICAL_B,
    )
    rows.append(
        row_result(
            "RandomForest",
            "B",
            "time (last ~20% of dates)",
            len(time_train),
            len(time_test),
            scores_t,
        )
    )

    table = metrics_table(rows)
    print()
    print(table)
    print()

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    y_test = y.loc[random_test]
    plot_pred_vs_actual(y_test.to_numpy(), rf_pred, results_dir / "pred_vs_actual.png")
    plot_mae_by_traffic(
        df.loc[random_test, "traffic"],
        y_test,
        rf_pred,
        results_dir / "mae_by_traffic.png",
    )

    run_date = datetime.now().astimezone().date().isoformat()
    runtime_s = time.perf_counter() - started
    write_metrics(
        results_dir / "metrics.md",
        {
            "run_date": run_date,
            "table": table,
            "counts": counts,
            "time_info": time_info,
            "grouped_info": grouped_info,
            "n_random_train": len(random_train),
            "n_random_test": len(random_test),
            "n_estimators": args.n_estimators,
            "runtime_s": runtime_s,
        },
    )
    print(f"wrote {results_dir / 'metrics.md'}, pred_vs_actual.png, mae_by_traffic.png")
    print(f"runtime: {runtime_s:.1f} s")
    return rows


if __name__ == "__main__":
    main(sys.argv[1:])
