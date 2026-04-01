import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor


DROP_COLUMNS = [
    "ID",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "City",
]


def haversine(lat1, lon1, lat2, lon2):
    radius_km = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return radius_km * c


def extract_numeric(value) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not match:
        return np.nan
    return float(match.group(1))


def clean_weather(value: str) -> str:
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    parts = text.split(maxsplit=1)
    if len(parts) == 2 and parts[0].lower().startswith("weather"):
        return parts[1].strip()
    if len(parts) == 2 and parts[0].lower().startswith("conditions"):
        return parts[1].strip()
    return text


def load_dataset(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[column for column in DROP_COLUMNS if column in df.columns], errors="ignore")
    df = df.replace({"NaN": np.nan, "NaN ": np.nan, "nan": np.nan})

    if "Weatherconditions" in df.columns:
        df["Weatherconditions"] = df["Weatherconditions"].map(clean_weather)

    if "Time_taken(min)" not in df.columns:
        raise ValueError("Dataset must include a `Time_taken(min)` column.")

    df["Time_taken(min)"] = df["Time_taken(min)"].map(extract_numeric)

    numeric_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ["Weatherconditions", "Road_traffic_density", "Festival"]:
        if column in df.columns:
            df[column] = df[column].where(df[column].notna(), np.nan)
            df[column] = df[column].astype("string").str.strip()

    df = df.dropna().copy()
    df["distance"] = haversine(
        df["Restaurant_latitude"],
        df["Restaurant_longitude"],
        df["Delivery_location_latitude"],
        df["Delivery_location_longitude"],
    )
    df = df[df["distance"] <= 35].copy()

    dedupe_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Weatherconditions",
        "Road_traffic_density",
        "Festival",
        "distance",
    ]
    dedupe_columns = [column for column in dedupe_columns if column in df.columns]
    df = df.drop_duplicates(subset=dedupe_columns).reset_index(drop=True)
    return df


def build_feature_matrix(df: pd.DataFrame):
    target = "Time_taken(min)"
    feature_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "Weatherconditions",
        "Road_traffic_density",
        "Festival",
        "distance",
    ]
    feature_columns = [column for column in feature_columns if column in df.columns]
    X = df[feature_columns]
    y = df[target]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [column for column in X.columns if column not in categorical_cols]

    numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        sparse_threshold=0.0,
    )


def evaluate_regressor(model, X_train, X_test, y_train, y_test, name: str) -> dict:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return {
        "model": name,
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }


def rank_restaurants(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    model: RandomForestRegressor,
    delivery_lat: float,
    delivery_lon: float,
    weather: str,
    traffic: str,
    festival: str,
    top_n: int,
) -> pd.DataFrame:
    restaurants = df[["Restaurant_latitude", "Restaurant_longitude"]].drop_duplicates().copy()
    restaurants["Delivery_location_latitude"] = delivery_lat
    restaurants["Delivery_location_longitude"] = delivery_lon
    restaurants["Weatherconditions"] = weather
    restaurants["Road_traffic_density"] = traffic
    restaurants["Festival"] = festival
    restaurants["distance"] = haversine(
        restaurants["Restaurant_latitude"],
        restaurants["Restaurant_longitude"],
        restaurants["Delivery_location_latitude"],
        restaurants["Delivery_location_longitude"],
    )
    restaurants = restaurants[restaurants["distance"] <= 35].copy()
    encoded = preprocessor.transform(restaurants)
    restaurants["predicted_delivery_time_min"] = model.predict(encoded)
    return restaurants.sort_values("predicted_delivery_time_min").head(top_n)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train delivery time models and rank restaurants.")
    parser.add_argument("--input-path", required=True, help="Path to the training CSV.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for exported outputs.")
    parser.add_argument("--delivery-lat", type=float, help="Optional delivery latitude for restaurant ranking.")
    parser.add_argument("--delivery-lon", type=float, help="Optional delivery longitude for restaurant ranking.")
    parser.add_argument("--weather", help="Weather value for ranking, for example `Sunny`.")
    parser.add_argument("--traffic", help="Traffic value for ranking, for example `High`.")
    parser.add_argument("--festival", help="Festival flag for ranking, for example `No`.")
    parser.add_argument("--top-n", type=int, default=10, help="Number of restaurant options to export.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataframe(load_dataset(Path(args.input_path)))
    X, y = build_feature_matrix(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X_train)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    rows = []
    rows.append(
        evaluate_regressor(
            LinearRegression(), X_train_encoded, X_test_encoded, y_train, y_test, "linear_regression"
        )
    )
    rows.append(
        evaluate_regressor(
            DecisionTreeRegressor(random_state=42),
            X_train_encoded,
            X_test_encoded,
            y_train,
            y_test,
            "decision_tree",
        )
    )
    rows.append(
        evaluate_regressor(
            ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42),
            X_train_encoded,
            X_test_encoded,
            y_train,
            y_test,
            "elastic_net",
        )
    )

    random_forest = RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    rows.append(
        evaluate_regressor(
            random_forest, X_train_encoded, X_test_encoded, y_train, y_test, "random_forest"
        )
    )

    metrics = pd.DataFrame(rows).sort_values("rmse")
    metrics.to_csv(output_dir / "model_metrics.csv", index=False)

    if all(
        value is not None
        for value in [args.delivery_lat, args.delivery_lon, args.weather, args.traffic, args.festival]
    ):
        random_forest.fit(X_train_encoded, y_train)
        recommendations = rank_restaurants(
            df=df,
            preprocessor=preprocessor,
            model=random_forest,
            delivery_lat=args.delivery_lat,
            delivery_lon=args.delivery_lon,
            weather=args.weather,
            traffic=args.traffic,
            festival=args.festival,
            top_n=args.top_n,
        )
        recommendations.to_csv(output_dir / "restaurant_recommendations.csv", index=False)

    print(f"Saved outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
