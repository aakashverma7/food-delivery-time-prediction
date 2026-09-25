from datetime import date, timedelta

import numpy as np
import pandas as pd

import delivery_time_prediction as dtp


def synthetic_orders(n: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    restaurants = [
        ("INDORES13", 22.75, 75.89),
        ("BANGRES18", 12.91, 77.68),
        ("HYDRES05", 17.43, 78.40),
        ("PUNERES20", 18.52, 73.86),
        ("MUMRES15", 19.12, 72.85),
        ("CHENRES12", 13.06, 80.25),
        ("JAPRES01", 26.91, 75.79),
        ("KOLRES08", 22.57, 88.36),
    ]
    weathers = ["Sunny", "Stormy", "Fog", "Cloudy"]
    traffic = ["Low", "Medium", "High", "Jam"]
    cities = ["Urban", "Metropolitian", "Semi-Urban"]
    orders = ["Snack", "Drinks", "Meal", "Buffet"]
    vehicles = ["motorcycle", "scooter", "electric_scooter"]
    start = date(2022, 3, 1)
    rows = []
    for i in range(n):
        code, lat, lon = restaurants[i % len(restaurants)]
        if i < 4:
            lat, lon = 0.0, 0.0
        elif i < 8:
            lat = -abs(lat)
            lon = -abs(lon)
        offset = 0.02 + 0.01 * (i % 5)
        weather = weathers[i % len(weathers)]
        age = 22 + (i % 15)
        rating = 3.5 + (i % 15) / 10
        hour = 8 + (i % 14)
        minutes = (i * 5) % 60
        rows.append(
            {
                "ID": f"0x{i:04x} ",
                "Delivery_person_ID": f"{code}DEL01 ",
                "Delivery_person_Age": "NaN " if i % 17 == 0 else str(age),
                "Delivery_person_Ratings": "NaN " if i % 19 == 0 else f"{rating:.1f}",
                "Restaurant_latitude": lat,
                "Restaurant_longitude": lon,
                "Delivery_location_latitude": abs(lat) + offset,
                "Delivery_location_longitude": abs(lon) + offset,
                "Order_Date": (start + timedelta(days=i % 10)).strftime("%d-%m-%Y"),
                "Time_Orderd": "NaN " if i % 23 == 0 else f"{hour:02d}:{minutes:02d}:00",
                "Time_Order_picked": f"{hour:02d}:{(minutes + 10) % 60:02d}:00",
                "Weatherconditions": "conditions NaN" if i % 21 == 0 else f"conditions {weather}",
                "Road_traffic_density": "NaN " if i % 29 == 0 else f"{traffic[i % len(traffic)]} ",
                "Vehicle_condition": i % 4,
                "Type_of_order": f"{orders[i % len(orders)]} ",
                "Type_of_vehicle": f"{vehicles[i % len(vehicles)]} ",
                "multiple_deliveries": "NaN " if i % 13 == 0 else str(i % 3),
                "Festival": "NaN " if i % 31 == 0 else ("Yes " if i % 11 == 0 else "No "),
                "City": "NaN " if i % 37 == 0 else f"{cities[i % len(cities)]} ",
                "Time_taken(min)": f"(min) {int(16 + offset * 80 + rng.integers(0, 8))}",
            }
        )
    return pd.DataFrame(rows)


def test_pipeline_runs_on_synthetic_data(tmp_path):
    csv_path = tmp_path / "train.csv"
    synthetic_orders().to_csv(csv_path, index=False)
    results = tmp_path / "results"

    rows = dtp.main(
        [
            str(csv_path),
            "--results-dir",
            str(results),
            "--n-estimators",
            "8",
            "--n-jobs",
            "1",
        ]
    )

    assert [row["Model"] for row in rows[:5]] == [
        "Mean baseline",
        "LinearRegression",
        "DecisionTree",
        "ElasticNet",
        "RandomForest",
    ]
    assert rows[5]["Features"] == "A"
    assert rows[6]["Split"] == "grouped by restaurant code"
    assert rows[7]["Split"] == "time (last ~20% of dates)"
    for row in rows:
        assert np.isfinite([row["MAE (min)"], row["RMSE (min)"], row["R2"]]).all()

    expected = {"metrics.md", "pred_vs_actual.png", "mae_by_traffic.png"}
    assert {p.name for p in results.iterdir()} == expected
    assert all((results / name).stat().st_size > 0 for name in expected)
    report = (results / "metrics.md").read_text(encoding="utf-8")
    assert "Mean baseline" in report
    assert "RandomForest" in report
    assert "8 trees" in report
    assert "nan" not in report.lower()
