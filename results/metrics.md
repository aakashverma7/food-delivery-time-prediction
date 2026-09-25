# Results

Run on 2026-09-26.

## Metrics

Errors are in minutes on held-out orders. The mean baseline predicts the training-set mean of `Time_taken(min)`.
Imputation and encoding are fitted on the training rows of that split only.

| Model | Features | Split | MAE (min) | RMSE (min) | R2 |
|---|---|---|---:|---:|---:|
| Mean baseline | B | random 80/20 | 7.51 | 9.28 | -0.000 |
| LinearRegression | B | random 80/20 | 4.78 | 6.05 | 0.575 |
| DecisionTree | B | random 80/20 | 3.32 | 4.25 | 0.790 |
| ElasticNet | B | random 80/20 | 4.92 | 6.23 | 0.549 |
| RandomForest | B | random 80/20 | 3.19 | 4.05 | 0.810 |
| RandomForest | A | random 80/20 | 5.93 | 7.67 | 0.317 |
| RandomForest | B | grouped by restaurant code | 3.23 | 4.08 | 0.808 |
| RandomForest | B | time (last ~20% of dates) | 3.25 | 4.06 | 0.818 |

RandomForest uses 300 trees (`random_state` 42).
DecisionTree uses `min_samples_leaf=20`.
ElasticNet uses `alpha=0.1`, `l1_ratio=0.5`, and StandardScaler on the numeric columns.

## Data

- File: `train.csv` (45593 rows x 20 columns)
- Parsed all fields; missing target after parse: 0
- Rows after requiring a numeric target: 45593
- Dropped restaurant location (0, 0): 3640 rows; 41953 remain
- `abs()` on negative restaurant latitude: 431 rows
- `abs()` on negative restaurant longitude: 162 rows (162 of these also had a negative latitude)
- Rows used for modelling (no feature-only dedupe): 41953
- Restaurant codes that did not match `[A-Z]+RES` + digits: 0

## Synthetic delivery geometry

After dropping (0, 0) and taking absolute restaurant coordinates, every remaining row has
`Delivery_location_latitude = Restaurant_latitude + offset` and
`Delivery_location_longitude = Restaurant_longitude + offset` with the same positive offset
in both coordinates.

- Offset range (degrees): 0.009999 to 0.140000
- Latitude and longitude offsets match on 100.0% of rows (41953 of 41953)
- Haversine distance: min 1.465 km, max 20.969 km

## Splits

- Seed: 42
- Main random split: 80/20 on rows (`train_test_split`, `random_state=42`). Train 33562, test 8391. Feature set A reuses these same rows.
- Grouped split: `GroupShuffleSplit` 80/20 on restaurant code, `random_state=42`. Restaurant code is the leading `[A-Z]+RES` + digits token of `Delivery_person_ID` (example: `INDORES13` from `INDORES13DEL02`; the remainder is a `DEL` + person suffix). Train 34069 rows / 312 codes, test 7884 rows / 78 codes. Codes in both sides: 0.
- Time split: unique `Order_Date` values sorted; last 9 of 44 dates are test (about 20%). Cutoff date 2022-03-29 (test dates on or after this day). Date range 2022-02-11 to 2022-04-06. Train 33102, test 8851.

Feature set B (dispatch-time): restaurant and delivery latitude/longitude, `distance_km`, weather, traffic, festival, City, courier age and rating, `Vehicle_condition`, `Type_of_order`, `Type_of_vehicle`, `multiple_deliveries`, order hour from `Time_Orderd`.
Feature set A (course set): restaurant and delivery latitude/longitude, `distance_km`, weather, traffic, festival.
Excluded from both: `Time_Order_picked`, `ID`, `Delivery_person_ID`, restaurant code, `Order_Date`.

## Environment

- Python 3.13.13
- pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2
- Runtime: 76.6 s
