# Data

`train.csv` is not in this repository. Download it from the Kaggle dataset
[Food Delivery Dataset](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset)
(Gaurav Malik) and place it here as `train.csv`.

The archive also contains `test.csv` (no target) and `Sample_Submission.csv`. Do not use them.

Kaggle's license field is **Other**. The dataset page does not state terms or origin. Do not
redistribute the file.

## Schema (`train.csv`)

| Column | Role | Notes |
|---|---|---|
| `ID` | unused | Order id. Trailing spaces in the file. |
| `Delivery_person_ID` | split only | Looks like `INDORES13DEL02`. Restaurant code is the leading `[A-Z]+RES` + digits token (`INDORES13`). Not a model feature. |
| `Delivery_person_Age` | feature B | Text in the file; some cells are the token `NaN `. |
| `Delivery_person_Ratings` | feature B | Same junk token. |
| `Restaurant_latitude` | feature | Some rows are `(0, 0)`; some latitudes are negated (India). |
| `Restaurant_longitude` | feature | A subset of the negated-latitude rows also have a negated longitude. |
| `Delivery_location_latitude` | feature | Restaurant latitude (after `abs`) plus a positive offset. |
| `Delivery_location_longitude` | feature | Restaurant longitude (after `abs`) plus the same offset. |
| `Order_Date` | time split only | `DD-MM-YYYY`. Not a model feature. |
| `Time_Orderd` | feature B | `HH:MM:SS`; parsed to hour. Some cells are `NaN `. |
| `Time_Order_picked` | unused | Excluded. |
| `Weatherconditions` | feature | Values like `conditions Sunny` or `conditions NaN`. |
| `Road_traffic_density` | feature | `Low`, `Medium`, `High`, `Jam`, plus `NaN ` and trailing spaces. |
| `Vehicle_condition` | feature B | Integer 0–3. |
| `Type_of_order` | feature B | `Snack`, `Drinks`, `Meal`, `Buffet` (trailing spaces). |
| `Type_of_vehicle` | feature B | `motorcycle`, `scooter`, `electric_scooter`, `bicycle` (trailing spaces). |
| `multiple_deliveries` | feature B | `0`–`3` as text, plus `NaN `. |
| `Festival` | feature | `Yes`, `No`, plus `NaN `. |
| `City` | feature B | `Urban`, `Metropolitian` (spelling in the file), `Semi-Urban`, plus `NaN `. |
| `Time_taken(min)` | target | Values like `(min) 24`. |

Feature set A uses coordinates, haversine `distance_km`, weather, traffic and festival only.
