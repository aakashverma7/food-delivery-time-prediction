# Food Delivery Time Prediction

This project predicts food delivery time and can rank restaurants for a target delivery location.

Data not included in this repository.

## Workflow

- cleaning a noisy delivery dataset
- feature engineering with haversine distance
- categorical encoding and regression modeling
- model comparison across linear, tree-based, and regularized methods
- optional restaurant recommendation by predicted delivery time

## Project Structure

- `src/delivery_time_prediction.py`: training, evaluation, and optional recommendation script
- `data/README.md`: expected dataset file and schema
- `requirements.txt`: Python dependencies

## Quick Start

```bash
pip install -r requirements.txt
python src/delivery_time_prediction.py --input-path data/train.csv
```

To generate a ranked restaurant list for a delivery destination:

```bash
python src/delivery_time_prediction.py \
  --input-path data/train.csv \
  --delivery-lat 12.97 \
  --delivery-lon 77.64 \
  --weather Sunny \
  --traffic High \
  --festival No
```

## Outputs

The script writes results under `outputs/`:

- `model_metrics.csv`
- `restaurant_recommendations.csv` when recommendation arguments are provided
