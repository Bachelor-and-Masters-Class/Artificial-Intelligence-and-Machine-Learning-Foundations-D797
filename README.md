# COVID-19 Clustering and Prediction

**WGU D797 · Artificial Intelligence and Machine Learning Foundations**

Takes Maryland's public COVID-19 data, groups similar records together, then trains a model to predict a numeric outcome from it.

## How it works

1. **Clean:** loads the Excel file and drops rows with missing values.
2. **Pick features:** removes ID columns like `OBJECTID`, because IDs don't predict anything and would fake a better score.
3. **Scale:** squeezes every number into a 0 to 1 range, so big numbers don't drown out small ones.
4. **Cluster:** K-Means splits the data into 3 groups. Each row's group is added back as a new feature.
5. **Predict:** a Random Forest (100 trees) trains on 80% of the data and is tested on the other 20%.
6. **Score:** reports RMSE (average error, in the same units as the target).
7. **Save:** writes the trained model to disk with `joblib` so it can be reused without retraining.

## Run it

```
python src/covid_model.py
```

Prints the RMSE and saves the model.

## Files

- `src/covid_model.py`: the whole pipeline
- `data/MD_COVID_19_dataset.xlsx`: the dataset

**Built with:** Python, pandas, scikit-learn, joblib
