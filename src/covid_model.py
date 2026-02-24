import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
df = pd.read_excel("../data/MD_COVID_19_dataset.xlsx")

# Clean data
df = df.dropna()

# Identify identifier and feature columns
id_cols = ["OBJECTID"]

# Select numeric feature columns excluding identifiers
num_cols = [
    c for c in df.select_dtypes(include=["int64", "float64"]).columns
    if c not in id_cols
]

# Normalize only meaningful numeric features
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(df[num_cols])

# Train Random Forest model
X = df[num_cols]
y = df[num_cols[0]]  # example target variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
rmse = mean_squared_error(y_test, predictions, squared=False)

print("RMSE:", rmse)

# Save trained model
joblib.dump(model, "../covid_rf_model.pkl")