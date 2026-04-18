# Challenge Documentation

## Part I: Model

### Bug Fix
The original `model.py` had a syntax error in the type hint:
```python
# Before (invalid)
Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame)

# After (correct)
Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]
```

### Model Selection
The Data Scientist proposed 6 model variants. Based on the notebook analysis, the final candidates were narrowed to models using the **top 10 features** and **class balancing**, since:
- Reducing to top 10 features (from XGBoost feature importance) does not hurt performance.
- Class balancing significantly improves recall for class `1` (delayed flights), which is the operationally critical class.

The chosen model is **XGBoost with `scale_pos_weight`** (class balancing):

```python
xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
```

**Why XGBoost over Logistic Regression?**
Both models produce similar metrics on this dataset, as noted by the DS. XGBoost was selected because:
- It handles non-linear feature interactions natively.
- It is more robust to feature scaling differences.
- It is an industry standard for tabular classification problems.
- The test client comment (`# when("xgboost.XGBClassifier")...`) confirms it as the expected model.

**Top 10 features used:**
| Feature | Description |
|---|---|
| `OPERA_Latin American Wings` | Airline |
| `MES_7` | July |
| `MES_10` | October |
| `OPERA_Grupo LATAM` | Airline |
| `MES_12` | December |
| `TIPOVUELO_I` | International flight |
| `MES_4` | April |
| `MES_11` | November |
| `OPERA_Sky Airline` | Airline |
| `OPERA_Copa Air` | Airline |

### Model Persistence
The trained model is serialized to disk using `joblib` (`challenge/trained_model.pkl`) so that the API can load it at startup without requiring a training step at runtime.

---

## Part II: API

The API was implemented using **FastAPI** with the following design decisions:

- **Input validation** is performed before calling the model. Invalid requests return HTTP `400`:
  - `MES` must be between 1 and 12.
  - `TIPOVUELO` must be `"I"` (international) or `"N"` (national).
  - `OPERA` must be one of the 23 airlines present in the training dataset.
- The model is instantiated once at module load time to avoid reloading the `.pkl` on every request.
- Pydantic models (`Flight`, `PredictRequest`) enforce the request schema.

**Endpoint:**
```
POST /predict
{
  "flights": [
    { "OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 3 }
  ]
}
```
**Response:**
```json
{ "predict": [0] }
```

---

## Part III: Cloud Deployment

The API was containerized with Docker and deployed to **GCP Cloud Run**.

- **Base image:** `python:3.9-slim`
- The model is trained during the Docker build step so the container starts instantly with a ready-to-serve model.
- Cloud Run was configured with `--min-instances 0` (scale to zero) to minimize costs.
- The image is built for `linux/amd64` to ensure compatibility with Cloud Run.

**API URL:** `https://latam-challenge-api-503036182178.us-central1.run.app`

**Stress test results (100 users, 60 seconds):**
| Metric | Value |
|---|---|
| Total requests | 5,245 |
| Failure rate | 0.00% |
| Avg response time | 336 ms |
| Max response time | 789 ms |

---

## Part IV: CI/CD

Two GitHub Actions workflows were implemented:

### Continuous Integration (`ci.yml`)
- Triggered on every push to `main`/`develop` and on pull requests to `main`.
- Installs dependencies and runs both model tests and API tests.

### Continuous Delivery (`cd.yml`)
- Triggered on every push to `main`.
- Authenticates to GCP using a service account stored as a GitHub Secret.
- Builds the Docker image for `linux/amd64` and pushes it to Artifact Registry.
- Deploys the new revision to Cloud Run automatically.

**Secrets required:**
- `GCP_PROJECT_ID`: GCP project identifier.
- `GCP_SA_KEY`: Service account JSON key with `roles/run.admin`, `roles/artifactregistry.writer`, and `roles/iam.serviceAccountUser`.
