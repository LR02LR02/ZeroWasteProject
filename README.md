# ZWaste

ZWaste is a research prototype for household food-waste mitigation. It combines multi-object food localisation, ordinal food-condition classification, visual explainability, smart inventory management, and inventory-constrained recipe recommendation in a Streamlit web application.

## Main Features

- **Multi-object food localisation:** YOLOE is used as a pilot localisation/segmentation layer for images containing multiple food items.
- **Food and condition prediction:** MobileNetV3-Small with CORN predicts the food item and ordered condition stage.
- **Condition stages:** Fresh, Medium / Use Soon, and Rotten / Do Not Use.
- **Visual explanation:** Grad-CAM highlights image regions that contributed to the model prediction.
- **Smart inventory:** Detected items can be added individually or together, while manually added items are shown as **Not Assessed**.
- **Recipe retrieval:** Allrecipes recipes are retrieved using TF-IDF and cosine similarity.
- **Neuro-symbolic control:** Inventory, structure, feasibility, priority-food, and post-generation checks constrain recipe generation.
- **Rate & Review:** A 5-point Likert interface collects understandability, trustworthiness, usefulness, visual-consistency ratings, and optional comments.

## Project Structure

```text
ZeroWasteProject/
├── app/
│   ├── app.py
│   └── utils.py
├── fruitveg_processed/
│   └── class_mappings.json
├── models/
│   └── mobilenetv3_corn.pt
├── recipes/
│   ├── allrecipes_clean.csv.gz
│   ├── recipe_tfidf_matrix.npz
│   └── tfidf_vectorizer.joblib
├── reviews/
│   └── reviews.csv
├── requirements.txt
└── README.md
```

## Application Entry Point

```text
app/app.py
```

## Supported Camera Food Classes

The trained food-condition classifier supports 11 produce classes:

- Apple
- Banana
- Brinjal
- Chillies
- Cucumber
- Guava
- Lemon
- Orange
- Pepper
- Potato
- Tomato

Unsupported foods can still be added manually to **My Kitchen** as **Not Assessed**.

## Deployment

GitHub stores the project source code and artifacts, but GitHub by itself does **not** run the Streamlit application. `README.md` is documentation only.

To run the deployed web app, connect this GitHub repository to **Streamlit Community Cloud** and use:

```text
app/app.py
```

as the Streamlit entry point.

Keep `requirements.txt` at the repository root.

## Streamlit Secrets

Do not place API keys or access tokens directly in the repository.

Add the following values in the Streamlit Community Cloud **Secrets** settings:

```toml
OPENAI_API_KEY = "your_openai_api_key"

GITHUB_REVIEW_TOKEN = "your_github_personal_access_token"
GITHUB_REVIEW_REPO = "YOUR_GITHUB_USERNAME/ZeroWasteProject"
GITHUB_REVIEW_BRANCH = "main"
GITHUB_REVIEW_FILE = "reviews/reviews.csv"
```

`GITHUB_REVIEW_TOKEN` should have only the repository permission required to update `reviews/reviews.csv`.

Never commit real secrets to `app.py`, `README.md`, or `.streamlit/secrets.toml`.

## Review Storage

The **Rate & Review** form records:

- Understandability
- Trustworthiness
- Usefulness
- Visual consistency
- Optional written feedback

Submitted reviews are appended to:

```text
reviews/reviews.csv
```

The review file should keep this header:

```csv
response_id,timestamp_utc,understandability,trustworthiness,usefulness,visual_consistency,comment
```

## Runtime Notes

- YOLOE uses the standard `yoloe-26s-seg.pt` weights at runtime.
- The YOLOE component is treated as a multi-object localisation/segmentation pilot.
- MobileNetV3-Small + CORN remains the final food-condition classifier.
- The application does not require physical smart-fridge hardware.

## Interpretation Notes

- Model confidence is **not** a biological freshness percentage.
- Grad-CAM shows **model attention** and does not automatically identify mould, bruising, browning, or another physical defect.
- Rotten / Do Not Use items are excluded from recipe generation.
- Medium / Use Soon items receive higher recipe priority.
- Manually added items are available for recipe matching but their condition is **Not Assessed**.
- Recipe reasoning is based mainly on ingredient availability rather than exact quantity estimation.

## Local Run

If required, the app can also be run locally:

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Configure the same secrets locally using `.streamlit/secrets.toml` or suitable environment variables.

## Research Context

This prototype was developed for research on a multimodal AI framework integrating edge-optimised computer vision, ordinal regression, explainable AI, and constraint-based NLP for household food-waste mitigation.
