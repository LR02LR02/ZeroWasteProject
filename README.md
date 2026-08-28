# ZWaste

ZWaste is a research prototype for household food-waste mitigation. It integrates multi-object food localisation, ordinal food-condition classification, visual explainability, smart inventory management, and inventory-constrained recipe recommendation within a Streamlit web application.

## Main Features

- **Multi-object food localisation:** YOLOE is used as a pilot localisation and segmentation layer for images containing multiple supported food items.
- **Food and condition prediction:** MobileNetV3-Small with CORN predicts the food category and ordered condition stage.
- **Condition stages:** Fresh, Medium / Use Soon, and Rotten / Do Not Use.
- **Visual explanation:** Grad-CAM provides spatial attribution showing image regions that contributed to the predicted condition stage.
- **Smart inventory:** Detected items can be added individually or together, while manually added items are recorded as **Not Assessed**.
- **Recipe retrieval:** Candidate recipes from the cleaned Allrecipes corpus are retrieved using TF-IDF and cosine-similarity ranking.
- **Neuro-symbolic control:** Deterministic inventory, structure, feasibility, priority-food, and post-generation checks constrain recipe adaptation.
- **LLM recipe adaptation:** Selected recipe candidates are adapted using a cloud-hosted Large Language Model under strict inventory and structured-output requirements.
- **Rate & Review:** A 5-point Likert interface records understandability, trustworthiness, usefulness, visual-consistency ratings, and optional comments.

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

The Streamlit application entry point is:

```text
app/app.py
```

## Supported Camera Food Classes

The trained food-condition classifier supports 11 fruit and vegetable categories:

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

Unsupported foods can still be added manually to **My Kitchen** as **Not Assessed**. Manually added foods may participate in recipe matching, but no computer-vision condition assessment is assigned to them.

## Deployment

GitHub stores the project source code and required project artifacts, but GitHub itself does not run the Streamlit application.

To deploy the application, connect this repository to **Streamlit Community Cloud** and use:

```text
app/app.py
```

as the Streamlit entry point.

Keep `requirements.txt` at the repository root.

## Streamlit Secrets

Do not place API keys or access tokens directly in the repository.

Configure the following values in the Streamlit Community Cloud **Secrets** settings:

```toml
OPENAI_API_KEY = "your_openai_api_key"

GITHUB_REVIEW_TOKEN = "your_github_personal_access_token"
GITHUB_REVIEW_REPO = "YOUR_GITHUB_USERNAME/ZeroWasteProject"
GITHUB_REVIEW_BRANCH = "main"
GITHUB_REVIEW_FILE = "reviews/reviews.csv"
```

`GITHUB_REVIEW_TOKEN` should be a fine-grained GitHub Personal Access Token restricted to this repository with:

```text
Contents: Read and write
```

permission.

The token is used only by the application to read and update the configured review CSV through the GitHub Contents API.

Never commit real secrets to:

```text
app.py
README.md
.streamlit/secrets.toml
```

or any other tracked repository file.

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

The review file uses the following header:

```csv
response_id,timestamp_utc,understandability,trustworthiness,usefulness,visual_consistency,comment
```

The Rate & Review interface supports future human-centred evaluation of the prototype. Its presence in the application does not by itself constitute participant-based validation.

## Runtime Notes

- YOLOE uses `yoloe-26s-seg.pt` for multi-object localisation and segmentation.
- YOLOE text-prompt initialisation may additionally require the MobileCLIP text encoder `mobileclip2_b.ts`. The deployment environment must allow this asset to be downloaded or otherwise make it available at runtime.
- YOLOE is treated as an exploratory multi-object localisation and segmentation pilot rather than a formally benchmarked detection model within this study.
- MobileNetV3-Small + CORN remains the final food-category and condition-stage classifier.
- Grad-CAM is generated from the condition-stage prediction to provide spatial attribution.
- Recipe recommendation uses preprocessed Allrecipes data, a stored TF-IDF vectoriser, and a stored TF-IDF matrix.
- LLM-based recipe adaptation requires a valid `OPENAI_API_KEY`.
- The application is software-based and does not require physical smart-fridge hardware.

## Interpretation Notes

- The classifier predicts three discrete but naturally ordered condition stages: **Fresh, Medium, and Rotten**.
- Model confidence is **not** a biological freshness percentage.
- Grad-CAM provides **spatial attribution**, presented in the interface as model attention. It does not independently identify semantic defects such as mould, bruising, browning, wrinkling, or surface discolouration.
- **Fresh** items remain usable with normal priority.
- **Medium / Use Soon** items remain usable and receive higher recipe priority.
- **Rotten / Do Not Use** items are considered unusable and are excluded from recipe generation.
- Manually added items are available for recipe matching but their condition is recorded as **Not Assessed**.
- Recipe matching and validation are primarily presence-based and do not verify exact ingredient quantities.
- Recipe recommendations are constrained by the available inventory and predefined pantry allowances rather than being generated freely from unrestricted ingredients.

## Local Run

The application can also be run locally.

Install the required packages:

```bash
pip install -r requirements.txt
```

Start the Streamlit application:

```bash
streamlit run app/app.py
```

For local execution, configure the GitHub review settings in:

```text
.streamlit/secrets.toml
```

For example:

```toml
OPENAI_API_KEY = "your_openai_api_key"

GITHUB_REVIEW_TOKEN = "your_github_personal_access_token"
GITHUB_REVIEW_REPO = "YOUR_GITHUB_USERNAME/ZeroWasteProject"
GITHUB_REVIEW_BRANCH = "main"
GITHUB_REVIEW_FILE = "reviews/reviews.csv"
```

The `OPENAI_API_KEY` may alternatively be provided through the `OPENAI_API_KEY` environment variable.

## Research Context

This prototype was developed as part of research on a multimodal AI framework integrating edge-optimised computer vision, ordinal learning, explainable AI, deterministic inventory reasoning, and constraint-based NLP for household food-waste mitigation.

The framework is intended as a software-based research proof of concept. Its current evaluation demonstrates technical feasibility within the study setting and should not be interpreted as evidence of physical edge-device deployment, long-term household food-waste reduction, or participant-validated improvements in user trust and understanding.
