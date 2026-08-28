from pathlib import Path
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from coral_pytorch.dataset import corn_label_from_logits

from openai import OpenAI


IMAGE_SIZE = 224
UTILS_VERSION = "2026-08-28-v5-deployment-and-recipe-validation"

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

STAGE_RULES = {
    "fresh": {
        "urgency": "Normal",
        "usable": True,
        "priority": 1
    },
    "medium": {
        "urgency": "High - Use Soon",
        "usable": True,
        "priority": 2
    },
    "rotten": {
        "urgency": "Do Not Use",
        "usable": False,
        "priority": 0
    }
}


MULTI_OBJECT_CLASSES = [
    "apple",
    "banana",
    "eggplant",
    "chili pepper",
    "cucumber",
    "guava",
    "lemon",
    "orange",
    "bell pepper",
    "potato",
    "tomato"
]

DETECTOR_TO_ZWASTE = {
    "apple": "Apple",
    "banana": "Banana",
    "eggplant": "Brinjal",
    "chili pepper": "Chillies",
    "cucumber": "Cucumber",
    "guava": "Guava",
    "lemon": "Lemon",
    "orange": "Orange",
    "bell pepper": "Pepper",
    "potato": "Potato",
    "tomato": "Tomato"
}

DEFAULT_DETECTION_CONF = 0.06
DEFAULT_DETECTOR_WEIGHTS = "yoloe-26s-seg.pt"

MOBILECLIP_FILENAME = "mobileclip2_b.ts"
MOBILECLIP_URL = (
    "https://github.com/ultralytics/assets/releases/download/"
    "v8.4.0/mobileclip2_b.ts"
)

PANTRY_STAPLES = {
    "water",
    "salt",
    "black pepper",
    "ground black pepper",
    "sugar",
    "olive oil",
    "vegetable oil",
    "cooking oil"
}

DERIVED_PRODUCT_WORDS = {
    # A basic inventory food must not automatically match a
    # processed/derived product that merely contains its name.
    #
    # Examples:
    # Chicken != chicken broth
    # Chicken != cream of chicken soup
    # Tomato  != tomato sauce
    # Apple   != apple juice
    #
    # If the user explicitly adds the full derived product
    # (for example "Chicken Broth"), it can still match.
    "juice", "vinegar", "cider", "sauce", "paste",
    "powder", "spice", "seasoning", "syrup",
    "jam", "jelly", "butter", "extract",
    "concentrate", "puree", "flour", "starch",
    "chip", "chips", "noodle", "noodles",
    "soup", "broth", "stock", "bouillon",
    "consomme", "gravy", "dressing", "ketchup",
    "sherbet", "sorbet", "gelatin", "jell", "jello",
    "marmalade"
}

MEASUREMENT_WORDS = {
    "cup", "cups", "teaspoon", "teaspoons",
    "tablespoon", "tablespoons", "ounce", "ounces",
    "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "kg", "kilogram", "kilograms",
    "ml", "milliliter", "milliliters", "liter", "liters",
    "quart", "quarts", "pint", "pints",
    "gallon", "gallons", "package", "packages",
    "packet", "packets", "can", "cans", "jar", "jars",
    "bottle", "bottles", "bunch", "bunches",
    "clove", "cloves", "slice", "slices",
    "piece", "pieces", "box"
}

PREPARATION_WORDS = {
    "large", "small", "medium", "divided", "minced",
    "sliced", "diced", "chopped", "ground", "freshly",
    "fresh", "prepared", "cut", "strips", "halves",
    "half", "cubes", "cube", "optional", "taste",
    "finely", "thinly", "roughly", "peeled", "seeded",
    "shredded", "grated", "melted", "softened",
    "drained", "rinsed", "lightly", "beaten"
}

RETRIEVAL_STOP_WORDS = (
    MEASUREMENT_WORDS
    | PREPARATION_WORDS
    | {"to", "and", "or", "of", "for", "as", "needed"}
)

MATCH_REMOVE_WORDS = {
    "cup", "cups", "teaspoon", "teaspoons",
    "tablespoon", "tablespoons", "ounce", "ounces",
    "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "kg", "ml", "liter", "liters",
    "package", "packages", "can", "cans",
    "clove", "cloves", "slice", "slices",
    "large", "small", "medium", "fresh", "chopped",
    "diced", "sliced", "minced", "ground", "peeled",
    "seeded", "shredded", "grated", "drained",
    "rinsed", "optional", "finely", "thinly",
    "to", "and", "or", "of", "for", "as", "needed"
}

FRACTIONS = "¼½¾⅓⅔⅛⅜⅝⅞"


class MobileNetV3_CORN(nn.Module):
    def __init__(self, num_items, num_stages):
        super().__init__()

        backbone = mobilenet_v3_small(weights=None)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        feature_dim = backbone.classifier[0].in_features

        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.Hardswish(),
            nn.Dropout(0.2)
        )

        self.item_head = nn.Linear(512, num_items)
        self.stage_head = nn.Linear(512, num_stages - 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared(x)

        return self.item_head(x), self.stage_head(x)


def corn_stage_probabilities(stage_logits):
    conditional = torch.sigmoid(stage_logits.float())
    cumulative = torch.cumprod(conditional, dim=1)

    return torch.cat([
        1 - cumulative[:, :1],
        cumulative[:, :-1] - cumulative[:, 1:],
        cumulative[:, -1:]
    ], dim=1)


def load_cv_model(model_path, mapping_path):
    with open(mapping_path, "r", encoding="utf-8") as file:
        mappings = json.load(file)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MobileNetV3_CORN(
        mappings["num_items"],
        mappings["num_stages"]
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    return model, mappings, device


def predict_with_gradcam(image, model, mappings, device):
    original = image.convert("RGB")

    image_tensor = (
        EVAL_TRANSFORM(original)
        .unsqueeze(0)
        .to(device)
    )

    stored = {}

    def save_activation(module, inputs, output):
        stored["activation"] = output

        output.register_hook(
            lambda gradient: stored.update(
                {"gradient": gradient}
            )
        )

    handle = model.features[-1].register_forward_hook(
        save_activation
    )

    try:
        model.zero_grad(set_to_none=True)

        item_logits, stage_logits = model(image_tensor)

        item_probs = torch.softmax(
            item_logits,
            dim=1
        )

        stage_probs = corn_stage_probabilities(
            stage_logits
        )

        item_id = int(
            item_probs.argmax(dim=1).item()
        )

        stage_id = int(
            corn_label_from_logits(
                stage_logits.float()
            ).item()
        )

        target = stage_probs[0, stage_id]
        target.backward()

    finally:
        handle.remove()

    activation = stored["activation"].detach()
    gradient = stored["gradient"].detach()

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    cam = (
        weights * activation
    ).sum(dim=1)

    cam = F.relu(cam)[0]

    cam = (
        (cam - cam.min())
        / (cam.max() - cam.min() + 1e-8)
    )

    cam = F.interpolate(
        cam[None, None],
        size=(original.height, original.width),
        mode="bilinear",
        align_corners=False
    )[0, 0].cpu().numpy()

    original_array = (
        np.asarray(original)
        .astype(np.float32)
        / 255.0
    )

    heatmap = plt.get_cmap()(cam)[..., :3]

    overlay = np.clip(
        0.60 * original_array
        + 0.40 * heatmap,
        0,
        1
    )

    id_to_item = {
        int(k): v
        for k, v in mappings["id_to_item"].items()
    }

    id_to_stage = {
        int(k): v
        for k, v in mappings["id_to_stage"].items()
    }

    item = id_to_item[item_id].title()
    stage = id_to_stage[stage_id].title()
    rule = STAGE_RULES[stage.lower()]

    record = {
        "item": item,
        "stage": stage,
        "urgency": rule["urgency"],
        "usable": rule["usable"],
        "priority": rule["priority"],
        "source": "computer_vision",
        "item_probability": round(
            float(item_probs[0, item_id].item()),
            4
        ),
        "stage_probability": round(
            float(stage_probs[0, stage_id].item()),
            4
        )
    }

    return record, overlay


def ensure_mobileclip_weights():
    """Ensure the YOLOE text encoder is available at runtime.

    YOLOE text prompting loads ``mobileclip2_b.ts`` by filename. On
    ephemeral deployments such as Streamlit Community Cloud, the lazy
    Ultralytics download can occasionally fail before the file exists.
    This helper downloads the official asset explicitly when needed.
    """
    file_path = Path.cwd() / MOBILECLIP_FILENAME

    if file_path.exists() and file_path.stat().st_size > 0:
        return file_path

    temporary_path = file_path.with_suffix(
        file_path.suffix + ".part"
    )

    try:
        response = requests.get(
            MOBILECLIP_URL,
            stream=True,
            timeout=300
        )
        response.raise_for_status()

        with open(temporary_path, "wb") as output_file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    output_file.write(chunk)

        if (
            not temporary_path.exists()
            or temporary_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "The MobileCLIP download completed without "
                "creating a usable file."
            )

        temporary_path.replace(file_path)

    except Exception as error:
        try:
            if temporary_path.exists():
                temporary_path.unlink()
        except Exception:
            pass

        raise RuntimeError(
            "YOLOE requires mobileclip2_b.ts for text-prompt "
            "initialisation, but the file could not be prepared. "
            f"Details: {error}"
        ) from error

    return file_path


def load_object_detector(
    weights=DEFAULT_DETECTOR_WEIGHTS,
    detection_classes=None
):
    from ultralytics import YOLOE

    # YOLOE-26 text prompting requires this encoder when set_classes()
    # is called. Keeping it in the current working directory matches
    # the filename-based lookup used by the runtime.
    ensure_mobileclip_weights()

    detector = YOLOE(weights)

    detector.set_classes(
        detection_classes
        or MULTI_OBJECT_CLASSES
    )

    return detector


def isolate_detected_food(
    image,
    box,
    polygon=None
):
    original = image.convert("RGB")

    x1, y1, x2, y2 = (
        box.xyxy[0]
        .detach()
        .cpu()
        .numpy()
        .astype(int)
    )

    x1 = max(0, min(x1, original.width - 1))
    y1 = max(0, min(y1, original.height - 1))
    x2 = max(x1 + 1, min(x2, original.width))
    y2 = max(y1 + 1, min(y2, original.height))

    mask = Image.new(
        "L",
        original.size,
        0
    )

    draw = ImageDraw.Draw(mask)

    if (
        polygon is not None
        and len(polygon) >= 3
    ):
        polygon_points = [
            tuple(map(float, point))
            for point in polygon
        ]

        draw.polygon(
            polygon_points,
            fill=255
        )

    else:
        draw.rectangle(
            (x1, y1, x2, y2),
            fill=255
        )

    food_crop = original.crop(
        (x1, y1, x2, y2)
    )

    mask_crop = mask.crop(
        (x1, y1, x2, y2)
    )

    background = Image.new(
        "RGB",
        food_crop.size,
        "white"
    )

    isolated_crop = Image.composite(
        food_crop,
        background,
        mask_crop
    )

    return isolated_crop


def predict_multiple_foods(
    image,
    detector,
    model,
    mappings,
    device,
    conf=DEFAULT_DETECTION_CONF
):
    original = image.convert("RGB")

    result = detector.predict(
        source=original,
        conf=conf,
        verbose=False
    )[0]

    if result.boxes is None:
        return [], result

    boxes = list(result.boxes)

    if len(boxes) == 0:
        return [], result

    polygons = (
        result.masks.xy
        if result.masks is not None
        else [None] * len(boxes)
    )

    detections = []

    for index, box in enumerate(boxes):
        polygon = (
            polygons[index]
            if index < len(polygons)
            else None
        )

        isolated_crop = isolate_detected_food(
            original,
            box,
            polygon
        )

        record, gradcam_overlay = (
            predict_with_gradcam(
                isolated_crop,
                model,
                mappings,
                device
            )
        )

        detector_class_id = int(
            box.cls.item()
        )

        detector_label = result.names[
            detector_class_id
        ]

        detector_food = (
            DETECTOR_TO_ZWASTE.get(
                detector_label
            )
        )

        detection = {
            **record,
            "detection_index": index + 1,
            "detector_label": detector_label,
            "detector_food": detector_food,
            "detection_confidence": round(
                float(box.conf.item()),
                4
            ),
            "agreement": (
                detector_food
                == record["item"]
            ),
            "crop": isolated_crop,
            "gradcam_overlay": gradcam_overlay
        }

        detections.append(
            detection
        )

    return detections, result


def inventory_record_from_detection(
    detection
):
    return {
        "item": detection["item"],
        "stage": detection["stage"],
        "urgency": detection["urgency"],
        "usable": detection["usable"],
        "priority": detection["priority"],
        "source": "computer_vision",
        "item_probability": detection[
            "item_probability"
        ],
        "stage_probability": detection[
            "stage_probability"
        ],
        "image": detection.get(
            "crop"
        )
    }


def deduplicate_inventory_records(
    detections
):
    """
    Keep one record per food + degradation stage.

    Examples:
    - Apple Fresh + Apple Fresh -> one Apple Fresh record
    - Orange Fresh + Orange Rotten -> keep both records

    When duplicate food-stage predictions exist, retain the
    prediction with the strongest confidence.
    """
    if not detections:
        return []

    selected = {}

    for detection in detections:
        item = str(
            detection["item"]
        ).strip()

        stage = str(
            detection["stage"]
        ).strip()

        group_key = (
            item.lower(),
            stage.lower()
        )

        candidate_key = (
            float(
                detection.get(
                    "stage_probability",
                    0.0
                )
                or 0.0
            ),
            float(
                detection.get(
                    "item_probability",
                    0.0
                )
                or 0.0
            )
        )

        current = selected.get(
            group_key
        )

        if (
            current is None
            or candidate_key > current[0]
        ):
            selected[group_key] = (
                candidate_key,
                detection
            )

    records = [
        inventory_record_from_detection(
            value[1]
        )
        for value in selected.values()
    ]

    stage_order = {
        "medium": 0,
        "fresh": 1,
        "rotten": 2,
        "not assessed": 3
    }

    records.sort(
        key=lambda record: (
            stage_order.get(
                str(
                    record["stage"]
                ).strip().lower(),
                4
            ),
            str(
                record["item"]
            ).lower()
        )
    )

    return records


def manual_record(item):
    return {
        "item": item.strip().title(),
        "stage": "Not Assessed",
        "urgency": "Not Assessed",
        "usable": True,
        "priority": 1,
        "source": "manual",
        "item_probability": None,
        "stage_probability": None
    }


def build_recipe_inventory(records):
    """
    Build a presence-based recipe inventory.

    If the same food exists in more than one condition,
    a usable instance keeps that ingredient available for
    recipes. A rotten instance does not block a separate
    fresh/medium instance of the same food.
    """
    if not records:
        return {
            "priority_ingredients": [],
            "available_ingredients": [],
            "excluded_items": []
        }

    grouped = {}

    for record in records:
        item = str(
            record["item"]
        ).strip()

        key = item.lower()

        grouped.setdefault(
            key,
            {
                "item": item,
                "has_usable": False,
                "has_priority": False,
                "has_unusable": False
            }
        )

        if bool(
            record.get(
                "usable",
                False
            )
        ):
            grouped[key][
                "has_usable"
            ] = True

            if int(
                record.get(
                    "priority",
                    0
                )
                or 0
            ) == 2:
                grouped[key][
                    "has_priority"
                ] = True

        else:
            grouped[key][
                "has_unusable"
            ] = True

    priority_ingredients = sorted(
        [
            data["item"]
            for data in grouped.values()
            if (
                data["has_usable"]
                and data["has_priority"]
            )
        ],
        key=str.lower
    )

    available_ingredients = sorted(
        [
            data["item"]
            for data in grouped.values()
            if data["has_usable"]
        ],
        key=lambda item: (
            item.lower()
            not in {
                priority.lower()
                for priority
                in priority_ingredients
            },
            item.lower()
        )
    )

    excluded_items = sorted(
        [
            data["item"]
            for data in grouped.values()
            if (
                data["has_unusable"]
                and not data["has_usable"]
            )
        ],
        key=str.lower
    )

    return {
        "priority_ingredients": (
            priority_ingredients
        ),
        "available_ingredients": (
            available_ingredients
        ),
        "excluded_items": (
            excluded_items
        )
    }


def singularize(word):
    if len(word) <= 3:
        return word

    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("oes"):
        return word[:-2]

    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]

    if word.endswith("s") and not word.endswith(
        ("ss", "us", "is")
    ):
        return word[:-1]

    return word


def normalize_ingredient_text(text):
    cleaned_phrases = []

    for ingredient in str(text).split(";"):
        ingredient = ingredient.lower()
        ingredient = re.sub(
            r"\([^)]*\)",
            " ",
            ingredient
        )
        ingredient = re.sub(
            fr"[{FRACTIONS}]",
            " ",
            ingredient
        )
        ingredient = re.sub(
            r"\d+(?:[./]\d+)?",
            " ",
            ingredient
        )
        ingredient = re.sub(
            r"[^a-z\s-]",
            " ",
            ingredient
        )

        words = []

        for word in ingredient.replace("-", " ").split():
            if word in RETRIEVAL_STOP_WORDS:
                continue

            word = singularize(word)

            if (
                word
                and word not in RETRIEVAL_STOP_WORDS
            ):
                words.append(word)

        if words:
            cleaned_phrases.append(
                " ".join(words)
            )

    return " ".join(cleaned_phrases)


def load_recipe_resources(
    recipes_path,
    vectorizer_path,
    matrix_path
):
    recipes = pd.read_csv(recipes_path)
    vectorizer = joblib.load(vectorizer_path)
    recipe_matrix = load_npz(matrix_path)

    if len(recipes) != recipe_matrix.shape[0]:
        raise ValueError(
            "Recipe table and TF-IDF matrix have different row counts."
        )

    return recipes, vectorizer, recipe_matrix


def retrieve_recipes(
    inventory_data,
    recipes,
    vectorizer,
    recipe_matrix,
    top_n=5
):
    excluded = {
        item.strip().lower()
        for item in inventory_data.get(
            "excluded_items",
            []
        )
    }

    available = [
        item
        for item in inventory_data.get(
            "available_ingredients",
            []
        )
        if item.strip().lower() not in excluded
    ]

    priority = [
        item
        for item in inventory_data.get(
            "priority_ingredients",
            []
        )
        if item.strip().lower() not in excluded
    ]

    if not available:
        raise ValueError(
            "No usable ingredients are available for retrieval."
        )

    query_items = available + priority

    query_text = normalize_ingredient_text(
        " ; ".join(query_items)
    )

    query_vector = vectorizer.transform(
        [query_text]
    )

    scores = cosine_similarity(
        query_vector,
        recipe_matrix
    ).ravel()

    ranked_indices = np.argsort(
        scores
    )[::-1]

    ranked_indices = [
        index
        for index in ranked_indices
        if scores[index] > 0
    ][:top_n]

    if not ranked_indices:
        raise ValueError(
            "No matching recipes were found."
        )

    columns = [
        "name",
        "category",
        "rating",
        "ingredients",
        "directions",
        "prep",
        "cook",
        "total"
    ]

    result = recipes.iloc[
        ranked_indices
    ][columns].copy()

    result.insert(
        1,
        "similarity_score",
        scores[ranked_indices]
    )

    return result.reset_index(drop=True)


def normalize_phrase(text):
    text = str(text).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\d+(?:[./]\d+)?", " ", text)
    text = re.sub(r"[^a-z\s-]", " ", text)

    words = []

    for word in text.replace("-", " ").split():
        if word in MATCH_REMOVE_WORDS:
            continue

        word = singularize(word)

        if word and word not in MATCH_REMOVE_WORDS:
            words.append(word)

    return " ".join(words)


def ingredient_matches_item(ingredient, item):
    """Conservatively match a recipe ingredient to an inventory item.

    Matching rules:
    1. Every normalized token in the inventory item must appear in
       the recipe ingredient.
    2. If the recipe ingredient contains an additional derived-product
       word such as "soup", "broth", "juice", or "sauce", the match is
       rejected unless that derived-product word is also explicitly
       part of the inventory item.

    This prevents false matches such as:
        Chicken -> chicken broth
        Chicken -> cream of chicken soup
        Apple   -> apple juice
        Tomato  -> tomato sauce

    while still allowing:
        Chicken -> chicken breast
        Chicken -> chicken wings
        Apple   -> diced apples
        Chicken Broth -> chicken broth
    """
    ingredient_tokens = set(
        normalize_phrase(ingredient).split()
    )

    item_tokens = set(
        normalize_phrase(item).split()
    )

    if not ingredient_tokens or not item_tokens:
        return False

    if not item_tokens.issubset(ingredient_tokens):
        return False

    ingredient_derived = (
        ingredient_tokens
        & DERIVED_PRODUCT_WORDS
    )

    item_derived = (
        item_tokens
        & DERIVED_PRODUCT_WORDS
    )

    extra_derived_words = (
        ingredient_derived
        - item_derived
    )

    if extra_derived_words:
        return False

    return True


def is_pantry(ingredient):
    return any(
        ingredient_matches_item(
            ingredient,
            staple
        )
        for staple in PANTRY_STAPLES
    )


def evaluate_recipe(row, inventory_data):
    available = inventory_data.get(
        "available_ingredients",
        []
    )

    priority = inventory_data.get(
        "priority_ingredients",
        []
    )

    excluded = inventory_data.get(
        "excluded_items",
        []
    )

    ingredient_list = [
        item.strip()
        for item in str(row["ingredients"]).split(";")
        if item.strip()
    ]

    matched = []
    pantry = []
    unavailable = []
    excluded_found = []

    for ingredient in ingredient_list:
        if any(
            ingredient_matches_item(
                ingredient,
                item
            )
            for item in excluded
        ):
            excluded_found.append(ingredient)

        elif any(
            ingredient_matches_item(
                ingredient,
                item
            )
            for item in available
        ):
            matched.append(ingredient)

        elif is_pantry(ingredient):
            pantry.append(ingredient)

        else:
            unavailable.append(ingredient)

    required_total = (
        len(matched)
        + len(unavailable)
        + len(excluded_found)
    )

    coverage = (
        len(matched) / required_total
        if required_total > 0
        else 0.0
    )

    priority_used = any(
        ingredient_matches_item(
            ingredient,
            item
        )
        for ingredient in matched
        for item in priority
    )

    structure_valid = all(
        pd.notna(row[field])
        and str(row[field]).strip()
        for field in [
            "name",
            "ingredients",
            "directions"
        ]
    )

    if not structure_valid or excluded_found:
        decision = "REJECT"
    elif required_total > 0 and not unavailable:
        decision = "ACCEPT"
    elif matched:
        decision = "ADAPT"
    else:
        decision = "REJECT"

    return {
        "inventory_coverage": round(coverage, 4),
        "priority_used": priority_used,
        "matched_inventory": " | ".join(matched),
        "pantry_ingredients": " | ".join(pantry),
        "unavailable_ingredients": " | ".join(unavailable),
        "excluded_ingredients": " | ".join(excluded_found),
        "structure_valid": structure_valid,
        "decision": decision
    }


def apply_constraints(candidates, inventory_data):
    evaluations = candidates.apply(
        lambda row: evaluate_recipe(
            row,
            inventory_data
        ),
        axis=1,
        result_type="expand"
    )

    constrained = pd.concat(
        [candidates, evaluations],
        axis=1
    )

    decision_rank = {
        "ACCEPT": 2,
        "ADAPT": 1,
        "REJECT": 0
    }

    constrained["_decision_rank"] = (
        constrained["decision"]
        .map(decision_rank)
    )

    return (
        constrained
        .sort_values(
            by=[
                "_decision_rank",
                "priority_used",
                "inventory_coverage",
                "similarity_score"
            ],
            ascending=[
                False,
                False,
                False,
                False
            ]
        )
        .drop(columns="_decision_rank")
        .reset_index(drop=True)
    )


def adapt_recipe(
    candidate,
    inventory_data,
    api_key,
    model_name="gpt-5.6-luna"
):
    excluded = {
        item.strip().lower()
        for item in inventory_data.get(
            "excluded_items",
            []
        )
    }

    available = [
        item
        for item in inventory_data.get(
            "available_ingredients",
            []
        )
        if item.strip().lower() not in excluded
    ]

    priority = [
        item
        for item in inventory_data.get(
            "priority_ingredients",
            []
        )
        if item.strip().lower() not in excluded
    ]

    allowed_ingredients = (
        available
        + sorted(PANTRY_STAPLES)
    )

    instructions = """
You are the recipe-adaptation module of a food-waste reduction system.

Follow these rules strictly:
1. Use only ingredients listed under ALLOWED INGREDIENTS.
2. Never use ingredients listed under EXCLUDED ITEMS.
3. Candidate-recipe ingredients marked unavailable are not allowed unless they
   also appear explicitly under ALLOWED INGREDIENTS.
4. Use priority ingredients whenever they are available and culinarily feasible.
5. Treat the retrieved recipe only as a reference; adapt it to the bounded inventory.
6. Do not invent additional foods, seasonings, garnishes, or optional ingredients.
7. Produce simple, realistic cooking steps.
8. If no reasonable recipe can be made from the allowed ingredients, return
   status NO_VALID_ADAPTATION with empty title, ingredients, and steps.
"""

    prompt_data = {
        "allowed_ingredients": allowed_ingredients,
        "priority_ingredients": priority,
        "excluded_items": inventory_data.get(
            "excluded_items",
            []
        ),
        "source_candidate": {
            "name": candidate["name"],
            "ingredients": candidate["ingredients"],
            "directions": candidate["directions"],
            "unavailable_ingredients": candidate[
                "unavailable_ingredients"
            ]
        }
    }

    recipe_schema = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": [
                    "ADAPTED",
                    "NO_VALID_ADAPTATION"
                ]
            },
            "source_candidate": {
                "type": "string"
            },
            "title": {
                "type": "string"
            },
            "ingredients": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "priority_ingredients_used": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "adaptation_summary": {
                "type": "string"
            }
        },
        "required": [
            "status",
            "source_candidate",
            "title",
            "ingredients",
            "steps",
            "priority_ingredients_used",
            "adaptation_summary"
        ],
        "additionalProperties": False
    }

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model_name,
        instructions=instructions,
        input=json.dumps(
            prompt_data,
            ensure_ascii=False,
            indent=2
        ),
        text={
            "format": {
                "type": "json_schema",
                "name": "adapted_recipe",
                "schema": recipe_schema,
                "strict": True
            }
        }
    )

    return json.loads(
        response.output_text
    )


def validate_recipe(recipe_data, inventory_data):
    required_fields = {
        "status",
        "source_candidate",
        "title",
        "ingredients",
        "steps",
        "priority_ingredients_used",
        "adaptation_summary"
    }

    missing_fields = sorted(
        required_fields - set(recipe_data)
    )

    status = recipe_data.get("status")

    if status == "NO_VALID_ADAPTATION":
        structure_valid = (
            not missing_fields
            and recipe_data.get("title", "") == ""
            and recipe_data.get("ingredients", []) == []
            and recipe_data.get("steps", []) == []
        )

        return {
            "status": status,
            "structure_valid": structure_valid,
            "allowed_ingredients_valid": True,
            "excluded_ingredients_valid": True,
            "priority_compliant": True,
            "priority_field_consistent": True,
            "hallucinated_ingredients": [],
            "excluded_ingredients_found": [],
            "missing_priority_ingredients": [],
            "hallucination_rate": 0.0,
            "overall_validation": (
                "PASS" if structure_valid else "FAIL"
            ),
            "usable_recipe": False
        }

    if status != "ADAPTED":
        return {
            "status": status,
            "structure_valid": False,
            "allowed_ingredients_valid": False,
            "excluded_ingredients_valid": False,
            "priority_compliant": False,
            "priority_field_consistent": False,
            "hallucinated_ingredients": [],
            "excluded_ingredients_found": [],
            "missing_priority_ingredients": [],
            "hallucination_rate": 0.0,
            "overall_validation": "FAIL",
            "usable_recipe": False
        }

    ingredients = recipe_data.get(
        "ingredients",
        []
    )

    steps = recipe_data.get(
        "steps",
        []
    )

    structure_valid = (
        not missing_fields
        and bool(
            str(
                recipe_data.get(
                    "title",
                    ""
                )
            ).strip()
        )
        and isinstance(ingredients, list)
        and len(ingredients) > 0
        and isinstance(steps, list)
        and len(steps) > 0
        and all(
            str(step).strip()
            for step in steps
        )
    )

    excluded = inventory_data.get(
        "excluded_items",
        []
    )

    excluded_lower = {
        item.strip().lower()
        for item in excluded
    }

    available = [
        item
        for item in inventory_data.get(
            "available_ingredients",
            []
        )
        if item.strip().lower() not in excluded_lower
    ]

    priority = [
        item
        for item in inventory_data.get(
            "priority_ingredients",
            []
        )
        if item.strip().lower() not in excluded_lower
    ]

    allowed = (
        available
        + list(PANTRY_STAPLES)
    )

    hallucinated = []
    excluded_found = []

    for ingredient in ingredients:
        if any(
            ingredient_matches_item(
                ingredient,
                item
            )
            for item in excluded
        ):
            excluded_found.append(ingredient)

        elif not any(
            ingredient_matches_item(
                ingredient,
                item
            )
            for item in allowed
        ):
            hallucinated.append(ingredient)

    actual_priority_used = [
        item
        for item in priority
        if any(
            ingredient_matches_item(
                ingredient,
                item
            )
            for ingredient in ingredients
        )
    ]

    missing_priority = [
        item
        for item in priority
        if item not in actual_priority_used
    ]

    declared_priority = recipe_data.get(
        "priority_ingredients_used",
        []
    )

    normalized_actual = {
        normalize_phrase(item)
        for item in actual_priority_used
    }

    normalized_declared = {
        normalize_phrase(item)
        for item in declared_priority
    }

    priority_field_consistent = (
        normalized_actual
        == normalized_declared
    )

    hallucination_rate = (
        len(hallucinated)
        / len(ingredients)
        if ingredients
        else 0.0
    )

    allowed_valid = (
        len(hallucinated) == 0
    )

    excluded_valid = (
        len(excluded_found) == 0
    )

    priority_compliant = (
        len(missing_priority) == 0
    )

    overall_pass = all([
        structure_valid,
        allowed_valid,
        excluded_valid,
        priority_compliant,
        priority_field_consistent
    ])

    return {
        "status": status,
        "structure_valid": structure_valid,
        "allowed_ingredients_valid": allowed_valid,
        "excluded_ingredients_valid": excluded_valid,
        "priority_compliant": priority_compliant,
        "priority_field_consistent": priority_field_consistent,
        "hallucinated_ingredients": hallucinated,
        "excluded_ingredients_found": excluded_found,
        "missing_priority_ingredients": missing_priority,
        "hallucination_rate": round(
            hallucination_rate,
            4
        ),
        "overall_validation": (
            "PASS" if overall_pass else "FAIL"
        ),
        "usable_recipe": overall_pass
    }


def generate_validated_recipe(
    inventory_data,
    constrained_candidates,
    api_key,
    model_name="gpt-5.6-luna",
    max_attempts=3
):
    usable = constrained_candidates[
        constrained_candidates["decision"] != "REJECT"
    ]

    if usable.empty:
        return None, None, []

    attempts = []
    last_adapted = None
    last_report = None

    for _, row in usable.head(
        max_attempts
    ).iterrows():
        candidate = row.to_dict()

        adapted = adapt_recipe(
            candidate,
            inventory_data,
            api_key,
            model_name=model_name
        )

        report = validate_recipe(
            adapted,
            inventory_data
        )

        last_adapted = adapted
        last_report = report

        attempts.append({
            "candidate": candidate["name"],
            "status": adapted.get("status"),
            "validation": report[
                "overall_validation"
            ],
            "usable_recipe": report[
                "usable_recipe"
            ],
            "structure_valid": report.get(
                "structure_valid",
                False
            ),
            "allowed_ingredients_valid": report.get(
                "allowed_ingredients_valid",
                False
            ),
            "excluded_ingredients_valid": report.get(
                "excluded_ingredients_valid",
                False
            ),
            "priority_compliant": report.get(
                "priority_compliant",
                False
            ),
            "priority_field_consistent": report.get(
                "priority_field_consistent",
                False
            ),
            "missing_priority_ingredients": report.get(
                "missing_priority_ingredients",
                []
            ),
            "hallucinated_ingredients": report.get(
                "hallucinated_ingredients",
                []
            ),
            "excluded_ingredients_found": report.get(
                "excluded_ingredients_found",
                []
            )
        })

        if report["usable_recipe"]:
            return adapted, report, attempts

    # Preserve the final failed adaptation and its detailed validation
    # report so the interface can explain WHY the recipe was rejected.
    return last_adapted, last_report, attempts
