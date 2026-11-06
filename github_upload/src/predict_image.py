
"""
Single-image road surface prediction using the trained ResNet18 + SVM model.

Run from the project root:
.venv\Scripts\python.exe src\predict_image.py "path\to\road_image.jpg"
"""

from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

# Import the existing handcrafted feature extractor.
from features import extract_all


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cnn_feature_classifier.pkl"

IMAGE_SIZE = (224, 224)
CROP_TOP_FRACTION = 0.50


def load_classifier():
    """Load the saved classifier and its metadata."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    artifact = joblib.load(MODEL_PATH)

    required_keys = {
        "classifier_pipeline",
        "feature_view",
        "handcrafted_feature_columns",
        "class_order",
    }
    missing = required_keys - set(artifact.keys())

    if missing:
        raise ValueError(
            f"The saved model is missing required information: {missing}"
        )

    return artifact


def load_cnn():
    """Load pretrained ResNet18 as a 512-feature extractor."""
    weights = models.ResNet18_Weights.DEFAULT

    network = models.resnet18(weights=weights)
    network.fc = torch.nn.Identity()
    network.eval()
    network.to(torch.device("cpu"))

    # Match the normalization used in the training experiment.
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=weights.transforms().mean,
            std=weights.transforms().std,
        ),
    ])

    return network, image_transform


def extract_cnn_features(image_path, network, image_transform):
    """Generate the 512-dimensional ResNet18 embedding."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Could not open the image: {image_path}"
        )

    height, width = image.shape[:2]

    # Keep the lower half, matching the training pipeline.
    image = image[int(height * CROP_TOP_FRACTION):, :]

    if image.size == 0:
        raise ValueError("The road image is empty after cropping.")

    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    )

    # OpenCV loads BGR; ResNet18 expects RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    tensor = image_transform(image).unsqueeze(0)

    with torch.inference_mode():
        features = network(tensor)

    return features.cpu().numpy().astype(np.float32)


def predict_image(image_path):
    """Predict one image using the saved model."""
    image_path = Path(image_path).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    artifact = load_classifier()

    feature_view = artifact["feature_view"]
    class_order = artifact["class_order"]
    classifier = artifact["classifier_pipeline"]

    print("Loading ResNet18 feature extractor...")
    network, image_transform = load_cnn()

    print("Extracting image features...")
    cnn_features = extract_cnn_features(
        image_path,
        network,
        image_transform,
    )

    if feature_view == "cnn_plus_handcrafted":
        handcrafted = extract_all(image_path)

        if handcrafted is None:
            raise RuntimeError(
                "Handcrafted feature extraction failed."
            )

        columns = artifact["handcrafted_feature_columns"]

        missing = [
            name for name in columns
            if name not in handcrafted
        ]

        if missing:
            raise ValueError(
                f"Handcrafted features are missing: {missing}"
            )

        handcrafted_features = np.array(
            [[handcrafted[name] for name in columns]],
            dtype=np.float32,
        )

        combined_features = np.hstack([
            cnn_features,
            handcrafted_features,
        ])

    elif feature_view == "cnn":
        combined_features = cnn_features

    else:
        raise ValueError(
            f"Unsupported saved feature configuration: {feature_view}"
        )

    expected_features = getattr(
        classifier, "n_features_in_", None
    )

    if (
        expected_features is not None
        and combined_features.shape[1] != expected_features
    ):
        raise ValueError(
            "Feature count mismatch. "
            f"Model expects {expected_features}, "
            f"but prediction generated {combined_features.shape[1]}."
        )

    prediction = classifier.predict(combined_features)[0]

    print("\n" + "=" * 42)
    print("ROAD SURFACE CLASSIFICATION RESULT")
    print("=" * 42)
    print(f"Image: {image_path.name}")
    print(f"Predicted class: {prediction}")
    print("=" * 42)


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print(
            '.venv\\Scripts\\python.exe '
            'src\\predict_image.py "path\\to\\road_image.jpg"'
        )
        raise SystemExit(1)

    try:
        predict_image(sys.argv[1])
    except Exception as exc:
        print(f"\nPrediction failed: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
