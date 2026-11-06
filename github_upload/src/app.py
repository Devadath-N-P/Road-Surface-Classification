
from pathlib import Path
import sys
import tempfile

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from skimage.feature import canny, local_binary_pattern

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from predict_image import (
    load_classifier,
    load_cnn,
    extract_cnn_features,
)
from features import extract_all, preprocess_image


st.set_page_config(
    page_title="Road Surface Classification",
    page_icon="🛣️",
    layout="wide",
)

st.title("Road Surface Condition Classification")
st.write(
    "Digital image processing and feature-based classification "
    "of Smooth Road, Rough Road, and Tiled/Paved Road."
)

st.caption(
    "Processing stages: road-region selection → grayscale "
    "preprocessing → texture and edge analysis → classification."
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_classifier()


@st.cache_resource(show_spinner=False)
def get_cnn():
    return load_cnn()


uploaded_file = st.file_uploader(
    "Upload a road image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload an image to inspect its processing features and prediction.")
    st.stop()


# Display the original input.
image_bytes = uploaded_file.getvalue()

try:
    original_image = Image.open(
        __import__("io").BytesIO(image_bytes)
    ).convert("RGB")
except Exception as exc:
    st.error(f"Unable to open the image: {exc}")
    st.stop()

st.subheader("1. Input Image")
st.image(
    original_image,
    caption=uploaded_file.name,
    use_container_width=True,
)

run_prediction = st.button(
    "Process Image and Classify",
    type="primary",
    use_container_width=True,
)

if run_prediction:
    temporary_path = None

    try:
        with st.spinner(
            "Extracting image features and classifying the road..."
        ):
            suffix = (
                Path(uploaded_file.name).suffix.lower() or ".jpg"
            )

            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temp_file:
                temp_file.write(image_bytes)
                temporary_path = Path(temp_file.name)

            artifact = get_model()
            network, image_transform = get_cnn()

            # Use the same handcrafted feature extractor used in training.
            handcrafted = extract_all(temporary_path)

            if handcrafted is None:
                raise RuntimeError(
                    "Could not extract handcrafted image features."
                )

            # Generate the 512-dimensional ResNet18 representation.
            cnn_features = extract_cnn_features(
                temporary_path,
                network,
                image_transform,
            )

            feature_view = artifact["feature_view"]
            classifier = artifact["classifier_pipeline"]

            if feature_view == "cnn_plus_handcrafted":
                feature_columns = artifact[
                    "handcrafted_feature_columns"
                ]

                handcrafted_vector = np.asarray(
                    [[handcrafted[name] for name in feature_columns]],
                    dtype=np.float32,
                )

                model_features = np.hstack(
                    [cnn_features, handcrafted_vector]
                )

            elif feature_view == "cnn":
                model_features = cnn_features

            else:
                raise ValueError(
                    f"Unsupported feature configuration: {feature_view}"
                )

            expected_features = getattr(
                classifier, "n_features_in_", None
            )

            if (
                expected_features is not None
                and model_features.shape[1] != expected_features
            ):
                raise ValueError(
                    f"The classifier expects {expected_features} features "
                    f"but received {model_features.shape[1]}."
                )

            prediction = classifier.predict(model_features)[0]

            # Recreate the classical image-processing outputs.
            gray_raw, gray_blur = preprocess_image(temporary_path)

            edge_map = canny(
                gray_raw.astype(np.float64),
                sigma=1.0,
            )

            lbp_map = local_binary_pattern(
                gray_blur,
                P=8,
                R=1,
                method="uniform",
            )

            histogram, _ = np.histogram(
                gray_raw,
                bins=16,
                range=(0, 256),
            )
            histogram = histogram.astype(np.float64)
            histogram /= max(histogram.sum(), 1)

            # Keep the original color crop for visual explanation.
            width, height = original_image.size
            road_crop = original_image.crop(
                (0, int(height * 0.50), width, height)
            )

        # Show the predicted result.
        st.divider()
        st.subheader("2. Classification Result")
        st.success(f"Predicted class: {prediction}")

        st.caption(
            "The prediction is based on the trained classifier. "
            "Individual mistakes are possible."
        )

        # Show preprocessing.
        st.divider()
        st.subheader("3. Image Preprocessing")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                road_crop,
                caption="Road region used: lower half of the image",
                use_container_width=True,
            )

        with col2:
            st.image(
                gray_raw,
                clamp=True,
                caption="Grayscale image resized to 128 × 128",
                use_container_width=True,
            )

        st.caption(
            "A mildly blurred grayscale version is also used for "
            "GLCM and LBP texture extraction."
        )

        # Show histogram and edges.
        st.divider()
        st.subheader("4. Histogram and Edge Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Grayscale Histogram — 16 bins**")

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.bar(range(16), histogram)
            ax.set_xlabel("Intensity bin (0 = dark, 15 = bright)")
            ax.set_ylabel("Normalized frequency")
            ax.set_title("Pixel Intensity Distribution")
            ax.set_xticks(range(16))
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Shows how pixel brightness values are distributed "
                "across the road region."
            )

        with col2:
            st.markdown("**Canny Edge Map**")
            st.image(
                edge_map.astype(np.uint8) * 255,
                clamp=True,
                caption="Detected edges in the road region",
                use_container_width=True,
            )

            edge_density = float(edge_map.mean())
            st.metric(
                "Edge density",
                f"{edge_density:.4f}",
            )

            st.caption(
                "Edge density is the fraction of pixels identified "
                "as edges."
            )

        # Show texture processing.
        st.divider()
        st.subheader("5. Texture Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Local Binary Pattern (LBP)**")

            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.imshow(
                lbp_map,
                cmap="nipy_spectral",
                vmin=0,
                vmax=10,
            )
            ax.set_title("Local Texture Pattern Map")
            ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "LBP describes local intensity patterns by comparing "
                "a pixel with its neighbors."
            )

        with col2:
            st.markdown("**GLCM Texture Features**")

            glcm_names = [
                ("glcm_contrast", "Contrast"),
                ("glcm_dissimilarity", "Dissimilarity"),
                ("glcm_homogeneity", "Homogeneity"),
                ("glcm_energy", "Energy"),
                ("glcm_correlation", "Correlation"),
                ("glcm_ASM", "Angular second moment (ASM)"),
            ]

            rows = []
            for key, label in glcm_names:
                if key in handcrafted:
                    rows.append({
                        "Feature": label,
                        "Value": f"{handcrafted[key]:.5f}",
                    })

            if rows:
                st.dataframe(
                    pd.DataFrame(rows),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("GLCM values were not found.")

            st.caption(
                "GLCM measures neighboring-pixel relationships in "
                "four directions, averaged across angles."
            )

        # Brief technical explanation.
        st.divider()
        st.subheader("6. How the Features Support Classification")

        st.write(
            "The handcrafted pipeline extracts a 16-bin grayscale "
            "histogram, 6 GLCM measures, 10 LBP histogram values, "
            "and an edge-density value — 33 features in total. "
            "The selected model combines these features with a "
            "512-dimensional ResNet18 image representation and uses "
            "an RBF-kernel Support Vector Machine (SVM) to predict "
            "the road-surface class."
        )

        st.caption(
            "The visual panels illustrate selected image-processing "
            "results. The classifier uses the complete feature "
            "representation, not just the displayed histogram or edge map."
        )

    except Exception as exc:
        st.error(f"Processing failed: {exc}")

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


st.divider()
st.caption(
    "FDIP Project | Smooth Road • Rough Road • Tiled/Paved Road"
)
