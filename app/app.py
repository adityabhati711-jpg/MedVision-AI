
from pathlib import Path
from html import escape

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import plotly.graph_objects as go

from PIL import Image
from torchvision import transforms
from torchvision.models import densenet121

# -------------------- CONFIG --------------------

st.set_page_config(
    page_title="MedVision AI | Research Dashboard",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "Models" / "densenet121_v2_original_backup.pth"

CLASS_NAMES = [
    "Lung Opacity",
    "No Lung Opacity / Not Normal",
    "Normal",
]

# Verified DenseNet121 V2 held-out test metrics
TEST_ACCURACY = 73.15
TEST_MACRO_F1 = 0.7189
TEST_IMAGES = 4003

TEST_CONFUSION_MATRIX = [
    [508, 371, 23],
    [236, 1314, 223],
    [1, 221, 1106],
]

# -------------------- STYLING: SINGLE WHITE + WINE THEME --------------------
# Self-contained CSS: no extra theme configuration file required.
# Applying one light-theme block before the UI prevents the old blue CSS
# from rendering first and then being overridden by a second CSS block.
st.markdown("""
<style>
:root { color-scheme: light; }

html, body, .stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: #FFFFFF !important;
    color: #191919 !important;
}

[data-testid="stHeader"] {
    background: #FFFFFF !important;
}

.block-container,
[data-testid="stMainBlockContainer"] {
    padding-top: 2.5rem !important;
    max-width: 1450px;
}

[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] h4 {
    color: #621B36 !important;
}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] span {
    color: #202020;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebarContent"] {
    background: #F8F2F5 !important;
    border-right: 1px solid #E5D5DE !important;
}

[data-testid="stSidebar"] *,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #222222 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: #EBD8E1 !important;
    border-radius: 8px;
}

[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #772443 !important;
}

/* Banner and analytics cards */
.hero {
    background: #FFFFFF !important;
    border: 1px solid #E5D5DE !important;
    border-radius: 18px !important;
    padding: 28px;
    margin: 0 0 20px;
    overflow: visible !important;
    box-shadow: none !important;
}

.hero-title {
    font-size: 42px;
    font-weight: 800;
    color: #621B36 !important;
}

.hero-subtitle {
    color: #464646 !important;
    font-size: 16px;
    margin-top: 8px;
}

.info-card,
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E5D5DE !important;
    border-radius: 14px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: none !important;
}

.info-label {
    color: #464646 !important;
    font-size: 13px;
}

.info-value {
    color: #621B36 !important;
    font-size: 30px;
    font-weight: 750;
}

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #FCF8FA !important;
    border: 1px dashed #B68A9E !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #252525 !important;
}

/* Custom notices are used instead of native st.info / st.warning / st.error.
   This avoids Streamlit dark-mode notification backgrounds overriding text contrast. */
.medvision-notice {
    display: flex !important;
    align-items: flex-start !important;
    gap: 12px !important;
    width: 100% !important;
    box-sizing: border-box !important;
    padding: 16px 18px !important;
    margin: 12px 0 20px !important;
    background-color: #F9F3F6 !important;
    border: 1px solid #E4D1DA !important;
    border-left: 4px solid #772443 !important;
    border-radius: 10px !important;
    color: #252025 !important;
    font-size: 0.95rem !important;
    line-height: 1.55 !important;
}

.medvision-notice,
.medvision-notice * {
    color: #252025 !important;
}

.medvision-notice .medvision-notice-icon {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    flex: 0 0 20px !important;
    width: 20px !important;
    height: 20px !important;
    margin-top: 1px !important;
    border: 1.5px solid #772443 !important;
    border-radius: 50% !important;
    color: #772443 !important;
    font-size: 12px !important;
    font-weight: 800 !important;
    line-height: 1 !important;
}

.medvision-notice .medvision-notice-text {
    flex: 1 1 auto !important;
    color: #252025 !important;
    overflow-wrap: anywhere;
}

.medvision-notice--error {
    background-color: #FFF5F5 !important;
    border-color: #E8BFC5 !important;
    border-left-color: #A52D49 !important;
}

.medvision-notice--error .medvision-notice-icon {
    border-color: #A52D49 !important;
    color: #A52D49 !important;
}

/* Buttons */
div.stButton > button,
[data-testid="stAppViewContainer"] button[kind="primary"] {
    background: #772443 !important;
    color: #FFFFFF !important;
    border: 1px solid #772443 !important;
    border-radius: 10px;
    font-weight: 700;
}

div.stButton > button:hover,
[data-testid="stAppViewContainer"] button[kind="primary"]:hover {
    background: #621B36 !important;
    color: #FFFFFF !important;
}

[data-testid="stAppViewContainer"] button[kind="primary"] * {
    color: #FFFFFF !important;
}

[data-testid="stAppViewContainer"] button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #621B36 !important;
    border-color: #BEA3AF !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- THEME-SAFE STATUS MESSAGES --------------------

def notice(kind, message):
    """Render readable status messages in both Streamlit light and dark modes.

    Use escaped text and an app-owned element instead of native notification
    surfaces, whose dark-theme background may clash with the fixed white UI.
    """
    kind = kind if kind in {"info", "warning", "error"} else "info"
    symbol = "i" if kind == "info" else "!"
    role = "status" if kind == "info" else "alert"
    safe_message = escape(str(message))

    st.markdown(
        f'<div class="medvision-notice medvision-notice--{kind}" role="{role}">'
        f'<span class="medvision-notice-icon" aria-hidden="true">{symbol}</span>'
        f'<span class="medvision-notice-text">{safe_message}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# -------------------- MODEL --------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"DenseNet121 V2 checkpoint missing: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    class_to_idx = checkpoint["class_to_idx"]
    class_names = [
        name for name, index in sorted(
            class_to_idx.items(), key=lambda item: item[1]
        )
    ]
    if class_names != CLASS_NAMES:
        raise ValueError(f"Unexpected checkpoint classes: {class_names}")

    model = densenet121(weights=None)
    model.classifier = torch.nn.Linear(
        model.classifier.in_features, len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # Match the exact preprocessing stored in the trained V2 checkpoint.
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=checkpoint["mean"],
            std=checkpoint["std"],
        ),
    ])
    return model, class_names, preprocess


# -------------------- PREDICTION CHART --------------------

def score_chart(names, scores):
    fig = go.Figure(
        go.Bar(
            x=[score * 100 for score in scores],
            y=names,
            orientation="h",
            marker_color=[
                "#27d4e8",
                "#347dba",
                "#5d7392",
            ],
            text=[
                f"{score * 100:.2f}%"
                for score in scores
            ],
            textposition="auto",
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#202020"),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, 100],
            title="Model score (%)",
        ),
        yaxis=dict(
            autorange="reversed"
        ),
        margin=dict(
            l=15,
            r=15,
            t=20,
            b=35,
        ),
        height=300,
        showlegend=False,
    )

    return fig


# -------------------- GRAD-CAM --------------------

def generate_gradcam(model, tensor, class_index):
    """
    Generate Grad-CAM for the predicted class.

    This explains model attribution.
    It does not identify confirmed medical lesions.
    """

    activations = []
    gradients = []

    def forward_hook(module, inputs, output):
        activations.append(output.detach())

        output.register_hook(
            lambda grad: gradients.append(grad.detach())
        )

    handle = model.features.norm5.register_forward_hook(
        forward_hook
    )

    try:
        model.zero_grad(set_to_none=True)

        # Grad-CAM requires gradients.
        with torch.enable_grad():
            scores = model(tensor)
            scores[0, class_index].backward()

        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = F.relu(
            (weights * activation).sum(
                dim=1,
                keepdim=True,
            )
        )

        cam = F.interpolate(
            cam,
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (
                cam_max - cam_min
            )
        else:
            cam = np.zeros_like(cam)

        return cam

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def gradcam_figure(image, cam, opacity=0.45):
    """
    Overlay the Grad-CAM heatmap on the uploaded X-ray.
    """

    display_image = image.resize((224, 224))

    fig = go.Figure()

    fig.add_trace(
        go.Image(
            z=np.asarray(display_image)
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=cam,
            colorscale="Jet",
            zmin=0,
            zmax=1,
            opacity=opacity,
            showscale=True,
            colorbar=dict(
                title="Attribution",
                thickness=12,
            ),
            hovertemplate=(
                "Relative attribution: %{z:.2f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#202020"),
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        xaxis=dict(
            visible=False,
            range=[-0.5, 223.5],
            constrain="domain",
        ),
        yaxis=dict(
            visible=False,
            range=[223.5, -0.5],
            scaleanchor="x",
        ),
    )

    return fig


# -------------------- NAVIGATION --------------------

with st.sidebar:
    st.title("🩻 MedVision AI")
    st.caption("Deep Learning Research Platform")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "AI Scanner",
            "Model Analytics",
            "Research & Limitations",
            "Future Improvements",
        ],
    )

    st.divider()

    st.caption("Classifier: DenseNet121 — V2")
    st.caption("Inference device: CPU")
    st.caption("Research use only")


# -------------------- HEADER --------------------

st.markdown("""
<div class="hero">
    <div class="hero-title">MEDVISION AI</div>
    <div class="hero-subtitle">
        Chest X-ray Classification • Computer Vision •
        Explainable AI Research
    </div>
</div>
""", unsafe_allow_html=True)

notice("warning",
    "Research and educational demonstration only. "
    "Not validated for medical diagnosis or clinical decisions."
)


# -------------------- AI SCANNER --------------------

if page == "AI Scanner":

    st.header("AI X-ray Scanner")

    st.caption(
        "Upload a chest X-ray for three-class prediction and Grad-CAM attribution. "
        "This app is designed for chest X-rays only; other images may produce misleading results."
    )

    uploaded_file = st.file_uploader(
        "Upload X-ray image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:

        try:
            image = Image.open(
                uploaded_file
            ).convert("RGB")

            # Clear old results when the uploaded image changes.
            uploaded_bytes = uploaded_file.getvalue()

            import hashlib

            image_hash = hashlib.sha256(
                uploaded_bytes
            ).hexdigest()

            if st.session_state.get(
                "last_image_hash"
            ) != image_hash:

                st.session_state[
                    "last_image_hash"
                ] = image_hash

                st.session_state.pop(
                    "analysis_result",
                    None,
                )

            left, right = st.columns(
                [1, 1.3],
                gap="large",
            )

            with left:
                st.subheader("Uploaded Image")

                st.image(
                    image,
                    width="stretch",
                )

            with right:
                st.subheader("AI Analysis")

                analyze = st.button(
                    "Analyze X-ray",
                    type="primary",
                    width="stretch",
                )

                if analyze:
                    with st.spinner("Running DenseNet121 V2 inference and Grad-CAM..."):
                        model, names, preprocess = load_model()
                        tensor = preprocess(image).unsqueeze(0)
                        with torch.inference_mode():
                            outputs = model(tensor)
                            probabilities = torch.softmax(outputs, dim=1)[0].tolist()

                        predicted_index = int(np.argmax(probabilities))
                        cam = generate_gradcam(model, tensor, predicted_index)
                        st.session_state["analysis_result"] = {
                            "names": names,
                            "scores": probabilities,
                            "predicted_index": predicted_index,
                            "cam": cam,
                        }

                result = st.session_state.get("analysis_result")

                if result is not None:
                    st.markdown("#### Predicted class")
                    notice("info", result["names"][result["predicted_index"]])
                    st.plotly_chart(
                        score_chart(result["names"], result["scores"]),
                        width="stretch",
                    )
                    st.caption(
                        "These are uncalibrated model scores, "
                        "not probabilities of disease."
                    )

            # ---------------- GRAD-CAM DISPLAY ----------------

            result = st.session_state.get(
                "analysis_result"
            )

            if result is not None:

                st.divider()

                st.header(
                    "Explainable AI | Grad-CAM"
                )

                st.write(
                    "Grad-CAM highlights image regions "
                    "that contributed to the model's "
                    "predicted class."
                )

                opacity = st.slider(
                    "Heatmap overlay intensity",
                    min_value=0.10,
                    max_value=0.90,
                    value=0.45,
                    step=0.05,
                )

                col1, col2 = st.columns(
                    2,
                    gap="large",
                )

                with col1:
                    st.subheader(
                        "Original X-ray"
                    )

                    st.image(
                        image,
                        width="stretch",
                    )

                with col2:
                    st.subheader(
                        "Grad-CAM Heatmap"
                    )

                    fig = gradcam_figure(
                        image,
                        result["cam"],
                        opacity,
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                    )

                notice("info",
                    "Red and yellow areas indicate higher "
                    "relative model attribution. "
                    "They do not confirm disease or "
                    "identify medically verified lesions."
                )

        except Exception as error:

            notice("error",
                f"Analysis failed: {error}"
            )

    else:

        notice("info",
            "Upload a chest X-ray to begin analysis."
        )


# -------------------- ANALYTICS --------------------

elif page == "Model Analytics":

    st.header("Model Performance")

    c1, c2, c3 = st.columns(3)

    for column, label, value in [
        (
            c1,
            "Test Accuracy",
            f"{TEST_ACCURACY:.2f}%",
        ),
        (
            c2,
            "Test Macro F1",
            f"{TEST_MACRO_F1:.4f}",
        ),
        (
            c3,
            "Test Images",
            f"{TEST_IMAGES:,}",
        ),
    ]:

        with column:

            st.markdown(
                f'<div class="info-card">'
                f'<div class="info-label">{label}</div>'
                f'<div class="info-value">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.subheader(
        "Held-out Test Confusion Matrix"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=TEST_CONFUSION_MATRIX,
            x=CLASS_NAMES,
            y=CLASS_NAMES,
            colorscale="Blues",
            text=TEST_CONFUSION_MATRIX,
            texttemplate="%{text}",
            showscale=True,
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#202020"),
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Predicted label",
        yaxis_title="Actual label",
        yaxis=dict(
            autorange="reversed"
        ),
        height=460,
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

    st.caption(
        "Metrics are recorded from the DenseNet121 V2 "
        "held-out test evaluation, not calculated "
        "from website uploads."
    )


    # ---------------- EXPLAINABILITY RESEARCH ----------------
    # These are recorded results from a separate 100-image,
    # validation-only Lung Opacity experiment (not test-set scores).
    st.divider()
    st.header("Historical ResNet18 Explainability Research | Grad-CAM")
    st.caption(
        "Historical ResNet18 validation-only experiment: 100 Lung Opacity-labeled X-rays. "
        "For each image, Grad-CAM for the Lung Opacity class was "
        "compared with radiologist bounding boxes and 100 random "
        "box placements."
    )

    research_c1, research_c2, research_c3, research_c4 = st.columns(4)

    with research_c1:
        st.metric("Validation X-rays", "100")

    with research_c2:
        st.metric("Radiologist-box energy", "18.78%")

    with research_c3:
        st.metric("Random-box energy", "12.56%")

    with research_c4:
        st.metric("Mean difference", "+6.22 pp")

    st.subheader("Annotation Alignment vs Random Baseline")

    validation_graph = (
        ROOT / "reports" / "gradcam_random_baseline_validation_100.png"
    )

    if validation_graph.is_file():
        st.image(
            str(validation_graph),
            caption="100 validation images: recorded Grad-CAM energy comparison",
            width="stretch",
        )
    else:
        # Keep the dashboard usable even if the saved PNG is not deployed.
        comparison_fig = go.Figure(
            go.Bar(
                x=["Radiologist boxes", "Random boxes"],
                y=[18.78, 12.56],
                marker_color=["#27d4e8", "#8290a5"],
                text=["18.78%", "12.56%"],
                textposition="outside",
            )
        )
        comparison_fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#202020"),
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Mean Grad-CAM energy in boxes (%)",
            yaxis=dict(range=[0, 24]),
            height=400,
            showlegend=False,
            margin=dict(l=25, r=20, t=35, b=30),
        )
        st.plotly_chart(comparison_fig, width="stretch")
        st.caption(
            "Saved PNG unavailable. Showing the recorded experiment "
            "summary as an interactive chart instead."
        )

    st.subheader("Statistical Findings")

    stats_c1, stats_c2, stats_c3 = st.columns(3)

    with stats_c1:
        st.metric("Better than random", "80 / 100")

    with stats_c2:
        st.metric("95% CI for mean difference", "+4.90 to +7.54 pp")

    with stats_c3:
        st.metric("Paired-difference t-test p-value", "3.31e-15")

    st.caption(
        "The confidence interval and two-sided t-test apply to the "
        "per-image difference between radiologist-box and mean "
        "random-box Grad-CAM energy. Results were recorded from the "
        "100-image validation sample, not website uploads."
    )

    notice("info",
        "These historical Grad-CAM statistics were measured on ResNet18, not DenseNet121 V2. "
        "Research limitation: Grad-CAM is a coarse model-attribution "
        "method, not a lesion detector. Randomly positioned boxes are "
        "not anatomically matched controls. Positive alignment does "
        "not establish diagnostic accuracy, reliable localization, "
        "or clinical validity."
    )


# -------------------- RESEARCH --------------------

elif page == "Research & Limitations":

    st.header(
        "Research & Limitations"
    )

    st.subheader(
        "Project Overview"
    )

    st.write(
        "MedVision AI explores deep learning "
        "for three-class chest X-ray image "
        "classification using a DenseNet121 V2 model."
    )

    st.subheader(
        "Dataset Labels"
    )

    st.write(
        "1. Lung Opacity\n"
        "2. No Lung Opacity / Not Normal\n"
        "3. Normal"
    )

    st.subheader(
        "Evaluation"
    )

    st.write(
        "The model was evaluated on 4,003 "
        "held-out test images. "
        "Patient identifiers were audited "
        "to check that training, validation, "
        "and test splits did not overlap."
    )

    st.subheader(
        "Explainable AI"
    )

    st.write(
        "Grad-CAM uses gradients from the "
        "final DenseNet121 feature layer "
        "to visualize relative model attribution "
        "for a selected prediction."
    )

    st.subheader(
        "Important Limitations"
    )

    st.write(
        "Only chest X-ray images are supported. The app does not automatically "
        "verify the image type; uploading other images can produce misleading classifications."
    )

    st.write(
        "Lung opacity is not equivalent to "
        "a confirmed pneumonia diagnosis."
    )

    st.write(
        "Grad-CAM visualizations represent "
        "model attribution rather than "
        "verified medical lesions."
    )

    st.write(
        "The model makes classification errors, "
        "may perform differently on external "
        "datasets, and has not been "
        "clinically validated."
    )


# -------------------- FUTURE IMPROVEMENTS --------------------

elif page == "Future Improvements":

    st.header("Future Improvements")
    st.caption(
        "Research roadmap | Implemented features and remaining improvements"
    )

    notice("info",
        "Current release: DenseNet121 V2 three-class chest X-ray classifier. "
        "Held-out test accuracy: 73.15% on 4,003 images; macro F1: 0.7189."
    )

    st.subheader("01 | Improve Classification Performance")
    st.write(
        "**Current limitation:** DenseNet121 V2 achieves 73.15% "
        "accuracy and 0.7189 macro F1 on the held-out three-class test set; "
        "Lung Opacity recall is 56%."
    )
    st.write(
        "**Progress:** DenseNet121 V2, weighted V3 and V4 experiments "
        "were completed. Future work may compare EfficientNet and "
        "additional augmentation, calibration, and training settings."
    )
    st.write(
        "**Evaluation plan:** Tune models on patient-separated training "
        "and validation data, then confirm performance on new independent "
        "evaluation data. Report accuracy, macro F1, class-wise recall, "
        "and confusion matrices. An 85-90% accuracy goal is aspirational, "
        "not a promised result."
    )

    st.divider()

    st.subheader("02 | Chest X-ray Input Validation Layer")
    st.write(
        "**Status: Planned, not implemented.** A future version could include "
        "a dedicated image-validation layer to check whether an upload is "
        "a chest X-ray before sending it to DenseNet121."
    )
    st.write(
        "**Why it matters:** The current classifier is designed for chest X-rays "
        "and can return misleading predictions for photographs, documents, "
        "or other medical imaging types."
    )
    st.write(
        "**Evaluation plan:** Train and independently test a robust image-type "
        "classifier using diverse chest X-rays and non-X-ray images. Measure "
        "false acceptance and false rejection rates before considering deployment."
    )

    st.divider()
    st.subheader("Roadmap Status")
    st.write("**Model architecture and training experiments:** DenseNet121 V2/V3/V4 completed; more planned")
    st.write("**Independent performance validation:** V2 held-out test completed; external validation planned")
    st.write("**Chest X-ray input validation layer:** Planned; not available in the current app")

    notice("warning",
        "MedVision AI is an educational research project, not a clinically "
        "validated diagnostic system. These future improvements would "
        "not make it suitable for clinical decisions without further validation."
    )


# -------------------- FOOTER --------------------

st.divider()

st.caption(
    "MedVision AI • DenseNet121 V2 • "
    "Educational Research Demonstration"
)
