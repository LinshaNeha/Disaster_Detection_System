import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from collections import Counter
import datetime
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import json

# Load model and labels
model = load_model("model/disaster_classifier_model.h5")
labels = ["Drought", "Flood", "Normal", "Wildfire"]



# --- Full Image Prediction ---
def predict_full_image(img):
    img = img.resize((128, 128))
    arr = np.asarray(img) / 255.0
    arr = arr.reshape((1, 128, 128, 3))
    pred = model.predict(arr, verbose=0)
    return labels[np.argmax(pred)], float(np.max(pred))

# --- Heatmap Overlay ---
def overlay_heatmap_on_image(original_img, heatmap_array, alpha=0.5):
    heatmap_normalized = np.uint8(255 * heatmap_array / (np.max(heatmap_array) + 1e-8))
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    original_np = np.array(original_img.resize(heatmap_colored.shape[:2][::-1]))
    overlayed = cv2.addWeighted(original_np, 1 - alpha, heatmap_colored, alpha, 0)
    return Image.fromarray(overlayed)

# --- Patch-wise Prediction ---
def predict_large_image(img, mode="pad", visualize_padding=False, confidence_threshold=0.6):
    patch_size = 128
    width, height = img.size

    if mode == "resize":
        img = img.resize(((width // patch_size) * patch_size, (height // patch_size) * patch_size))
        width, height = img.size
    elif mode == "pad":
        new_width = ((width + patch_size - 1) // patch_size) * patch_size
        new_height = ((height + patch_size - 1) // patch_size) * patch_size
        padded = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded.paste(img, (0, 0))
        if visualize_padding:
            for x in range(width, new_width):
                for y in range(new_height):
                    padded.putpixel((x, y), (128, 128, 128))  #  neutral gray instead of red
            for x in range(new_width):
                for y in range(height, new_height):
                    padded.putpixel((x, y), (128, 128, 128))  #  neutral gray
        img = padded
        width, height = new_width, new_height

    results, y_true, y_pred, confidences = [], [], [], []
    heatmap = np.zeros((height, width))
    table_data = []

    for x in range(0, width, patch_size):
        for y in range(0, height, patch_size):
            patch = img.crop((x, y, x + patch_size, y + patch_size))
            arr = np.asarray(patch) / 255.0
            if arr.shape == (128, 128, 3):
                arr = arr.reshape((1, 128, 128, 3))
                pred = model.predict(arr, verbose=0)
                max_prob = np.max(pred)
                label_idx = np.argmax(pred)
                if max_prob >= confidence_threshold:
                    label = labels[label_idx]
                    results.append(label)
                    confidences.append(max_prob)
                    y_pred.append(label_idx)
                    y_true.append(label_idx)  # mock true = pred for visualization
                    heatmap[y:y+patch_size, x:x+patch_size] = max_prob
                    table_data.append({
                        "Patch (x,y)": f"({x},{y})",
                        "Prediction": label,
                        "Confidence": f"{max_prob:.2f}"
                    })

    df_table = pd.DataFrame(table_data)
    return results, img, confidences, heatmap, df_table, y_true, y_pred

# --- Streamlit UI ---
st.set_page_config(page_title="Disaster Detection System", layout="centered")
st.title(" Disaster Detection System")

uploaded_files = st.file_uploader("Upload image(s)", type=["jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f" Image: {uploaded_file.name}")

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        width, height = image.size

        # Suggest processing based on image size
        if width % 128 == 0 and height % 128 == 0:
            st.info("Image size is already a multiple of 128. Padding or resizing not required.")
        else:
            st.warning(f" Image size ({width}×{height}) is not divisible by 128. Padding or resizing will be applied.")
            if min(width, height) < 128:
                st.error(" One or more image dimensions are smaller than 128px. Consider resizing it to at least 128x128.")

        #  Resize vs Pad suggestion
        if width < 300 and height < 300:
            st.info(" Suggestion: Use 'Resize' since the image is relatively small.")
        else:
            st.info(" Suggestion: Use 'Pad' to preserve structure of large image.")

        prediction_type = st.radio(" Prediction Type", ["Patch-wise (for large images)", "Full-Image Only(128×128 )"], key=uploaded_file.name)

        if prediction_type == "Patch-wise (for large images)":
            mode = st.radio(" Preprocessing", ["Resize", "Pad"], index=1, key="mode"+uploaded_file.name)
            visualize_padding = st.checkbox(" Show padding regions", key="pad"+uploaded_file.name) if mode == "Pad" else False
            threshold = st.slider(" Confidence Threshold", 0.0, 1.0, 0.6, key="conf"+uploaded_file.name)
        else:
            mode = None
            visualize_padding = False
            threshold = 0.6

        if st.button(" Analyze", key="analyze"+uploaded_file.name):
            with st.spinner("Processing image..."):
                if (width == 128 and height == 128) or prediction_type == "Full-Image Only(128×128 )":
                    label, confidence = predict_full_image(image)
                    st.success(f"Prediction: **{label}** with **{confidence*100:.2f}%** confidence")
                else:
                    results, processed_img, confidences, heatmap, df_table, y_true, y_pred = predict_large_image(
                        image, mode.lower(), visualize_padding, threshold)

                    if results:
                        st.image(overlay_heatmap_on_image(processed_img, heatmap), caption=" Heatmap Overlay")
                        st.dataframe(df_table)

                        counts = Counter(results)
                        st.metric("Most Common Disaster", counts.most_common(1)[0][0])
                        st.metric("Total Patches", len(results))
                        st.metric("Average Confidence", f"{np.mean(confidences)*100:.2f}%")

                        st.subheader("Patch-wise Distribution")
                        st.table(counts.most_common())

                        if len(set(y_pred)) > 1:
                            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
                            fig, ax = plt.subplots()
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
                            plt.xlabel("Predicted")
                            plt.ylabel("Actual")
                            st.subheader(" Confusion Matrix")
                            st.pyplot(fig)

                            #  Classification Report
                            st.subheader("📄 Classification Report")
                            present_class_indices = sorted(list(set(y_true) | set(y_pred)))
                            present_class_names = [labels[i] for i in present_class_indices]
                            report = classification_report(
                                y_true, y_pred,
                                labels=present_class_indices,
                                target_names=present_class_names,
                                zero_division=0
                            )
                            st.text(report)

                        # Download summary
                        result_text = "\n".join([f"{k}: {v}" for k, v in counts.items()])
                        result_text += f"\n\nFinal Prediction: {counts.most_common(1)[0][0]}"
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(" Download Result Summary", result_text, file_name=f"summary_{timestamp}.txt")

