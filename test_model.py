import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# -----------------------------
# Paths and class labels
# -----------------------------
model_path = "lung_disease_resnet_finetuned.keras"  # your trained model path
test_folder = "sample_images"                        # folder containing test CT images
class_labels = ["NORMAL", "PNEUMONIA", "ADENOCARCINOMA", "LARGE_CELL_CARCINOMA", "SQUAMOUS_CELL_CARCINOMA"]

# -----------------------------
# Load trained model
# -----------------------------
model = load_model(model_path)
print("✅ Model loaded successfully!")

# -----------------------------
# Grad-CAM helper functions
# -----------------------------
def get_gradcam_heatmap(model, img_array, last_conv_layer_name="conv5_block3_out"):
    """Generate Grad-CAM heatmap from the last convolutional layer."""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_path, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on original image."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

# -----------------------------
# Predict each test image
# -----------------------------
for file_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, file_name)
    if not os.path.isfile(img_path):
        continue

    # Preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array_exp = np.expand_dims(img_array, axis=0)

    # Predict
    pred_probs = model.predict(img_array_exp)
    pred_class = np.argmax(pred_probs)
    confidence = np.max(pred_probs) * 100

    print(f"\n🩻 Image: {file_name}")
    print(f"Predicted Disease: {class_labels[pred_class]} ({confidence:.2f}%)")

    # Grad-CAM Visualization
    heatmap = get_gradcam_heatmap(model, img_array_exp)
    overlay_img = overlay_heatmap(img_path, heatmap)

    # Display
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title(f"{class_labels[pred_class]} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()
