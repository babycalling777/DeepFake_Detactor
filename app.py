import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
import timm
import tempfile
import os

# --- UI Setup ---
st.set_page_config(page_title="Deepfake Detector Pro AI", page_icon="🛡️")
st.title("🛡️ Pro Deepfake Detector (CPU Mode)")
st.info("यह AI CPU पर चलेगा और वीडियो को स्कैन करेगा।")

# --- Model & Tools Loading ---
@st.cache_resource
def load_tools():
    device = torch.device('cpu')

    mtcnn = MTCNN(
        margin=20,
        keep_all=False,
        post_process=False,
        device=device,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7]
    )

    model = timm.create_model('legacy_xception', pretrained=False, num_classes=2).to(device)

    # ✅ LOCAL MODEL (REPO SE LOAD)
    model_path = "deepfake_detector_smart_v2.pth"

    if os.path.exists(model_path):
        # 🔥 FINAL FIX (IMPORTANT)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        model.eval()
    else:
        st.error("❌ Model file nahi mili! Repo me .pth file upload karo")

    return mtcnn, model, device


# Load tools
try:
    mtcnn, model, device = load_tools()
    st.sidebar.success("Model Loaded on CPU ✔")
except Exception as e:
    st.error(f"Error loading AI tools: {e}")

# --- Video Upload Section ---
file = st.file_uploader("Video File Chunein...", type=["mp4", "avi", "mov"])

if file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.close()

    col1, col2 = st.columns(2)

    with col1:
        st.video(file)

    if st.button('🚀 Start Full Video Scan'):
        with st.spinner('AI वीडियो के 20 फ्रेम्स को स्कैन कर रहा है...'):

            cap = cv2.VideoCapture(tfile.name)
            v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = np.linspace(0, v_len - 1, 20, dtype=int)
            all_probs = []

            transform = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])

            for idx in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()

                if not ret:
                    continue

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_rgb = np.array(img_rgb).astype(np.uint8)

                try:
                    face = mtcnn(img_rgb)

                    if face is not None:
                        face_img = transforms.ToPILImage()(face.byte())
                        img_t = transform(face_img).unsqueeze(0).to(device)

                        with torch.no_grad():
                            out = model(img_t)
                            prob = torch.softmax(out, dim=1).cpu().numpy()
                            all_probs.append(prob)

                except Exception:
                    continue

            cap.release()

            if all_probs:
                res = np.mean(all_probs, axis=0)
                label = np.argmax(res)
                conf = np.max(res) * 100

                final_res = "REAL" if label == 1 else "FAKE"

                with col2:
                    if conf < 60:
                        st.warning(f"AI is Uncertain ({conf:.2f}%)")
                    elif final_res == "FAKE":
                        st.error(f"🚨 RESULT: {final_res}")
                    else:
                        st.success(f"✅ RESULT: {final_res}")

                    st.metric("Confidence", f"{conf:.2f}%")
            else:
                st.warning("कोई चेहरा नहीं मिला!")

    try:
        os.unlink(tfile.name)
    except:
        pass
