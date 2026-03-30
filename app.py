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
import requests

# --- UI Setup ---
st.set_page_config(page_title="Deepfake Detector Pro AI", page_icon="🛡️")
st.title("🛡️ Pro Deepfake Detector (CPU + Online Model)")
st.info("यह AI CPU पर चलेगा और मॉडल Google Drive से डाउनलोड होगा।")

# --- MODEL DOWNLOAD ---
@st.cache_resource
def download_model():
    model_path = "deepfake_detector_smart_v2.pth"

    if not os.path.exists(model_path):
        st.info("⬇️ Downloading model from Google Drive...")

        file_id = "1VIgN44gUu8VqdkjG18EbHhRnAKc8xOwM"

        session = requests.Session()
        URL = "https://drive.google.com/uc?export=download"

        response = session.get(URL, params={'id': file_id}, stream=True)

        # handle Google Drive confirm token
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)

        # save file
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)

    return model_path


# --- LOAD MODEL ---
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

    model_path = download_model()

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    return mtcnn, model, device


# --- LOAD ---
try:
    mtcnn, model, device = load_tools()
    st.sidebar.success("Model Loaded Successfully ✔")
except Exception as e:
    st.error(f"Error loading AI tools: {e}")


# --- VIDEO UPLOAD ---
file = st.file_uploader(
    "Video File Chunein...",
    type=["mp4", "avi", "mov"],
    key="video_uploader_unique"   # 🔥 FIXED ERROR
)

if file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())
    tfile.close()

    col1, col2 = st.columns(2)

    with col1:
        st.video(file)

    if st.button('🚀 Start Full Video Scan'):
        with st.spinner('AI वीडियो के फ्रेम्स स्कैन कर रहा है...'):

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

                except:
                    continue

            cap.release()

            if all_probs:
                res = np.mean(all_probs, axis=0)
                label = np.argmax(res)
                conf = np.max(res) * 100

                final_res = "REAL" if label == 1 else "FAKE"

                with col2:
                    if conf < 60:
                        st.warning(f"AI Uncertain ({conf:.2f}%)")
                    elif final_res == "FAKE":
                        st.error(f"🚨 FAKE ({conf:.2f}%)")
                    else:
                        st.success(f"✅ REAL ({conf:.2f}%)")

                    st.metric("Confidence", f"{conf:.2f}%")
            else:
                st.warning("कोई face detect नहीं हुआ")

    try:
        os.unlink(tfile.name)
    except:
        pass
