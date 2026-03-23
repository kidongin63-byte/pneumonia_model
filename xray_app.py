# pip install streamlit tensorflow pillow opencv-python-headless matplotlib numpy

import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model, Model

APP_TITLE = 'X-Ray 폐렴 분류 + Grad-CAM 시각화'
MODEL_PATH = Path(__file__).with_name('pneumonia_model.h5')
IMG_SIZE = 150


st.set_page_config(page_title=APP_TITLE, layout='wide')


@st.cache_resource
def load_keras_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
    return model


def find_last_conv_layer_name(model: tf.keras.Model) -> str:
    """모델에서 마지막 Conv2D 레이어 이름을 찾습니다."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("모델에서 Conv2D 레이어를 찾지 못했습니다.")


@st.cache_resource
def build_gradcam_model(_model: tf.keras.Model, last_conv_layer_name: str):
    """Sequential / Functional 모델 모두에서 동작하도록 Grad-CAM용 보조 모델 생성"""
    input_shape = _model.input_shape[1:]
    dummy_input = tf.keras.Input(shape=input_shape)

    x = dummy_input
    last_conv_output = None

    for layer in _model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x

    if last_conv_output is None:
        raise ValueError(f"마지막 Conv 레이어 '{last_conv_layer_name}'를 찾지 못했습니다.")

    grad_model = tf.keras.Model(inputs=dummy_input, outputs=[last_conv_output, x])
    return grad_model


def preprocess_uploaded_image(uploaded_file) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """업로드 이미지를 PIL과 numpy로 읽고 모델 입력 형태로 전처리합니다."""
    pil_image = Image.open(uploaded_file).convert('L')
    original_np = np.array(pil_image)

    resized = pil_image.resize((IMG_SIZE, IMG_SIZE))
    resized_np = np.array(resized).astype('float32') / 255.0
    model_input = np.expand_dims(resized_np, axis=(0, -1))  # (1, 150, 150, 1)

    return pil_image, original_np, model_input


def make_gradcam_heatmap(img_array: np.ndarray, grad_model: tf.keras.Model) -> np.ndarray:
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        if preds.shape[-1] == 1:
            loss = preds[:, 0]
        else:
            pred_index = tf.argmax(preds[0])
            loss = preds[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def apply_gradcam_overlay(img_array: np.ndarray, grad_model: tf.keras.Model, alpha: float = 0.4) -> np.ndarray:
    heatmap = make_gradcam_heatmap(img_array, grad_model)
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    orig = img_array[0]
    if orig.ndim == 3 and orig.shape[-1] == 1:
        orig = orig.squeeze(-1)

    if orig.max() <= 1.0:
        orig = orig * 255.0

    orig = np.uint8(orig)
    orig_rgb = np.stack([orig] * 3, axis=-1)
    overlay = cv2.addWeighted(orig_rgb, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay


def prediction_to_label(prob: float, threshold: float) -> tuple[str, str, float]:
    """sigmoid 출력이 폐렴 확률이라고 가정"""
    if prob >= threshold:
        return '폐렴', '위험 신호가 감지되었습니다.', prob
    return '정상', '정상 범위로 예측되었습니다.', 1.0 - prob



def image_to_download_bytes(image_array: np.ndarray) -> bytes:
    pil_img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer.read()


# -------------------------------
# UI
# -------------------------------
st.title(APP_TITLE)
st.caption('업로드한 흉부 X-Ray 이미지를 기반으로 폐렴 여부를 예측하고 Grad-CAM 시각화를 제공합니다.')

with st.sidebar:
    st.header('설정')
    threshold = st.slider('신뢰도 임계값', min_value=0.3, max_value=0.7, value=0.5, step=0.01)
    alpha = st.slider('히트맵 강도', min_value=0.1, max_value=0.9, value=0.4, step=0.05)
    st.markdown(
        '- **신뢰도 임계값**: 폐렴으로 판정하는 기준값\n'
        '- **히트맵 강도**: Grad-CAM 색상 강조 정도'
    )

try:
    model = load_keras_model()
    last_conv_name = find_last_conv_layer_name(model)
    grad_model = build_gradcam_model(model, last_conv_name)
except Exception as e:
    st.error(f'모델 로드 중 오류가 발생했습니다: {e}')
    st.stop()

uploaded_file = st.file_uploader('X-Ray 이미지를 업로드하세요', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    try:
        pil_image, original_np, model_input = preprocess_uploaded_image(uploaded_file)

        pred = model.predict(model_input, verbose=0)
        prob_pneumonia = float(pred.flatten()[0])
        pred_label, pred_message, displayed_confidence = prediction_to_label(prob_pneumonia, threshold)

        overlay = apply_gradcam_overlay(model_input, grad_model, alpha=alpha)
        overlay_for_download = cv2.resize(overlay, pil_image.size)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader('원본 X-Ray 이미지')
            st.image(original_np, caption='업로드 원본', use_container_width=True, clamp=True)

        with col2:
            st.subheader('Grad-CAM 히트맵 오버레이')
            st.image(overlay_for_download, caption=f'기준 레이어: {last_conv_name}', use_container_width=True)

            st.download_button(
                label='결과 이미지 다운로드',
                data=image_to_download_bytes(overlay_for_download),
                file_name='xray_gradcam_result.png',
                mime='image/png'
            )

        with col3:
            st.subheader('예측 결과')
            delta_text = f"임계값 {threshold:.2f}"
            st.metric('판정 결과', pred_label, delta=delta_text)

            st.write(pred_message)
            st.metric('폐렴 확률', f'{prob_pneumonia * 100:.1f}%')
            st.progress(int(np.clip(prob_pneumonia * 100, 0, 100)))

            normal_prob = 1.0 - prob_pneumonia
            st.metric('정상 확률', f'{normal_prob * 100:.1f}%')
            st.progress(int(np.clip(normal_prob * 100, 0, 100)))

            st.metric('최종 신뢰도', f'{displayed_confidence * 100:.1f}%')

            if pred_label == '폐렴':
                st.error('의심 소견이 있어 전문의 상담이 필요할 수 있습니다.')
            else:
                st.success('정상으로 예측되었습니다.')

        with st.expander('전처리 및 예측 정보 보기'):
            st.write(f'- 모델 파일: `{MODEL_PATH.name}`')
            st.write(f'- 입력 크기: `{model_input.shape}`')
            st.write(f'- 마지막 Conv 레이어: `{last_conv_name}`')
            st.write(f'- 폐렴 확률: `{prob_pneumonia:.4f}`')
            st.write(f'- 현재 임계값: `{threshold:.2f}`')

    except Exception as e:
        st.error(f'이미지 처리 또는 예측 중 오류가 발생했습니다: {e}')
else:
    st.info('좌측이 아닌 본문 업로드 영역에서 X-Ray 이미지를 선택하면 즉시 예측이 실행됩니다.')
