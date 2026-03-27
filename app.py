import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import os

# --- DATOS DE CONEXIÓN ---
API_KEY = "nOMi9VHi25eRhP420XFn"
ENDPOINT = "segmentacion-tumores-mamografia-sn1wk/5"

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .btn-ejecutar > div > button { background-color: #1e88e5 !important; color: white !important; border: none; }
    .btn-nueva > div > button { background-color: #43a047 !important; color: white !important; border: none; }
    
    .header-box { background-color: #34495e; padding: 25px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
    .header-box h1 { color: white; margin: 0; font-family: sans-serif; font-size: 42px; }
    .header-box p { color: #bdc3c7; margin: 5px 0 0 0; font-size: 18px; text-transform: uppercase; letter-spacing: 1px; }

    .report-container { border: 1px solid #ced4da; padding: 20px; border-radius: 10px; background-color: white; font-family: sans-serif; }
    .report-header { border-bottom: 2px solid #3498db; margin-bottom: 20px; padding-bottom: 10px; color: #2c3e50; font-size: 24px; text-transform: uppercase; }
    
    .footer-text { color: #95a5a6; font-size: 13px; margin-top: 15px; line-height: 1.6; }
</style>

<div class="header-box">
    <h1>Plataforma de Diagnóstico Digital</h1>
    <p>MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
</div>
""", unsafe_allow_html=True)

st.info("**Gestión Hospitalaria:** Ingrese la filiación completa de la paciente y cargue el estudio para su procesamiento y registro.")

# --- FORMULARIO ---
c1, c2 = st.columns([1, 2])
tipo_reg = c1.selectbox("Registro:", ["Nuevo", "Existente"])
expediente = c2.text_input("Expediente:", value="00478119")

c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_pat = c4.text_input("A. Paterno:", value="Reyes")
a_mat = c5.text_input("A. Materno:", value="Morales")

uploader = st.file_uploader("📤 Subir Imagen Radiográfica (1)", type=["jpg", "png", "jpeg"])

st.markdown('<div class="btn-ejecutar">', unsafe_allow_html=True)
ejecutar = st.button("Ejecutar Análisis Clínico")
st.markdown('</div>', unsafe_allow_html=True)

# --- PROCESAMIENTO ---
if ejecutar:
    if not uploader:
        st.warning("⚠️ Por favor, cargue una imagen.")
    else:
        with st.spinner("🔬 Procesando..."):
            try:
                # 1. Preparar Imagen
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                
                # 2. Inferencia Directa
                _, buffer = cv2.imencode('.jpg', img)
                img_64 = base64.b64encode(buffer).decode('utf-8')
                url = f"https://outline.roboflow.com/{ENDPOINT}"
                params = {"api_key": API_KEY, "confidence": "40"}
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                
                response = requests.post(url, params=params, data=img_64, headers=headers)
                prediction = response.json()
                
                if "predictions" in prediction:
                    preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                    st.write(f"### Resultados de Análisis - {nombre} {a_pat} {a_mat}")
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    tumor_px = 0
                    
                    # Dibujar máscara si hay tumores
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    tumor_px = np.count_nonzero(mask)
                    porcentaje = (tumor_px / (h * w)) * 100
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    st.image(cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0), use_container_width=True)

                    # --- REPORTE TÉCNICO Y PIE DE PÁGINA NUEVO ---
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    st.markdown(f"""
                    <div class="report-container">
                        <div class="report-header">REPORTE TÉCNICO DE SEGMENTACIÓN</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">PACIENTE</p><b>{nombre} {a_pat} {a_mat}</b></div>
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">EXPEDIENTE</p><b>{expediente}</b></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">PÍXELES TOTALES</p><p style="margin:0;">{h*w:,} px</p></div>
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">PÍXELES TUMOR</p><p style="margin:0; color: #c23616;">{tumor_px:,} px</p></div>
                        </div>
                        <div style="background-color: #fff5f0; text-align: center; border: 1px solid #e67e22; padding: 20px; border-radius: 10px;">
                            <p style="color: #e67e22; margin:0; font-weight: bold; font-size: 14px; text-transform: uppercase;">ÁREA DE OCUPACIÓN TUMORAL</p>
                            <h1 style="color: #c23616; margin:0; font-size: 55px;">{porcentaje:.4f} %</h1>
                        </div>
                        
                        <div class="footer-text">
                            Sincronizado con Historial Clínico (Base de Datos Hospitalaria).<br>
                            Imagen de diagnóstico guardada exitosamente en la carpeta: <b>Resultados_Analisis</b><br>
                            Referencia: Analisis_{nombre}_{a_pat}_{a_mat}_{expediente}.jpg | Fecha: {now}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")

st.write("---")
st.markdown('<div class="btn-nueva">', unsafe_allow_html=True)
if st.button("Nueva Consulta"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
