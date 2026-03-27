import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
from datetime import datetime
import os
import gspread
from google.oauth2.service_account import Credentials

# --- CONFIGURACIÓN ---
API_KEY_PRIVATE = "nOMi9VHi25eRhP420XFn" 
WORKSPACE_ID = "diseo-de-proyectos"
PROJECT_NAME = "segmentacion-tumores-mamografia-sn1wk"
VERSION_NUM = 3 

# --- INTERFAZ ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #3498db !important; color: white !important; }
    .header-box { background-color: #2c3e50; padding: 20px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
</style>
<div class="header-box">
    <h1 style="color: white; margin: 0;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">Análisis Clínico Avanzado</p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns([1, 2])
tipo_reg = c1.selectbox("Registro:", ["Nuevo", "Existente"])
expediente = c2.text_input("Expediente:", value="00478119")

c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_pat = c4.text_input("A. Paterno:", value="Reyes")
a_mat = c5.text_input("A. Materno:", value="Morales")

uploader = st.file_uploader("📤 Subir Imagen (1)", type=["jpg", "png", "jpeg"])
ejecutar = st.button("Ejecutar Análisis Clínico")

if ejecutar:
    if not uploader:
        st.warning("⚠️ Cargue una imagen.")
    else:
        with st.spinner("🔬 Conectando con Roboflow..."):
            try:
                # --- NUEVA LÓGICA DE CONEXIÓN FORZADA ---
                rf = Roboflow(api_key=API_KEY_PRIVATE)
                workspace = rf.workspace(WORKSPACE_ID)
                project = workspace.project(PROJECT_NAME)
                
                # Intentamos obtener el modelo
                version = project.version(VERSION_NUM)
                model = version.model
                
                if model is None:
                    # Intento de rescate si el modelo falla
                    st.error("⚠️ El modelo no cargó correctamente. Intentando reconexión...")
                    model = project.version(VERSION_NUM).model

                # Si logramos conectar
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                temp_file = "img_analisis.jpg"
                cv2.imwrite(temp_file, img)

                # Inferencia
                prediction = model.predict(temp_file, confidence=40).json()
                preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                
                # Dibujar y mostrar
                mask = np.zeros((h, w), dtype=np.uint8)
                if not preds:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    st.success("✅ No se detectaron hallazgos.")
                else:
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    tumor_px = np.count_nonzero(mask)
                    porcentaje = (tumor_px / (h * w)) * 100
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    st.image(cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0), use_container_width=True)

                    st.markdown(f"""
                    <div style="border: 2px solid #3498db; padding: 20px; border-radius: 10px; background-color: #f8f9fa;">
                        <h2 style="color: #2c3e50;">REPORTE TÉCNICO</h2>
                        <h1 style="color: #c23616; text-align: center;">{porcentaje:.4f} %</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                if os.path.exists(temp_file): os.remove(temp_file)

            except Exception as e:
                st.error(f"❌ Error crítico de Roboflow: {str(e)}")
                st.info("Esto suele pasar si el proyecto en Roboflow no está desplegado o la versión 6 no tiene el modelo entrenado (Train).")

st.button("Nueva Consulta", on_click=lambda: st.rerun())
