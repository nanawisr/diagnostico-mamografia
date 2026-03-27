import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
from datetime import datetime
import os
import gspread
from google.oauth2.service_account import Credentials

# --- DATOS DE CONEXIÓN (VERIFICADOS) ---
# Usamos tu Private Key que termina en 0XFn
API_KEY_PRIVATE = "nOMi9VHi25eRhP420XFn" 
WORKSPACE_ID = "diseo-de-proyectos"
PROJECT_NAME = "segmentacion-tumores-mamografia-sn1wk"
VERSION_NUM = 6 
SPREADSHEET_NAME = "Base_Datos_Pacientes"

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# Estilos exactos de tu interfaz
st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .btn-ejecutar > div > button { background-color: #3498db !important; color: white !important; border: none; }
    .btn-nueva > div > button { background-color: #f8f9fa !important; color: #2c3e50 !important; border: 1px solid #ccc; }
    .header-box { background-color: #2c3e50; padding: 20px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
</style>
<div class="header-box">
    <h1 style="color: white; margin: 0; font-family: sans-serif;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
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

uploader = st.file_uploader("📤 Subir Imagen (1)", type=["jpg", "png", "jpeg"])

st.markdown('<div class="btn-ejecutar">', unsafe_allow_html=True)
ejecutar = st.button("Ejecutar Análisis Clínico")
st.markdown('</div>', unsafe_allow_html=True)

# --- LÓGICA DE PROCESAMIENTO ---
if ejecutar:
    if not uploader:
        st.warning("⚠️ Por favor, cargue una imagen primero.")
    else:
        with st.spinner("🔬 Conectando con el motor de IA..."):
            try:
                # 1. Conexión Directa (Sin caché para evitar el error NoneType)
                rf = Roboflow(api_key=API_KEY_PRIVATE)
                project = rf.workspace(WORKSPACE_ID).project(PROJECT_NAME)
                model = project.version(VERSION_NUM).model
                
                # 2. Procesar Imagen
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                temp_file = "img_analisis.jpg"
                cv2.imwrite(temp_file, img)

                # 3. Predicción
                prediction = model.predict(temp_file, confidence=40).json()
                preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                
                st.subheader(f"Resultados de Análisis - {nombre} {a_pat} {a_mat}")
                
                mask = np.zeros((h, w), dtype=np.uint8)
                tumor_px, porcentaje = 0, 0.0

                if not preds:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.success("✅ No se detectaron hallazgos tumorales en la muestra.")
                else:
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    tumor_px = np.count_nonzero(mask)
                    porcentaje = (tumor_px / (h * w)) * 100
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0] # Color rojo para tumor
                    st.image(cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0), use_container_width=True)

                    # REPORTE TÉCNICO (Tu diseño original)
                    st.markdown(f"""
                    <div style="border: 2px solid #3498db; padding: 20px; border-radius: 10px; background-color: #f8f9fa;">
                        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; margin-top: 0;">REPORTE TÉCNICO DE SEGMENTACIÓN</h2>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                            <div><p style="color: #7f8c8d; margin:0;">PACIENTE</p><b>{nombre} {a_pat} {a_mat}</b></div>
                            <div><p style="color: #7f8c8d; margin:0;">EXPEDIENTE</p><b>{expediente}</b></div>
                        </div>
                        <div style="background-color: #fdf2e9; text-align: center; border: 1px solid #e67e22; padding: 15px; border-radius: 5px;">
                            <p style="color: #e67e22; margin:0; font-weight: bold;">ÁREA DE OCUPACIÓN TUMORAL</p>
                            <h1 style="color: #c23616; margin:0; font-size: 45px;">{porcentaje:.4f} %</h1>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # 4. Sincronizar con Google Sheets (opcional, si tienes st.secrets configurado)
                try:
                    creds_dict = st.secrets["google_drive_credentials"]
                    creds = Credentials.from_service_account_info(creds_dict, scopes=[
                        "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"
                    ])
                    gc = gspread.authorize(creds)
                    sh = gc.open(SPREADSHEET_NAME).sheet1
                    sh.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tipo_reg, expediente, nombre, a_pat, a_mat, h*w, tumor_px, round(porcentaje, 4)])
                    st.toast("✅ Sincronizado con Historial Clínico.")
                except:
                    pass

                if os.path.exists(temp_file): os.remove(temp_file)

            except Exception as e:
                st.error(f"❌ Error de conexión: {str(e)}")
                st.info("Sugerencia: Revisa que la API Key no tenga espacios invisibles al final.")

st.write("---")
st.markdown('<div class="btn-nueva">', unsafe_allow_html=True)
if st.button("Nueva Consulta"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
