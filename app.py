import streamlit as st
import cv2
import numpy as np
from roboflow import Roboflow
from datetime import datetime
import os
import gspread
from google.oauth2.service_account import Credentials

# --- CONFIGURACIÓN ---
API_KEY = "nOMi9VHi25eRhP420XFn"
WORKSPACE = "diseo-de-proyectos"
PROJECT_ID = "segmentacion-tumores-mamografia-sn1wk"
VERSION = 6 
SPREADSHEET_NAME = "Base_Datos_Pacientes"

# --- FUNCIONES DE CONEXIÓN ---
@st.cache_resource
def cargar_modelo():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT_ID)
        return project.version(VERSION).model
    except:
        return None

def conectar_google():
    try:
        creds_dict = st.secrets["google_drive_credentials"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        return gspread.authorize(creds)
    except:
        return None

# --- INTERFAZ (Tu diseño exacto) ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# Estilos CSS para los botones (Azul para ejecutar, Verde para nueva consulta)
st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .btn-ejecutar > div > button { background-color: #3498db; color: white; }
    .btn-nueva > div > button { background-color: #27ae60; color: white; }
    .header-box { background-color: #2c3e50; padding: 20px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# Encabezado
st.markdown("""
<div class="header-box">
    <h1 style="color: white; margin: 0; font-family: sans-serif;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
</div>
""", unsafe_allow_html=True)

st.info("**Gestión Hospitalaria:** Ingrese la filiación completa de la paciente y cargue el estudio para su procesamiento y registro.")

# --- FORMULARIO ---
# Usamos columnas para que se vea igual a tu imagen
c1, c2 = st.columns([1, 2])
tipo_registro = c1.selectbox("Registro:", ["Nuevo", "Existente"], key="reg_tipo")
expediente = c2.text_input("Expediente:", value="00478119")

c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_paterno = c4.text_input("A. Paterno:", value="Reyes")
a_materno = c5.text_input("A. Materno:", value="Morales")

uploader = st.file_uploader("📤 Subir Imagen (1)", type=["jpg", "png", "jpeg"])

# Contenedor para el botón de ejecutar con estilo azul
st.markdown('<div class="btn-ejecutar">', unsafe_allow_html=True)
ejecutar = st.button("Ejecutar Análisis Clínico")
st.markdown('</div>', unsafe_allow_html=True)

# --- PROCESAMIENTO ---
if ejecutar:
    if not uploader:
        st.error("❌ Por favor, cargue una imagen radiográfica para iniciar.")
    else:
        model = cargar_modelo()
        if model is None:
            st.error("❌ Error: No se pudo conectar con Roboflow. Verifique su API Key.")
        else:
            with st.spinner("🔬 Procesando imagen con Inteligencia Artificial..."):
                # 1. Leer y guardar imagen temporal
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                temp_path = "temp_analysis.jpg"
                cv2.imwrite(temp_path, img)

                try:
                    # 2. Inferencia Roboflow
                    prediction = model.predict(temp_path, confidence=40).json()
                    preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                    
                    st.subheader(f"Resultados de Análisis - {nombre} {a_paterno} {a_materno}")
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    tumor_px = 0
                    porcentaje = 0.0

                    # 3. Dibujar resultados
                    if not preds:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.success("✅ Análisis finalizado: No se detectaron hallazgos tumorales.")
                    else:
                        for p in preds:
                            pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                        
                        tumor_px = np.count_nonzero(mask)
                        porcentaje = (tumor_px / (h * w)) * 100
                        
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        overlay = img_rgb.copy()
                        overlay[mask > 0] = [255, 0, 0] # Rojo para el tumor
                        st.image(cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0), use_container_width=True)

                        # REPORTE TÉCNICO (Formato exacto de tu imagen)
                        st.markdown(f"""
                        <div style="border: 2px solid #3498db; padding: 20px; border-radius: 10px; background-color: #f8f9fa;">
                            <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; margin-top: 0;">REPORTE TÉCNICO DE SEGMENTACIÓN</h2>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                                <div><p style="color: #7f8c8d; margin:0;">PACIENTE</p><b>{nombre} {a_paterno} {a_materno}</b></div>
                                <div><p style="color: #7f8c8d; margin:0;">EXPEDIENTE</p><b>{expediente}</b></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                                <div><p style="color: #7f8c8d; margin:0;">PÍXELES TOTALES</p><b>{h*w:,} px</b></div>
                                <div><p style="color: #7f8c8d; margin:0;">PÍXELES TUMOR</p><b style="color: #c23616;">{tumor_px:,} px</b></div>
                            </div>
                            <div style="background-color: #fdf2e9; text-align: center; border: 1px solid #e67e22; padding: 15px; border-radius: 5px;">
                                <p style="color: #e67e22; margin:0; font-weight: bold;">ÁREA DE OCUPACIÓN TUMORAL</p>
                                <h1 style="color: #c23616; margin:0; font-size: 45px;">{porcentaje:.4f} %</h1>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # 4. Guardar en Google Sheets
                    gc = conectar_google()
                    if gc:
                        sh = gc.open(SPREADSHEET_NAME).sheet1
                        sh.append_row([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                            tipo_registro, expediente, nombre, a_paterno, a_materno, 
                            h*w, tumor_px, round(porcentaje, 4)
                        ])
                        st.toast("✅ Registro guardado en Base de Datos Hospitalaria.")

                except Exception as e:
                    st.error(f"Ocurrió un error técnico: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

# Espacio y botón de Nueva Consulta (Verde)
st.write("---")
st.markdown('<div class="btn-nueva">', unsafe_allow_html=True)
if st.button("Nueva Consulta"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
