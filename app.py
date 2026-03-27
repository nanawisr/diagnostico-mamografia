import streamlit as st
import cv2
import numpy as np
import pandas as pd
from roboflow import Roboflow
from datetime import datetime
import os
import gspread
from google.oauth2.service_account import Credentials

# --- CONFIGURACIÓN BASADA EN TU CAPTURA ---
API_KEY = "nOMi9VHi25eRhP420XFn"
WORKSPACE = "diseo-de-proyectos"
PROJECT_ID = "segmentacion-tumores-mamografia-sn1wk"
VERSION = 6 
SPREADSHEET_NAME = "Base_Datos_Pacientes"

# --- FUNCIÓN PARA CARGAR EL MODELO (OPTIMIZADA) ---
@st.cache_resource
def cargar_modelo():
    try:
        rf = Roboflow(api_key=API_KEY)
        project = rf.workspace(WORKSPACE).project(PROJECT_ID)
        # Usamos .model para inferencia en tiempo real, no .download
        model = project.version(VERSION).model
        return model
    except Exception as e:
        st.error(f"Error al conectar con Roboflow: {e}")
        return None

# --- CONEXIÓN CON GOOGLE SHEETS ---
def conectar_google():
    try:
        creds_dict = st.secrets["google_drive_credentials"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Error de permisos en Google: {e}")
        return None

# --- INTERFAZ VISUAL PROFESIONAL ---
st.set_page_config(page_title="Diagnóstico Digital - Anáhuac", layout="wide")

st.markdown("""
<style>
    .main-header {
        background-color: #2c3e50;
        padding: 30px;
        border-radius: 15px;
        border-left: 12px solid #3498db;
        margin-bottom: 25px;
    }
    div.stButton > button:first-child {
        background-color: #3498db;
        color: white;
        height: 3.5em;
        width: 100%;
        border-radius: 10px;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #2980b9;
        border: none;
    }
</style>
<div class="main-header">
    <h1 style="color: white; margin: 0; font-family: sans-serif;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">Análisis de Mamografías | Ingeniería Biomédica</p>
</div>
""", unsafe_allow_html=True)

# --- FORMULARIO ---
with st.container():
    col_exp, col_nom = st.columns([1, 2])
    exp = col_exp.text_input("Número de Expediente:", placeholder="Ej. 00478119")
    nom = col_nom.text_input("Nombre Completo del Paciente:")

    c1, c2 = st.columns(2)
    ap_p = c1.text_input("Apellido Paterno:")
    ap_m = c2.text_input("Apellido Materno:")

    uploader = st.file_uploader("Subir Imagen Radiográfica (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    boton = st.button("EJECUTAR ANÁLISIS CLÍNICO")

# --- PROCESAMIENTO ---
if boton:
    if not exp or not uploader:
        st.warning("⚠️ Ingrese el número de expediente y cargue una imagen.")
    else:
        # Cargamos el modelo (usará la versión en caché si ya se cargó una vez)
        model = cargar_modelo()
        
        if model:
            with st.spinner("🔬 Procesando imagen con Red Neuronal..."):
                try:
                    # 1. Preparar imagen
                    file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    temp_name = "analisis_actual.jpg"
                    cv2.imwrite(temp_name, img)

                    # 2. Inferencia (Predicción)
                    prediction = model.predict(temp_name, confidence=40).json()
                    preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                    
                    h, w, _ = img.shape
                    mask = np.zeros((h, w), dtype=np.uint8)

                    if not preds:
                        st.success("✅ Análisis Finalizado: Tejido sin anomalías detectadas.")
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Radiografía Original")
                        tumor_px, porcentaje = 0, 0
                    else:
                        for p in preds:
                            pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                        
                        tumor_px = np.count_nonzero(mask)
                        porcentaje = (tumor_px / (h * w)) * 100
                        
                        # Visualización de Segmentación
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        overlay = img_rgb.copy()
                        overlay[mask > 0] = [255, 0, 0] # Color rojo para tumor
                        resultado = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
                        
                        st.image(resultado, caption="Detección de Segmentación Tumoral", use_container_width=True)
                        
                        # Reporte en pantalla
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 10px; border: 1px solid #eee; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                            <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db;">Resultado del Análisis</h3>
                            <p><strong>Paciente:</strong> {nom} {ap_p} {ap_m}</p>
                            <p style="font-size: 24px; color: #c23616; font-weight: bold;">Ocupación Tumoral: {porcentaje:.4f} %</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 3. Sincronizar con Google Sheets
                    gc = conectar_google()
                    if gc:
                        sh = gc.open(SPREADSHEET_NAME).sheet1
                        sh.append_row([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                            "Nuevo", exp, nom, ap_p, ap_m, h*w, tumor_px, round(porcentaje, 4)
                        ])
                        st.toast("✅ Datos guardados en la nube.")

                except Exception as e:
                    st.error(f"Error durante el análisis: {e}")
                finally:
                    if os.path.exists(temp_name): os.remove(temp_name)
        else:
            st.error("No se pudo establecer conexión con el motor de IA.")

if st.sidebar.button("Nueva Consulta"):
    st.rerun()
