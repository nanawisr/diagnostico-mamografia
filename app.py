import streamlit as st
import cv2
import numpy as np
import pandas as pd
from roboflow import Roboflow
from datetime import datetime
import contextlib
import io
import os
import gspread
from google.oauth2.service_account import Credentials

# --- CONFIGURACIÓN DE ACCESO ---
API_KEY = "nOMi9VHi25eRhP420XFn"
WORKSPACE = "diseo-de-proyectos"
PROJECT_ID = "segmentacion-tumores-mamografia-sn1wk"
VERSION = 6 
SPREADSHEET_NAME = "Base_Datos_Pacientes"

# --- CONEXIÓN CON GOOGLE DRIVE/SHEETS ---
def conectar_google():
    try:
        creds_dict = st.secrets["google_drive_credentials"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Error de autenticación con Google: {e}")
        return None

# --- CONFIGURACIÓN DE INTERFAZ ---
st.set_page_config(page_title="Diagnóstico Digital Mamográfico", layout="wide")

st.markdown("""
<style>
    .main-header {
        background-color: #2c3e50;
        padding: 25px;
        border-radius: 12px;
        border-left: 10px solid #3498db;
        margin-bottom: 20px;
    }
    div.stButton > button:first-child {
        background-color: #3498db;
        color: white;
        height: 3.5em;
        width: 100%;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        font-size: 16px;
    }
    div.stButton > button:hover {
        background-color: #2980b9;
        border: none;
    }
</style>
<div class="main-header">
    <h1 style="color: white; margin: 0; font-family: sans-serif; font-weight: 300;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0; font-size: 14px; text-transform: uppercase;">Módulo de Análisis Clínico Avanzado | Gestión Hospitalaria</p>
</div>
""", unsafe_allow_html=True)

# --- FORMULARIO DE PACIENTE ---
with st.container():
    col_reg, col_exp = st.columns([1, 2])
    tipo_reg = col_reg.selectbox("Registro:", ["Nuevo", "Existente"])
    expediente = col_exp.text_input("Expediente:", placeholder="Ej. 00478119")

    c1, c2, c3 = st.columns(3)
    nombre = c1.text_input("Nombre(s):")
    ap_paterno = c2.text_input("A. Paterno:")
    ap_materno = c3.text_input("A. Materno:")

    uploader = st.file_uploader("Cargar Radiografía para Análisis", type=["jpg", "png", "jpeg"])
    
    ejecutar = st.button("EJECUTAR ANÁLISIS Y SINCRONIZAR")

# --- LÓGICA DE PROCESAMIENTO ---
if ejecutar:
    if not expediente or not uploader:
        st.warning("⚠️ El número de expediente y la imagen son obligatorios.")
    else:
        with st.spinner("🔬 Conectando con la red neuronal de Roboflow..."):
            temp_file = "temp_img.jpg"
            try:
                # 1. Preparación de imagen
                img_data = uploader.read()
                img_np = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                cv2.imwrite(temp_file, img)

                # 2. Conexión Roboflow Blindada
                rf = Roboflow(api_key=API_KEY)
                try:
                    workspace = rf.workspace(WORKSPACE)
                    project = workspace.project(PROJECT_ID)
                    model = project.version(VERSION).model
                except Exception:
                    # Segundo intento con ruta directa si falla la jerárquica
                    model = rf.workspace(WORKSPACE).project(PROJECT_ID).version(VERSION).model

                if model is None:
                    st.error("❌ El modelo no respondió (NoneType). Verifique su conexión y API Key.")
                    st.stop()

                # 3. Inferencia
                prediction_res = model.predict(temp_file, confidence=40).json()
                preds = [p for p in prediction_res['predictions'] if p.get('class') == 'tumor']
                
                h, w, _ = img.shape
                total_px = h * w
                mask = np.zeros((h, w), dtype=np.uint8)

                if not preds:
                    st.success("✅ Análisis Finalizado: No se detectaron hallazgos tumorales.")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Estudio sin anomalías")
                    tumor_px, porcentaje = 0, 0
                else:
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    tumor_px = np.count_nonzero(mask)
                    porcentaje = (tumor_px / total_px) * 100
                    
                    # Visualización
                    img_view = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_view.copy()
                    overlay[mask > 0] = [255, 0, 0] # Rojo intenso
                    img_final = cv2.addWeighted(img_view, 0.7, overlay, 0.3, 0)
                    
                    st.image(img_final, caption="Resultado de Segmentación Digital", use_container_width=True)

                    # Reporte HTML
                    st.markdown(f"""
                    <div style="border: 1px solid #dcdde1; padding: 25px; border-radius: 12px; background-color: white; font-family: sans-serif; box-shadow: 0 4px 12px rgba(0,0,0,0.05); margin-top: 20px;">
                        <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-top: 0;">REPORTE CLÍNICO DE RESULTADOS</h3>
                        <p style="margin: 10px 0;"><strong>Paciente:</strong> {nombre} {ap_paterno} {ap_materno}<br><strong>ID Expediente:</strong> {expediente}</p>
                        <div style="background: #fdf2e9; padding: 20px; border-radius: 8px; border: 1px solid #e67e22; text-align: center; margin: 15px 0;">
                            <span style="color: #e67e22; font-size: 13px; font-weight: bold; text-transform: uppercase;">Ocupación Tumoral Detectada</span><br>
                            <span style="color: #c23616; font-size: 32px; font-weight: bold;">{porcentaje:.4f} %</span>
                        </div>
                        <p style="font-size: 11px; color: #95a5a6; border-top: 1px solid #eee; padding-top: 10px; margin-bottom: 0;">
                            Registro sincronizado con Historial Clínico Digital (Google Sheets).
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # 4. Sincronización Google Sheets
                gc = conectar_google()
                if gc:
                    sh = gc.open(SPREADSHEET_NAME).sheet1
                    fecha_ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    datos_fila = [fecha_ahora, tipo_reg, expediente, nombre, ap_paterno, ap_materno, total_px, tumor_px, round(porcentaje, 4)]
                    sh.append_row(datos_fila)
                    st.toast("✅ Sincronización Exitosa.")

            except Exception as error:
                st.error(f"Error en el flujo de diagnóstico: {error}")
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

if st.sidebar.button("Nueva Consulta"):
    st.rerun()
