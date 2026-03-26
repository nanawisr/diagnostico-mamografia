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

# --- CONFIGURACIÓN (REEMPLAZA ESTO) ---
API_KEY = "nOMi9VHi25eRhP420XFn"
WORKSPACE = "diseo-de-proyectos"
PROJECT_ID = "segmentacion-tumores-mamografia-sn1wk"
VERSION = 6 

# AQUÍ DEBES PEGAR EL NOMBRE EXACTO DE TU HOJA DE CÁLCULO
SPREADSHEET_NAME = "Base_Datos_Pacientes" 

# --- CONEXIÓN CON GOOGLE SECRETS ---
def conectar_google():
    creds_dict = st.secrets["google_drive_credentials"]
    creds = Credentials.from_service_account_info(creds_dict, scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ])
    return gspread.authorize(creds)

# --- INTERFAZ VISUAL ---
st.set_page_config(page_title="Diagnóstico Digital", layout="wide")

st.markdown("""
<div style="background-color: #2c3e50; padding: 25px; border-radius: 12px; border-left: 10px solid #3498db; margin-bottom: 20px;">
    <h1 style="color: white; margin: 0; font-family: sans-serif;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">Módulo de Análisis Clínico | Gestión Hospitalaria</p>
</div>
""", unsafe_allow_html=True)

# --- ENTRADA DE DATOS ---
with st.container():
    col1, col2 = st.columns([1, 2])
    tipo = col1.selectbox("Registro:", ["Nuevo", "Existente"])
    exp = col2.text_input("Expediente:", placeholder="ID Paciente")

    c1, c2, c3 = st.columns(3)
    nom = c1.text_input("Nombre(s):")
    pat = c2.text_input("A. Paterno:")
    mat = c3.text_input("A. Materno:")

    uploader = st.file_uploader("Cargar Radiografía", type=["jpg", "png", "jpeg"])
    boton = st.button("Ejecutar Análisis y Sincronizar", type="primary", use_container_width=True)

if boton:
    if not exp or not uploader:
        st.error("Error: Complete el expediente y cargue una imagen.")
    else:
        with st.spinner("Procesando..."):
            try:
                # Procesar Imagen
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                temp_path = "temp.jpg"
                cv2.imwrite(temp_path, img)

                # Roboflow
                rf = Roboflow(api_key=API_KEY)
                model = rf.workspace(WORKSPACE).project(PROJECT_ID).version(VERSION).model
                res = model.predict(temp_path, confidence=40).json()
                
                preds = [p for p in res['predictions'] if p.get('class') == 'tumor']
                h, w, _ = img.shape
                total_px = h * w
                mask = np.zeros((h, w), dtype=np.uint8)

                if not preds:
                    st.success("Análisis Finalizado: Sin hallazgos tumorales.")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    porcentaje = 0
                else:
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    tumor_px = np.count_nonzero(mask)
                    porcentaje = (tumor_px / total_px) * 100
                    
                    img_res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_res.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    img_final = cv2.addWeighted(img_res, 0.7, overlay, 0.3, 0)
                    
                    st.image(img_final, caption="Resultado de Segmentación")

                    # --- GUARDAR EN GOOGLE SHEETS ---
                    gc = conectar_google()
                    sh = gc.open(SPREADSHEET_NAME).sheet1
                    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    fila = [fecha, tipo, exp, nom, pat, mat, total_px, tumor_px, round(porcentaje, 4)]
                    sh.append_row(fila)

                    # --- REPORTE FINAL ---
                    st.markdown(f"""
                    <div style="border: 1px solid #dcdde1; padding: 20px; border-radius: 12px; background-color: white; font-family: sans-serif;">
                        <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db;">REPORTE TÉCNICO</h3>
                        <p><strong>Paciente:</strong> {nom} {pat} {mat} | <strong>Expediente:</strong> {exp}</p>
                        <div style="background: #fdf2e9; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #e67e22;">
                            <span style="color: #c23616; font-size: 24px; font-weight: bold;">{porcentaje:.4f} % de Ocupación</span>
                        </div>
                        <p style="font-size: 10px; color: #95a5a6; margin-top: 10px;">Sincronizado con Historial Clínico en Google Sheets.</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                if os.path.exists(temp_path): os.remove(temp_path)
