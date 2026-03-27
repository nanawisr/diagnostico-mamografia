import streamlit as st
import cv2
import numpy as np
import pandas as pd
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

# --- CONEXIÓN GOOGLE SECRETS ---
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

# --- CARGA DEL MODELO ---
@st.cache_resource
def cargar_modelo():
    try:
        rf = Roboflow(api_key=API_KEY)
        return rf.workspace(WORKSPACE).project(PROJECT_ID).version(VERSION).model
    except:
        return None

# --- INTERFAZ (Tu diseño original) ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

st.markdown("""
<div style="background-color: #2c3e50; padding: 20px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px;">
    <h1 style="color: white; margin: 0; font-family: sans-serif;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
</div>
""", unsafe_allow_html=True)

st.info("**Gestión Hospitalaria:** Ingrese la filiación completa de la paciente y cargue el estudio para su procesamiento y registro.")

# Layout de campos como en tu imagen
c1, c2 = st.columns([1, 2])
tipo = c1.selectbox("Registro:", ["Nuevo", "Existente"])
exp = c2.text_input("Expediente:", value="00478119")

c3, c4, c5 = st.columns(3)
nom = c3.text_input("Nombre(s):", value="Ana")
pat = c4.text_input("A. Paterno:", value="Reyes")
mat = c5.text_input("A. Materno:", value="Morales")

uploader = st.file_uploader("📤 Subir Imagen (1)", type=["jpg", "png", "jpeg"])

# Botón Azul
st.markdown("""<style>div.stButton > button:first-child { background-color: #3498db; color: white; width: 100%; }</style>""", unsafe_allow_html=True)
boton = st.button("Ejecutar Análisis Clínico")

if boton:
    if not uploader:
        st.error("Por favor cargue una imagen.")
    else:
        model = cargar_modelo()
        if model:
            with st.spinner("Conectando con la red neuronal de Roboflow..."):
                # Procesamiento
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                cv2.imwrite("temp.jpg", img)
                
                prediction = model.predict("temp.jpg", confidence=40).json()
                preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                
                h, w, _ = img.shape
                mask = np.zeros((h, w), dtype=np.uint8)
                
                st.subheader(f"Resultados de Análisis - {nom} {pat} {mat}")
                
                if not preds:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    st.success("No se detectaron hallazgos.")
                    tumor_px, porcentaje = 0, 0
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

                    # REPORTE TÉCNICO (Como tu imagen)
                    st.markdown(f"""
                    <div style="border: 1px solid #3498db; padding: 20px; border-radius: 10px;">
                        <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db;">REPORTE TÉCNICO DE SEGMENTACIÓN</h2>
                        <table style="width:100%">
                            <tr>
                                <td><p>PACIENTE<br><b>{nom} {pat} {mat}</b></p></td>
                                <td><p>EXPEDIENTE<br><b>{exp}</b></p></td>
                            </tr>
                            <tr>
                                <td><p>PÍXELES TOTALES<br>{h*w} px</p></td>
                                <td><p>PÍXELES TUMOR<br><span style="color:red">{tumor_px} px</span></p></td>
                            </tr>
                        </table>
                        <div style="background-color: #fdf2e9; text-align: center; border: 1px solid #e67e22; padding: 10px;">
                            <p style="color: #e67e22; margin:0;">ÁREA DE OCUPACIÓN TUMORAL</p>
                            <h1 style="color: #c23616; margin:0;">{porcentaje:.4f} %</h1>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Sincronización
                gc = conectar_google()
                if gc:
                    sh = gc.open(SPREADSHEET_NAME).sheet1
                    sh.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), tipo, exp, nom, pat, mat, h*w, tumor_px, round(porcentaje, 4)])

# Botón verde abajo
st.markdown("""<style>div.stButton > button:last-child { background-color: #27ae60; color: white; }</style>""", unsafe_allow_html=True)
if st.button("Nueva Consulta"):
    st.rerun()
