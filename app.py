import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import os
import gspread
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- CONFIGURACIÓN MAESTRA ---
API_KEY_ROBOFLOW = "nOMi9VHi25eRhP420XFn"
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/5"

# Identificadores Actualizados
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"
DRIVE_FOLDER_ID = "1gQchYqpMWvLBRQ8U-1Kl3yvB8j6uxmzd" # Tu Carpeta Prueba
MI_CORREO = "nana.wowrara@gmail.com"

# --- INTERFAZ ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #1e88e5 !important; color: white !important; border: none; }
    .header-box { background-color: #34495e; padding: 25px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; color: white; }
    .report-container { border: 1px solid #ced4da; padding: 20px; border-radius: 10px; background-color: white; margin-top: 20px; }
</style>
<div class="header-box">
    <h1 style='margin:0;'>Plataforma de Diagnóstico Digital</h1>
    <p style='margin:0; color:#bdc3c7;'>MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
</div>
""", unsafe_allow_html=True)

# Formulario (Nombres separados para el Excel)
c1, c2 = st.columns([1, 2])
tipo_reg = c1.selectbox("Registro:", ["Nuevo", "Existente"])
expediente = c2.text_input("Expediente:", value="00478119")

c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_pat = c4.text_input("A. Paterno:", value="Reyes")
a_mat = c5.text_input("A. Materno:", value="Morales")

uploader = st.file_uploader("📤 Subir Imagen Radiográfica", type=["jpg", "png", "jpeg"])

if st.button("Ejecutar Análisis y Guardar en Drive"):
    if not uploader:
        st.warning("⚠️ Cargue una imagen antes de continuar.")
    else:
        with st.spinner("🔬 Sincronizando con Google Drive..."):
            try:
                # 1. IA Roboflow
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                _, buffer = cv2.imencode('.jpg', img)
                img_64 = base64.b64encode(buffer).decode('utf-8')
                
                res = requests.post(f"https://outline.roboflow.com/{ENDPOINT_ROBOFLOW}", 
                                  params={"api_key": API_KEY_ROBOFLOW, "confidence": "40"}, 
                                  data=img_64, headers={"Content-Type": "application/x-www-form-urlencoded"})
                prediction = res.json()
                
                if "predictions" in prediction:
                    preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts
