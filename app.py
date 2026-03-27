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

# --- CONEXIÓN CON GOOGLE SECRETS ---
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

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Diagnóstico Digital Mamográfico", layout="wide")

# --- ESTILO VISUAL ---
st.markdown("""
<div style="background-color: #2c3e50; padding: 25px; border-radius: 12px; border-left: 10px solid #3498db; margin-bottom: 20px;">
    <h1 style="color: white; margin: 0; font-family: 'Segoe UI', sans-serif; font-weight: 300;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Módulo de Análisis Clínico | Gestión Hospitalaria</p>
</div>
""", unsafe_allow_html=True)

# --- ENTRADA DE DATOS ---
with st.container():
    col1, col2 = st.columns([1, 2])
    tipo = col1.selectbox("Registro de Paciente:", ["Nuevo", "Existente"])
    exp = col2.text_input("Número de Expediente:", placeholder="Ej. 00478119")

    c1, c2, c3 = st.columns(3)
    nom = c1.text_input("Nombre(s):")
    pat = c2.text_input("Apellido Paterno:")
    mat = c3.text_input("Apellido Materno:")

    uploader = st.file_uploader("Cargar Radiografía (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    # Botón con color personalizado mediante CSS
    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #3498db;
            color: white;
            height: 3em;
            width: 100%;
            border-radius: 5px;
            border: none;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #2980b9;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    boton = st.button("EJECUTAR ANÁLISIS Y SINCRONIZAR")

# --- LÓGICA DE PROCESAMIENTO ---
if boton:
    if not exp or not uploader:
        st.warning("⚠️ Atención: Es obligatorio ingresar el expediente y cargar una imagen.")
    else:
        with st.spinner("🔄 Conectando con servidores médicos y procesando tejido..."):
            temp_path = "temp_analysis.jpg"
            try:
                # 1. Cargar imagen
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                cv2.imwrite(temp_path, img)

                # 2. Conectar a Roboflow (Corregido para evitar NoneType)
                rf = Roboflow(api_key=API_KEY)
                project = rf.workspace(WORKSPACE).project(PROJECT_ID)
                model = project.version(VERSION).model
                
                if model is None:
                    st.error("❌ Error: No se pudo instanciar el modelo de Roboflow. Verifique IDs.")
                    st.stop()

                # 3. Predicción
                res = model.predict(temp_path, confidence=40).json()
                preds = [p for p in res['predictions'] if p.get('class') == 'tumor']
                
                h, w, _ = img.shape
                total_px = h * w
                mask = np.zeros((h, w), dtype=np.uint8)

                # 4. Resultados visuales
                if not preds:
                    st.success("✅ Análisis Finalizado: Sin hallazgos tumorales detectados.")
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Estudio Limpio")
                    tumor_px = 0
                    porcentaje = 0
                else:
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    tumor_px = np.count_nonzero(mask)
                    porcentaje = (tumor_px / total_px) * 100
                    
                    img_res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_res.copy()
                    overlay[mask > 0] = [255, 0, 0] # Rojo para el tumor
                    img_final = cv2.addWeighted(img_res, 0.7, overlay, 0.3, 0)
                    
                    st.image(img_final, caption="Visualización de Segmentación Tumoral", use_container_width=True)

                    # --- REPORTE CLÍNICO EN PANTALLA ---
                    st.markdown(f"""
                    <div style="border: 1px solid #dcdde1; padding: 25px; border-radius: 12px; background-color: white; font-family: sans-serif; box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
                        <h3 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">REPORTE TÉCNICO DE SEGMENTACIÓN</h3>
                        <p style="margin: 10px 0;"><strong>Paciente:</strong> {nom} {pat} {mat} | <strong>ID:</strong> {exp}</p>
                        <div style="background: #fdf2e9; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid #e67e22; margin: 15px 0;">
                            <span style="color: #c23616; font-size: 28px; font-weight: bold;">{porcentaje:.4f} % Ocupación Tumoral</span>
                            <br><span style="color: #7f8c8d; font-size: 12px;">Detección basada en píxeles segmentados</span>
                        </div>
                        <p style="font-size: 11px; color: #95a5a6; border-top: 1px solid #eee; padding-top: 10px;">
                            Sincronizado con Historial Clínico Digital | Google Sheets Cloud.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # 5. Sincronización con Google Sheets
                gc = conectar_google()
                if gc:
                    sh = gc.open(SPREADSHEET_NAME).sheet1
                    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    fila = [fecha, tipo, exp, nom, pat, mat, total_px, tumor_px, round(porcentaje, 4)]
                    sh.append_row(fila)
                    st.toast("✅ Datos sincronizados correctamente en el historial.")

            except Exception as e:
                st.error(f"⚠️ Error en el sistema: {e}")
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

# Botón de Nueva Consulta (opcional, refresca la página)
if st.sidebar.button("Limpiar Pantalla / Nueva Consulta"):
    st.rerun()
