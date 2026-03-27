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

# Tus IDs (Asegúrate de que el DRIVE_FOLDER_ID sea el de 'base_datos_pacientes')
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"
DRIVE_FOLDER_ID = "1S66F3LwaWazDogCbcU8kGJH91vKTuyHb"
MI_CORREO = "nana.wowrara@gmail.com"

st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# --- ESTILOS ---
st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #1e88e5 !important; color: white !important; }
    .header-box { background-color: #34495e; padding: 25px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
    .header-box h1 { color: white; margin: 0; font-family: sans-serif; }
    .report-container { border: 1px solid #ced4da; padding: 20px; border-radius: 10px; background-color: white; margin-top: 20px; }
</style>
<div class="header-box">
    <h1>Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 0;">ANÁLISIS DE MAMOGRAFÍA - RUTA: BASE_DATOS_PACIENTES</p>
</div>
""", unsafe_allow_html=True)

# Formulario
c1, c2 = st.columns([1, 2])
tipo_reg = c1.selectbox("Registro:", ["Nuevo", "Existente"])
expediente = c2.text_input("Expediente:", value="00478119")
c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_pat = c4.text_input("A. Paterno:", value="Reyes")
a_mat = c5.text_input("A. Materno:", value="Morales")
uploader = st.file_uploader("📤 Subir Imagen Radiográfica", type=["jpg", "png", "jpeg"])

if st.button("Ejecutar Análisis Clínico"):
    if not uploader:
        st.warning("⚠️ Cargue una imagen.")
    else:
        with st.spinner("🔬 Procesando y sincronizando en la ruta especificada..."):
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
                        cv2.fillPoly(mask, [pts], 255)
                    
                    pix_tumor = np.count_nonzero(mask)
                    porcentaje = (pix_tumor / (h * w)) * 100
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    res_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
                    
                    st.image(res_img, use_container_width=True)

                    # 2. SINCRONIZACIÓN
                    drive_id = "Pendiente (Error de Cuota)"
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file_name = f"Analisis_{nombre}_{a_pat}.jpg"

                    if "service_account_base64" in st.secrets:
                        try:
                            b64_str = st.secrets["service_account_base64"].strip()
                            b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
                            info = json.loads(base64.b64decode(b64_str))
                            info["private_key"] = info["private_key"].replace("\\n", "\n")
                            creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
                            
                            # Intentar Drive
                            try:
                                ds = build('drive', 'v3', credentials=creds)
                                cv2.imwrite(file_name, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                                media = MediaFileUpload(file_name, mimetype='image/jpeg', resumable=True)
                                df = ds.files().create(body={'name': file_name, 'parents': [DRIVE_FOLDER_ID]}, media_body=media, fields='id', supportsAllDrives=True).execute()
                                drive_id = df.get('id')
                                ds.permissions().create(fileId=drive_id, body={'type': 'user', 'role': 'writer', 'emailAddress': MI_CORREO}).execute()
                            except: pass
                            
                            if os.path.exists(file_name): os.remove(file_name)

                            # Sheets
                            gc = gspread.authorize(creds)
                            sh = gc.open_by_key(SHEET_ID).sheet1
                            sh.append_row([now_str, tipo_reg, expediente, nombre, a_pat, a_mat, h*w, pix_tumor, round(porcentaje, 4), drive_id])
                            st.toast("✅ Base de Datos Sincronizada")
                        except Exception as e:
                            st.error(f"Error Sincronización: {e}")

                    # 3. REPORTE
                    st.markdown(f"""
                    <div class="report-container">
                        <h2 style="color: #2c3e50;">REPORTE TÉCNICO</h2>
                        <div style="background-color: #fff5f0; text-align: center; padding: 25px; border-radius: 10px; border: 1px solid #e67e22;">
                            <h1 style="color: #c23616; font-size: 65px; margin:0;">{porcentaje:.4f} %</h1>
                        </div>
                        <p style="color: #95a5a6; font-size: 13px; margin-top: 15px;">ID Drive: {drive_id} | Fecha: {now_str}</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error Crítico: {e}")

if st.button("Nueva Consulta"):
    st.rerun()
