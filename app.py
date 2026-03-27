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
SHEET_NAME = "Base_Datos_Pacientes"

st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #1e88e5 !important; color: white !important; border: none; }
    .header-box { background-color: #34495e; padding: 25px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
    .header-box h1 { color: white; margin: 0; font-family: sans-serif; font-size: 42px; }
    .header-box p { color: #bdc3c7; margin: 5px 0 0 0; font-size: 18px; text-transform: uppercase; }
    .report-container { border: 1px solid #ced4da; padding: 20px; border-radius: 10px; background-color: white; font-family: sans-serif; margin-top: 20px; }
</style>
<div class="header-box">
    <h1>Plataforma de Diagnóstico Digital</h1>
    <p>MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
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
uploader = st.file_uploader("📤 Subir Imagen Radiográfica (1)", type=["jpg", "png", "jpeg"])

if st.button("Ejecutar Análisis Clínico"):
    if not uploader:
        st.warning("⚠️ Cargue una imagen.")
    else:
        with st.spinner("🔬 Procesando y Sincronizando..."):
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

                    # 2. SINCRONIZACIÓN (AUTO-REPARACIÓN DE PADDING)
                    drive_id = "No sincronizado"
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file_name = f"Analisis_{nombre}_{a_pat}.jpg"

                    if "service_account_base64" in st.secrets:
                        try:
                            b64_str = st.secrets["service_account_base64"].strip()
                            # REPARACIÓN DE PADDING: Añade '=' si faltan
                            missing_padding = len(b64_str) % 4
                            if missing_padding:
                                b64_str += '=' * (4 - missing_padding)
                            
                            decoded_bytes = base64.b64decode(b64_str)
                            info = json.loads(decoded_bytes)
                            info["private_key"] = info["private_key"].replace("\\n", "\n")
                            
                            creds = Credentials.from_service_account_info(info, 
                                    scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
                            
                            ds = build('drive', 'v3', credentials=creds)
                            cv2.imwrite(file_name, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                            media = MediaFileUpload(file_name, mimetype='image/jpeg')
                            df = ds.files().create(body={'name': file_name}, media_body=media, fields='id').execute()
                            drive_id = df.get('id')
                            if os.path.exists(file_name): os.remove(file_name)

                            gc = gspread.authorize(creds)
                            sh = gc.open(SHEET_NAME).sheet1
                            sh.append_row([now_str, tipo_reg, expediente, nombre, a_pat, a_mat, h*w, pix_tumor, round(porcentaje, 4), drive_id])
                            st.toast("✅ Base de Datos Sincronizada")
                        except Exception as e_cloud:
                            st.error(f"Error Sincronización: {e_cloud}")

                    # 3. REPORTE
                    st.markdown(f"""
                    <div class="report-container">
                        <div style="border-bottom: 2px solid #3498db; margin-bottom: 20px; padding-bottom: 10px; color: #2c3e50; font-size: 24px;">REPORTE TÉCNICO DE SEGMENTACIÓN</div>
                        <div style="background-color: #fff5f0; text-align: center; border: 1px solid #e67e22; padding: 20px; border-radius: 10px;">
                            <p style="color: #e67e22; margin:0; font-weight: bold; text-transform: uppercase;">ÁREA DE OCUPACIÓN TUMORAL</p>
                            <h1 style="color: #c23616; margin:0; font-size: 55px;">{porcentaje:.4f} %</h1>
                        </div>
                        <p style="color: #95a5a6; font-size: 13px; margin-top: 15px;">
                            Sincronizado con Historial Clínico.<br>
                            Imagen en Drive ID: <b>{drive_id}</b><br>
                            Fecha: {now_str}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

st.write("---")
if st.button("Limpiar y Nueva Consulta"):
    st.rerun()
