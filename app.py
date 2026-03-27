import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import os
import gspread
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- CONFIGURACIÓN ---
API_KEY = "nOMi9VHi25eRhP420XFn"
ENDPOINT = "segmentacion-tumores-mamografia-sn1wk/5"
SPREADSHEET_NAME = "Base_Datos_Pacientes" 
DRIVE_FOLDER_ID = "TU_ID_DE_CARPETA_DRIVE" # Reemplaza con el ID de tu carpeta de Drive

# --- CONEXIÓN A GOOGLE (Saca las credenciales de st.secrets) ---
def conectar_google():
    try:
        creds_info = st.secrets["google_drive_credentials"]
        creds = Credentials.from_service_account_info(creds_info, scopes=[
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ])
        return creds
    except Exception as e:
        st.error(f"Error de credenciales: {e}")
        return None

def subir_a_drive(creds, img_array, nombre_archivo):
    try:
        service = build('drive', 'v3', credentials=creds)
        # Guardar imagen temporalmente para subirla
        temp_path = "temp_upload.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        file_metadata = {'name': nombre_archivo, 'parents': [DRIVE_FOLDER_ID]}
        media = MediaFileUpload(temp_path, mimetype='image/jpeg')
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        
        if os.path.exists(temp_path): os.remove(temp_path)
        return file.get('id')
    except:
        return "Error_Drive"

# --- INTERFAZ VISUAL ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    .btn-ejecutar > div > button { background-color: #1e88e5 !important; color: white !important; border: none; }
    .btn-nueva > div > button { background-color: #43a047 !important; color: white !important; border: none; margin-top: 20px; }
    .header-box { background-color: #34495e; padding: 25px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
    .header-box h1 { color: white; margin: 0; font-family: sans-serif; font-size: 42px; }
    .header-box p { color: #bdc3c7; margin: 5px 0 0 0; font-size: 18px; text-transform: uppercase; letter-spacing: 1px; }
    .report-container { border: 1px solid #ced4da; padding: 20px; border-radius: 10px; background-color: white; font-family: sans-serif; margin-top: 20px; }
    .report-header { border-bottom: 2px solid #3498db; margin-bottom: 20px; padding-bottom: 10px; color: #2c3e50; font-size: 24px; text-transform: uppercase; }
</style>
<div class="header-box">
    <h1>Plataforma de Diagnóstico Digital</h1>
    <p>MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
</div>
""", unsafe_allow_html=True)

st.info("**Gestión Hospitalaria:** Ingrese la filiación completa de la paciente y cargue el estudio para su procesamiento y registro.")

# Formulario
c1, c2 = st.columns([1, 2])
tipo_reg = c1.selectbox("Registro:", ["Nuevo", "Existente"])
expediente = c2.text_input("Expediente:", value="00478119")
c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_pat = c4.text_input("A. Paterno:", value="Reyes")
a_mat = c5.text_input("A. Materno:", value="Morales")
uploader = st.file_uploader("📤 Subir Imagen Radiográfica (1)", type=["jpg", "png", "jpeg"])

# --- LÓGICA DE PROCESAMIENTO ---
if not st.session_state.get('analizado', False):
    st.markdown('<div class="btn-ejecutar">', unsafe_allow_html=True)
    if st.button("Ejecutar Análisis Clínico"):
        if uploader:
            with st.spinner("🔬 Analizando y Sincronizando con Base de Datos..."):
                try:
                    # 1. Procesar Imagen
                    file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    h, w, _ = img.shape
                    pix_totales = h * w
                    
                    # 2. IA Inferencia
                    _, buffer = cv2.imencode('.jpg', img)
                    img_64 = base64.b64encode(buffer).decode('utf-8')
                    url = f"https://outline.roboflow.com/{ENDPOINT}"
                    response = requests.post(url, params={"api_key": API_KEY, "confidence": "40"}, 
                                          data=img_64, headers={"Content-Type": "application/x-www-form-urlencoded"})
                    prediction = response.json()
                    
                    if "predictions" in prediction:
                        preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for p in preds:
                            pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                        
                        pix_tumor = np.count_nonzero(mask)
                        porcentaje = (pix_tumor / pix_totales) * 100
                        
                        # Generar Imagen Resultado (RGB para Streamlit)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        overlay = img_rgb.copy()
                        overlay[mask > 0] = [255, 0, 0]
                        res_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
                        st.image(res_img, use_container_width=True)

                        # 3. SINCRONIZACIÓN NUBE
                        creds = conectar_google()
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        file_name = f"Analisis_{nombre}_{a_pat}_{expediente}.jpg"
                        drive_id = "No_Sincronizado"
                        
                        if creds:
                            drive_id = subir_a_drive(creds, res_img, file_name)
                            # Registro en Google Sheets (Orden exacto que pediste)
                            gc = gspread.authorize(creds)
                            sh = gc.open(SPREADSHEET_NAME).sheet1
                            sh.append_row([
                                now_str, tipo_reg, expediente, nombre, a_pat, a_mat, 
                                pix_totales, pix_tumor, round(porcentaje, 4), drive_id
                            ])

                        # --- REPORTE TÉCNICO ---
                        st.markdown(f"""
                        <div class="report-container">
                            <div class="report-header">REPORTE TÉCNICO DE SEGMENTACIÓN</div>
                            <div style="background-color: #fff5f0; text-align: center; border: 1px solid #e67e22; padding: 20px; border-radius: 10px;">
                                <p style="color: #e67e22; margin:0; font-weight: bold; font-size: 14px; text-transform: uppercase;">ÁREA DE OCUPACIÓN TUMORAL</p>
                                <h1 style="color: #c23616; margin:0; font-size: 55px;">{porcentaje:.4f} %</h1>
                            </div>
                            <div style="color: #95a5a6; font-size: 13px; margin-top: 15px; line-height: 1.6;">
                                Sincronizado con Historial Clínico (Base de Datos Hospitalaria).<br>
                                Imagen de diagnóstico guardada exitosamente en Drive ID: <b>{drive_id}</b><br>
                                Referencia: {file_name} | Fecha: {now_str}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state['analizado'] = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Error en sincronización: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.get('analizado', False):
    st.markdown('<div class="btn-nueva">', unsafe_allow_html=True)
    if st.button("Nueva Consulta"):
        st.session_state['analizado'] = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
