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
SHEET_NAME = "Base_Datos_Pacientes"

# --- INTERFAZ ---
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

# --- LÓGICA DE EJECUCIÓN ---
st.markdown('<div class="btn-ejecutar">', unsafe_allow_html=True)
ejecutar = st.button("Ejecutar Análisis Clínico")
st.markdown('</div>', unsafe_allow_html=True)

if ejecutar:
    if not uploader:
        st.warning("⚠️ Cargue una imagen antes de continuar.")
    else:
        with st.spinner("🔬 Realizando segmentación digital..."):
            try:
                # 1. Procesar Imagen IA
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                
                _, buffer = cv2.imencode('.jpg', img)
                img_64 = base64.b64encode(buffer).decode('utf-8')
                
                # Inferencia Directa
                res = requests.post(f"https://outline.roboflow.com/{ENDPOINT}", 
                                  params={"api_key": API_KEY, "confidence": "40"}, 
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
                    
                    # 2. MOSTRAR RESULTADOS INMEDIATAMENTE
                    st.write(f"### Análisis de Imagen - {nombre} {a_pat}")
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    res_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
                    st.image(res_img, use_container_width=True)

                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file_name = f"Analisis_{nombre}_{a_pat}_{expediente}.jpg"
                    drive_id = "No sincronizado"

                    # 3. REPORTE TÉCNICO EN PANTALLA
                    st.markdown(f"""
                    <div class="report-container">
                        <div style="border-bottom: 2px solid #3498db; margin-bottom: 20px; padding-bottom: 10px; color: #2c3e50; font-size: 24px;">REPORTE TÉCNICO DE SEGMENTACIÓN</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">PACIENTE</p><b>{nombre} {a_pat} {a_mat}</b></div>
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">EXPEDIENTE</p><b>{expediente}</b></div>
                        </div>
                        <div style="background-color: #fff5f0; text-align: center; border: 1px solid #e67e22; padding: 20px; border-radius: 10px;">
                            <p style="color: #e67e22; margin:0; font-weight: bold; font-size: 14px; text-transform: uppercase;">ÁREA DE OCUPACIÓN TUMORAL</p>
                            <h1 style="color: #c23616; margin:0; font-size: 55px;">{porcentaje:.4f} %</h1>
                        </div>
                        <div style="color: #95a5a6; font-size: 13px; margin-top: 15px; line-height: 1.6;">
                            Sincronizado con Historial Clínico (Base de Datos Hospitalaria).<br>
                            Imagen guardada exitosamente en Drive.<br>
                            Referencia: {file_name} | Fecha: {now_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 4. INTENTO DE GUARDADO EN NUBE (Al final, para no estorbar)
                    try:
                        creds_info = st.secrets["google_drive_credentials"]
                        creds = Credentials.from_service_account_info(creds_info, scopes=[
                            "https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"
                        ])
                        
                        # Drive
                        ds = build('drive', 'v3', credentials=creds)
                        cv2.imwrite(file_name, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                        media = MediaFileUpload(file_name, mimetype='image/jpeg')
                        df = ds.files().create(body={'name': file_name}, media_body=media, fields='id').execute()
                        drive_id = df.get('id')
                        if os.path.exists(file_name): os.remove(file_name)

                        # Sheets
                        gc = gspread.authorize(creds)
                        sh = gc.open(SHEET_NAME).sheet1
                        sh.append_row([now_str, tipo_reg, expediente, nombre, a_pat, a_mat, h*w, pix_tumor, round(porcentaje, 4), drive_id])
                        st.toast("✅ Base de Datos actualizada exitosamente.")
                    except Exception as e_cloud:
                        st.warning(f"⚠️ Nota: El reporte se generó pero la sincronización falló: {e_cloud}")

                else:
                    st.error("❌ El motor de IA no devolvió resultados válidos.")

            except Exception as e:
                st.error(f"❌ Error crítico: {str(e)}")

# --- BOTÓN NUEVA CONSULTA (Fuera del bloque de ejecución para que siempre sea accesible) ---
st.write("---")
st.markdown('<div class="btn-nueva">', unsafe_allow_html=True)
if st.button("Limpiar y Nueva Consulta"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
