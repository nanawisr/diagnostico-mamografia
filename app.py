import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import gspread
import json
from google.oauth2.service_account import Credentials

# --- CONFIGURACIÓN MAESTRA ---
# Estos IDs son los que ya tienes configurados
API_KEY_ROBOFLOW = "nOMi9VHi25eRhP420XFn"
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/5"
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Sistema de Diagnóstico Digital", layout="wide")

# Inicialización de estados para que la app no se reinicie sola
if 'analizado' not in st.session_state:
    st.session_state.analizado = False
if 'datos_reporte' not in st.session_state:
    st.session_state.datos_reporte = None
if 'res_img' not in st.session_state:
    st.session_state.res_img = None

# --- ESTILOS CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Serif+Pro:wght@400;600&display=swap');
    html, body, [class*="st-"] { font-family: 'Source Serif Pro', serif; }
    .stButton > button { width: 100%; border-radius: 2px; height: 3em; font-weight: 600; background-color: #2c3e50 !important; color: white !important; border: none; text-transform: uppercase; letter-spacing: 1px; }
    .header-box { background-color: #f8f9fa; padding: 30px; border-bottom: 3px solid #2c3e50; margin-bottom: 30px; text-align: center; }
    .header-box h1 { color: #2c3e50; margin: 0; font-family: 'Libre Baskerville', serif; font-size: 36px; font-weight: 700; }
    .instruction-text { color: #34495e; font-size: 18px; margin-bottom: 25px; text-align: center; padding-top: 15px; }
    .report-container { border: 2px solid #2c3e50; padding: 40px; background-color: #ffffff; margin-top: 30px; box-shadow: 5px 5px 15px rgba(0,0,0,0.05); }
    .report-header { border-bottom: 1px solid #2c3e50; margin-bottom: 30px; padding-bottom: 10px; color: #2c3e50; font-size: 22px; font-weight: 700; text-align: center; font-family: 'Libre Baskerville', serif; }
    .data-label { color: #7f8c8d; font-size: 13px; text-transform: uppercase; font-weight: 600; }
    .data-value { color: #2c3e50; font-size: 20px; margin-bottom: 20px; }
    .result-box { background-color: #fdfdfd; text-align: center; border: 1px solid #dcdde1; padding: 30px; margin-top: 30px; }
</style>
<div class="header-box">
    <h1>PLATAFORMA DE DIAGNÓSTICO DIGITAL</h1>
    <p>MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
</div>
""", unsafe_allow_html=True)

# --- FLUJO DE TRABAJO ---
if not st.session_state.analizado:
    st.markdown('<p class="instruction-text">Por favor, rellene los datos solicitados e inserte la mamografía para su análisis facultativo.</p>', unsafe_allow_html=True)
    
    expediente = st.text_input("Número de Expediente:", value="00478119")
    c1, c2, c3 = st.columns(3)
    nombre = c1.text_input("Nombre(s):", value="Ana")
    a_pat = c2.text_input("Apellido Paterno:", value="Reyes")
    a_mat = c3.text_input("Apellido Materno:", value="Morales")
    
    uploader = st.file_uploader("Seleccionar Imagen Radiográfica", type=["jpg", "png", "jpeg"])

    if st.button("INICIAR PROTOCOLO DE ANÁLISIS"):
        if not uploader:
            st.warning("Se requiere la carga de una imagen.")
        else:
            with st.spinner("Ejecutando análisis de segmentación..."):
                try:
                    # 1. IA Roboflow
                    file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    h, w, _ = img.shape
                    
                    _, buffer = cv2.imencode('.jpg', img)
                    img_64 = base64.b64encode(buffer).decode('utf-8')
                    
                    res = requests.post(
                        f"https://outline.roboflow.com/{ENDPOINT_ROBOFLOW}",
                        params={"api_key": API_KEY_ROBOFLOW, "confidence": "40"},
                        data=img_64,
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                    prediction = res.json()
                    
                    # 2. Procesamiento de Máscara
                    pix_tumor = 0
                    porcentaje = 0.0
                    mask = np.zeros((h, w), dtype=np.uint8)
                    
                    if "predictions" in prediction:
                        preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                        for p in preds:
                            pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                        
                        pix_tumor = int(np.count_nonzero(mask))
                        porcentaje = float((pix_tumor / (h * w)) * 100)

                    # 3. Superposición de colores
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0] # Tumor en Rojo
                    res_img = cv2.addWeighted(img_rgb, 0.8, overlay, 0.2, 0)

                    # 4. Subida a ImgBB para obtener URL
                    api_key_imgbb = st.secrets["API_KEY_IMGBB"]
                    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                    img_bb_64 = base64.b64encode(img_encoded).decode('utf-8')
                    
                    response_bb = requests.post(
                        "https://api.imgbb.com/1/upload",
                        data={"key": api_key_imgbb, "image": img_bb_64}
                    )
                    url_imagen = response_bb.json()["data"]["url"]

                    # 5. Registro en Google Sheets
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if "GCP_JSON" in st.secrets:
                        # Cargamos la cuenta de servicio desde el secreto JSON
                        info = json.loads(st.secrets["GCP_JSON"])
                        info["private_key"] = info["private_key"].replace("\\n", "\n")
                        
                        creds = Credentials.from_service_account_info(
                            info, 
                            scopes=["https://www.googleapis.com/auth/spreadsheets"]
                        )
                        gc = gspread.authorize(creds)
                        sh = gc.open_by_key(SHEET_ID).sheet1
                        
                        # Guardamos en las columnas correspondientes
                        sh.append_row([
                            now_str,        # A: Fecha
                            str(expediente),# B: Expediente
                            str(nombre),    # C: Nombre
                            str(a_pat),     # D: Apellido P
                            str(a_mat),     # E: Apellido M
                            int(h*w),       # F: Píxeles Totales
                            pix_tumor,      # G: Píxeles Tumor
                            round(porcentaje, 4), # H: %
                            url_imagen      # I: Link Imagen
                        ])

                    # Guardamos resultados en sesión
                    st.session_state.res_img = res_img
                    st.session_state.datos_reporte = {
                        "paciente": f"{nombre} {a_pat} {a_mat}",
                        "expediente": expediente,
                        "totales": f"{h*w:,}",
                        "tumor": f"{pix_tumor:,}",
                        "porcentaje": porcentaje,
                        "url": url_imagen,
                        "fecha": now_str
                    }
                    st.session_state.analizado = True
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error en el protocolo: {e}")

# --- PANTALLA DE RESULTADOS ---
if st.session_state.analizado and st.session_state.datos_reporte:
    st.image(st.session_state.res_img, use_container_width=True)
    
    d = st.session_state.datos_reporte
    
    st.markdown(f"""
    <div class="report-container">
        <div class="report-header">INFORME TÉCNICO DE SEGMENTACIÓN RADIOLÓGICA</div>
        <div style="display: flex; justify-content: space-between;">
            <div style="width: 45%;">
                <p class="data-label">Paciente</p><p class="data-value">{d['paciente']}</p>
                <p class="data-label">Resolución Total</p><p class="data-value">{d['totales']} px</p>
            </div>
            <div style="width: 45%; text-align: right;">
                <p class="data-label">Expediente</p><p class="data-value">{d['expediente']}</p>
                <p class="data-label">Detección Tumoral</p><p class="data-value" style="font-weight:700;">{d['tumor']} px</p>
            </div>
        </div>
        <div class="result-box">
            <p class="data-label">Proporción de Ocupación Tumoral</p>
            <h1 style="color: #2c3e50; margin:10px 0; font-size: 58px; font-family: 'Libre Baskerville', serif;">{d['porcentaje']:.4f} %</h1>
        </div>
        <p style="color: #95a5a6; font-size: 13px; margin-top: 30px; text-align: center; border-top: 1px solid #eee; padding-top: 10px;">
            Este documento digital tiene validez de registro clínico. Sincronizado el {d['fecha']}. 
            <br><a href="{d['url']}" target="_blank">Ver Evidencia en Servidor</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("REALIZAR NUEVO ANÁLISIS"):
        st.session_state.analizado = False
        st.session_state.datos_reporte = None
        st.rerun()
