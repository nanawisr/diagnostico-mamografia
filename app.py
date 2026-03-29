import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import gspread
import json
from google.oauth2.service_account import Credentials

# --- CONFIGURACION MAESTRA ---
API_KEY_ROBOFLOW = "nOMi9VHi25eRhP420XFn"
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/5"
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"

# --- CONFIGURACION DE PAGINA ---
st.set_page_config(page_title="Plataforma de Diagnostico Digital", layout="wide")

if 'analizado' not in st.session_state:
    st.session_state.analizado = False
if 'datos_reporte' not in st.session_state:
    st.session_state.datos_reporte = None

# --- ESTILOS CSS ---
st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #1e88e5 !important; color: white !important; border: none; }
    .header-box { background-color: #34495e; padding: 25px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
    .header-box h1 { color: white; margin: 0; font-family: sans-serif; font-size: 42px; }
    .instruction-text { color: #576574; font-size: 16px; margin-bottom: 20px; font-style: italic; }
    .report-container { border: 1px solid #ced4da; padding: 20px; border-radius: 10px; background-color: white; font-family: sans-serif; margin-top: 20px; }
    .report-header { border-bottom: 2px solid #3498db; margin-bottom: 20px; padding-bottom: 10px; color: #2c3e50; font-size: 24px; text-transform: uppercase; }
    .data-label { color: #95a5a6; font-size: 12px; text-transform: uppercase; margin-bottom: 2px; font-weight: bold; }
    .data-value { color: #2c3e50; font-weight: bold; font-size: 18px; margin-bottom: 15px; }
    .pixel-value { color: #c0392b; font-weight: bold; font-size: 18px; }
</style>
<div class="header-box">
    <h1>Plataforma de Diagnostico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0; font-size: 18px; text-transform: uppercase;">Modulo de Analisis Clinico Avanzado</p>
</div>
""", unsafe_allow_html=True)

# --- FLUJO DE TRABAJO ---
if not st.session_state.analizado:
    st.markdown('<p class="instruction-text">Por favor, rellene los datos solicitados e inserte la mamografia para su analisis.</p>', unsafe_allow_html=True)
    
    expediente = st.text_input("No. de Expediente:", value="00478119")

    c1, c2, c3 = st.columns(3)
    nombre = c1.text_input("Nombre(s):", value="Ana")
    a_pat = c2.text_input("A. Paterno:", value="Reyes")
    a_mat = c3.text_input("A. Materno:", value="Morales")

    uploader = st.file_uploader("Subir Imagen Radiografica", type=["jpg", "png", "jpeg"])

    if st.button("Ejecutar Analisis Clinico"):
        if not uploader:
            st.warning("Cargue una imagen antes de continuar.")
        else:
            with st.spinner("Procesando analisis clinico..."):
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
                        
                        pix_tumor = int(np.count_nonzero(mask))
                        porcentaje = float((pix_tumor / (h * w)) * 100)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        overlay = img_rgb.copy()
                        overlay[mask > 0] = [255, 0, 0]
                        res_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

                        # 2. Subida a ImgBB
                        api_key_imgbb = st.secrets["API_KEY_IMGBB"]
                        _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                        img_bb_64 = base64.b64encode(img_encoded).decode('utf-8')
                        response_bb = requests.post("https://api.imgbb.com/1/upload", data={"key": api_key_imgbb, "image": img_bb_64})
                        url_imagen = response_bb.json()["data"]["url"]

                        # 3. Google Sheets (Conexion Directa mediante Secrets)
                        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        if "gcp_service_account" in st.secrets:
                            info = dict(st.secrets["gcp_service_account"])
                            info["private_key"] = info["private_key"].replace("\\n", "\n")
                            
                            creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
                            gc = gspread.authorize(creds)
                            sh = gc.open_by_key(SHEET_ID).sheet1
                            
                            sh.append_row([now_str, str(expediente), str(nombre), str(a_pat), str(a_mat), int(h*w), pix_tumor, round(porcentaje, 4), url_imagen])

                        # Guardar estado para el reporte
                        st.session_state.res_img = res_img
                        st.session_state.datos_reporte = {
                            "paciente": f"{nombre} {a_pat} {a_mat}",
                            "expediente": expediente,
                            "totales": f"{h*w:,} px",
                            "tumor": f"{pix_tumor:,} px",
                            "porcentaje": porcentaje,
                            "url": url_imagen,
                            "fecha": now_str
                        }
                        st.session_state.analizado = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Error tecnico: {e}")

# --- SECCION DE RESULTADOS ---
if st.session_state.analizado and st.session_state.datos_reporte:
    st.image(st.session_state.res_img, use_container_width=True)
    
    d = st.session_state.datos_reporte
    st.markdown(f"""
    <div class="report-container">
        <div class="report-header">Reporte Tecnico de Segmentacion</div>
        <div style="display: flex; justify-content: space-between;">
            <div>
                <p class="data-label">Paciente</p>
                <p class="data-value">{d['paciente']}</p>
                <p class="data-label">Pixeles Totales</p>
                <p class="data-value">{d['totales']}</p>
            </div>
            <div style="text-align: right;">
                <p class="data-label">Expediente</p>
                <p class="data-value">{d['expediente']}</p>
                <p class="data-label">Pixeles Tumor</p>
                <p class="pixel-value">{d['tumor']}</p>
            </div>
        </div>
        <div style="background-color: #fff5f0; text-align: center; border: 1px solid #e67e22; padding: 25px; border-radius: 10px; margin-top: 20px;">
            <p style="color: #e67e22; margin:0; font-weight: bold; text-transform: uppercase; font-size: 14px;">Area de Ocupacion Tumoral</p>
            <h1 style="color: #c23616; margin:0; font-size: 65px;">{d['porcentaje']:.4f} %</h1>
        </div>
        <p style="color: #95a5a6; font-size: 12px; margin-top: 15px; text-align: center;">
            Evidencia: <a href="{d['url']}" target="_blank">Ver imagen analizada</a> | Sincronizado: {d['fecha']}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")
    if st.button("Nuevo Estudio"):
        st.session_state.analizado = False
        st.session_state.datos_reporte = None
        st.rerun()

st.write("---")
