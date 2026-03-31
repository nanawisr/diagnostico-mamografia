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
API_KEY_ROBOFLOW = "nOMi9VHi25eRhP420XFn"
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/3" 
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"
EXCEL_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"

st.set_page_config(page_title="Plataforma Médica Digital", layout="wide")

if 'analizado' not in st.session_state:
    st.session_state.analizado = False

# --- ESTILOS CSS (DISEÑO INSTITUCIONAL) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Serif+Pro:wght@400;600&display=swap');
    html, body, [class*="st-"] { font-family: 'Source Serif Pro', serif; }
    .stButton > button { width: 100%; border-radius: 2px; height: 3.5em; font-weight: 600; background-color: #2c3e50 !important; color: white !important; margin-top: 50px; text-transform: uppercase; letter-spacing: 2px;}
    .header-box { background-color: #f8f9fa; padding: 30px; border-bottom: 3px solid #2c3e50; text-align: center; margin-bottom: 30px; }
    .header-box h1 { color: #2c3e50; font-family: 'Libre Baskerville', serif; font-size: 32px; }
    .report-container { border: 2px solid #2c3e50; padding: 40px; background-color: #ffffff; box-shadow: 5px 5px 15px rgba(0,0,0,0.05); }
    .result-box { background-color: #fdfdfd; text-align: center; border: 1px solid #dcdde1; padding: 30px; margin-top: 20px; }
</style>
<div class="header-box">
    <h1>PLATAFORMA DE DIAGNÓSTICO DIGITAL</h1>
    <p style="color: #7f8c8d; letter-spacing: 1px;">SISTEMA DE ANÁLISIS RADIOLÓGICO</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.analizado:
    exp = st.text_input("No. Expediente:", value="00478119")
    col1, col2, col3 = st.columns(3)
    nom = col1.text_input("Nombre(s):", value="Ana")
    ap = col2.text_input("Ap. Paterno:", value="Reyes")
    am = col3.text_input("Ap. Materno:", value="Morales")
    uploader = st.file_uploader("Cargar Placa Mamográfica", type=["jpg", "png", "jpeg"])

    if st.button("INICIAR PROCESAMIENTO"):
        if uploader:
            with st.spinner("Ejecutando algoritmos de segmentación..."):
                try:
                    raw_bytes = uploader.read()
                    img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                    h, w, _ = img.shape
                    
                    _, buffer = cv2.imencode('.jpg', img)
                    img_64 = base64.b64encode(buffer).decode('utf-8')
                    res = requests.post(f"https://outline.roboflow.com/{ENDPOINT_ROBOFLOW}?api_key={API_KEY_ROBOFLOW}",
                                        data=img_64, headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    if "predictions" in res:
                        for p in res['predictions']:
                            if p['class'] == 'tumor':
                                pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                                cv2.fillPoly(mask, [pts], 255)
                    
                    pix_tumor = int(np.count_nonzero(mask))
                    porc = (pix_tumor / (h * w)) * 100
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    res_img = img_rgb.copy()
                    res_img[mask > 0] = [255, 0, 0]

                    _, enc = cv2.imencode('.jpg', cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                    img_url = requests.post("https://api.imgbb.com/1/upload", 
                                            data={"key": st.secrets["API_KEY_IMGBB"], "image": base64.b64encode(enc).decode('utf-8')}).json()["data"]["url"]

                    if "GCP_JSON_BASE64" in st.secrets:
                        decoded_data = base64.b64decode(st.secrets["GCP_JSON_BASE64"]).decode("utf-8")
                        info = json.loads(decoded_data)
                        info["private_key"] = info["private_key"].replace("\\n", "\n")
                        
                        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
                        gc = gspread.authorize(creds)
                        sheet = gc.open_by_key(SHEET_ID).sheet1
                        
                        num_fila = len(sheet.get_all_values()) + 1
                        now = datetime.now().strftime("%Y-%m-%d %H:%M")
                        nueva_fila = [now, exp, nom, ap, am, h*w, pix_tumor, round(porc, 4), img_url]
                        sheet.insert_row(nueva_fila, num_fila)

                    st.session_state.res_img = res_img
                    st.session_state.dat = {"p": f"{nom} {ap} {am}", "e": exp, "t": f"{h*w:,}", "tm": f"{pix_tumor:,}", "pc": porc, "u": img_url, "f": now}
                    st.session_state.analizado = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Falla en el procesamiento: {e}")

if st.session_state.analizado:
    d = st.session_state.dat
    # Título dinámico con datos del paciente
    st.subheader(f"Resultado de Análisis: {d['p']} | Expediente: {d['e']}")
    
    # Mostrar resultado visual con nuevo pie de foto
    st.image(st.session_state.res_img, use_container_width=True, caption="Placa con segmentación detectada")
    
    st.markdown(f"""
    <div class="report-container">
        <h2 style="text-align:center; font-family: 'Libre Baskerville', serif;">INFORME TÉCNICO RADIOLÓGICO</h2>
        <div style="display: flex; justify-content: space-between;">
            <div><p><b>Paciente:</b> {d['p']}</p><p><b>No. Expediente:</b> {d['e']}</p></div>
            <div style="text-align: right;"><p><b>Resolución Total:</b> {d['t']} px</p><p><b>Área Tumoral:</b> {d['tm']} px</p></div>
        </div>
        <div class="result-box">
            <p style="color: #7f8c8d; font-size: 14px; text-transform: uppercase;">Porcentaje de Ocupación sobre Lienzo</p>
            <h1 style="color: #2c3e50; font-size: 60px;">{d['pc']:.4f} %</h1>
        </div>
        <p style="text-align:center; color:#2c3e50; margin-top:20px; font-weight:600;">REGISTRO SINCRONIZADO EN LA NUBE</p>
        <p style="text-align:center;">
            <a href="{EXCEL_URL}" target="_blank" style="text-decoration:none; color:#2980b9;">[ VER BASE DE DATOS ]</a>
            <a href="{d['u']}" target="_blank" style="margin-left:20px; text-decoration:none; color:#2980b9;">[ EVIDENCIA DIGITAL ]</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("ANALIZAR NUEVA PACIENTE"):
        st.session_state.analizado = False
        st.rerun()
