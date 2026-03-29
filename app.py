import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials

# --- CONFIGURACIÓN ---
API_KEY_ROBOFLOW = "nOMi9VHi25eRhP420XFn"
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/5"
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"
EXCEL_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/edit"

st.set_page_config(page_title="Sistema Médico Digital", layout="wide")

if 'analizado' not in st.session_state:
    st.session_state.analizado = False

# --- ESTILO ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:wght@400;700&family=Source+Serif+Pro:wght@400;600&display=swap');
    html, body, [class*="st-"] { font-family: 'Source Serif Pro', serif; }
    .stButton > button { width: 100%; border-radius: 2px; height: 3.5em; font-weight: 600; background-color: #2c3e50 !important; color: white !important; margin-top: 50px; }
    .header-box { background-color: #f8f9fa; padding: 30px; border-bottom: 3px solid #2c3e50; text-align: center; margin-bottom: 30px; }
    .report-container { border: 2px solid #2c3e50; padding: 40px; background-color: #ffffff; box-shadow: 5px 5px 15px rgba(0,0,0,0.05); }
</style>
<div class="header-box">
    <h1 style="font-family: 'Libre Baskerville', serif;">PLATAFORMA DE DIAGNÓSTICO DIGITAL</h1>
</div>
""", unsafe_allow_html=True)

if not st.session_state.analizado:
    exp = st.text_input("Número de Expediente:", value="00478119")
    c1, c2, c3 = st.columns(3)
    nom = c1.text_input("Nombre(s):", value="Ana")
    ap = c2.text_input("Apellido Paterno:", value="Reyes")
    am = c3.text_input("Apellido Materno:", value="Morales")
    uploader = st.file_uploader("Seleccionar Imagen Radiográfica", type=["jpg", "png", "jpeg"])

    if st.button("INICIAR PROTOCOLO DE ANÁLISIS"):
        if uploader:
            with st.spinner("Procesando..."):
                try:
                    # 1. IA
                    img = cv2.imdecode(np.frombuffer(uploader.read(), np.uint8), 1)
                    h, w, _ = img.shape
                    _, buffer = cv2.imencode('.jpg', img)
                    img_64 = base64.b64encode(buffer).decode('utf-8')
                    res = requests.post(f"https://outline.roboflow.com/{ENDPOINT_ROBOFLOW}?api_key={API_KEY_ROBOFLOW}&confidence=40", data=img_64, headers={"Content-Type": "application/x-www-form-urlencoded"}).json()
                    
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for p in res.get('predictions', []):
                        if p['class'] == 'tumor':
                            pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                    
                    pix_tumor = int(np.count_nonzero(mask))
                    porc = (pix_tumor / (h * w)) * 100
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_rgb[mask > 0] = [255, 0, 0]

                    # 2. Link Imagen
                    _, enc = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                    url = requests.post("https://api.imgbb.com/1/upload", data={"key": st.secrets["API_KEY_IMGBB"], "image": base64.b64encode(enc).decode('utf-8')}).json()["data"]["url"]

                    # 3. Google Sheets
                    info = dict(st.secrets["gcp_service_account"])
                    info["private_key"] = info["private_key"].replace("\\n", "\n")
                    creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
                    now = datetime.now().strftime("%Y-%m-%d %H:%M")
                    gspread.authorize(creds).open_by_key(SHEET_ID).sheet1.append_row([now, exp, nom, ap, am, h*w, pix_tumor, round(porc, 4), url])

                    st.session_state.res_img = img_rgb
                    st.session_state.dat = {"p": f"{nom} {ap} {am}", "e": exp, "t": f"{h*w:,}", "tm": f"{pix_tumor:,}", "pc": porc, "u": url, "f": now}
                    st.session_state.analizado = True
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

if st.session_state.analizado:
    st.image(st.session_state.res_img, use_container_width=True)
    d = st.session_state.dat
    st.markdown(f"""
    <div class="report-container">
        <h2 style="text-align:center; font-family: 'Libre Baskerville', serif;">INFORME TÉCNICO DE SEGMENTACIÓN</h2>
        <p><b>Paciente:</b> {d['p']} | <b>Expediente:</b> {d['e']}</p>
        <p><b>Resolución:</b> {d['t']} px | <b>Detección:</b> {d['tm']} px</p>
        <h1 style="text-align:center; color:#2c3e50; font-size: 58px;">{d['pc']:.4f} %</h1>
        <p style="text-align:center; font-weight:600; color:#2c3e50; border-top: 1px solid #eee; padding-top:20px;">
            LA INFORMACIÓN HA SIDO ACTUALIZADA EXITOSAMENTE EN LA BASE DE DATOS.
        </p>
        <p style="text-align:center;">
            <a href="{EXCEL_URL}" target="_blank">[ CONSULTAR BASE DE DATOS (EXCEL) ]</a>
            <a href="{d['u']}" target="_blank" style="margin-left:20px;">[ VER EVIDENCIA DIGITAL ]</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("REALIZAR NUEVO ANÁLISIS"):
        st.session_state.analizado = False
        st.rerun()
