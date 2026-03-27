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
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/5"
API_KEY_IMGBB = "46f55aca9d7c7d9c64f532e4203a789c" # Tu llave integrada
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"

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
    <p style='margin:0; color:#bdc3c7;'>SISTEMA DE ANÁLISIS CLÍNICO CON RESPALDO EN NUBE (ImgBB + Sheets)</p>
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

if st.button("Ejecutar Análisis y Sincronizar"):
    if not uploader:
        st.warning("⚠️ Cargue una imagen antes de continuar.")
    else:
        with st.spinner("🔬 Procesando y subiendo a la nube..."):
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
                    st.image(res_img, use_container_width=True)

                    # 2. SUBIR IMAGEN A IMGBB (Sin límites de cuota de Google)
                    _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                    img_bb_64 = base64.b64encode(img_encoded).decode('utf-8')
                    
                    response_bb = requests.post(
                        "https://api.imgbb.com/1/upload",
                        data={"key": API_KEY_IMGBB, "image": img_bb_64}
                    )
                    url_imagen = response_bb.json()["data"]["url"]

                    # 3. GUARDAR EN GOOGLE SHEETS
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if "service_account_base64" in st.secrets:
                        b64_str = st.secrets["service_account_base64"].strip()
                        b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
                        info = json.loads(base64.b64decode(b64_str))
                        info["private_key"] = info["private_key"].replace("\\n", "\n")
                        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
                        gc = gspread.authorize(creds)
                        sh = gc.open_by_key(SHEET_ID).sheet1
                        
                        # Registro final con el link de ImgBB
                        sh.append_row([
                            now_str, str(tipo_reg), str(expediente), 
                            str(nombre), str(a_pat), str(a_mat), 
                            int(h*w), pix_tumor, round(porcentaje, 4), url_imagen
                        ])
                        st.toast("✅ Sincronizado correctamente")

                    # 4. REPORTE VISUAL
                    st.markdown(f"""
                    <div class="report-container">
                        <div style="background-color: #fff5f0; text-align: center; border: 1px solid #e67e22; padding: 25px; border-radius: 10px;">
                            <p style="color: #e67e22; margin:0; font-weight: bold; text-transform: uppercase;">ÁREA DE OCUPACIÓN TUMORAL</p>
                            <h1 style="color: #c23616; margin:0; font-size: 55px;">{porcentaje:.4f} %</h1>
                        </div>
                        <p style="color: #95a5a6; font-size: 13px; margin-top: 15px;">
                            <b>Evidencia en Nube:</b> <a href="{url_imagen}" target="_blank">Abrir Imagen Analizada</a>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No se detectaron hallazgos significativos.")
            except Exception as e:
                st.error(f"Error de sistema: {e}")

if st.button("Limpiar Pantalla"):
    st.rerun()
