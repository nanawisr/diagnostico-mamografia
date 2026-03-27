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

# --- CONFIGURACIÓN MAESTRA ---
API_KEY_ROBOFLOW = "nOMi9VHi25eRhP420XFn"
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/5"
SHEET_NAME = "Base_Datos_Pacientes"

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# --- ESTILOS CSS ---
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
    
    .footer-text { color: #95a5a6; font-size: 13px; margin-top: 15px; line-height: 1.6; }
</style>

<div class="header-box">
    <h1>Plataforma de Diagnóstico Digital</h1>
    <p>MÓDULO DE ANÁLISIS CLÍNICO AVANZADO</p>
</div>
""", unsafe_allow_html=True)

st.info("**Gestión Hospitalaria:** Ingrese la filiación completa de la paciente y cargue el estudio para su procesamiento y registro.")

# --- FORMULARIO DE FILIACIÓN ---
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
        st.warning("⚠️ Por favor, cargue una imagen radiográfica.")
    else:
        with st.spinner("🔬 Procesando imagen y sincronizando datos..."):
            try:
                # 1. PROCESAMIENTO DE IMAGEN E IA
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                pix_totales = h * w
                
                _, buffer = cv2.imencode('.jpg', img)
                img_64 = base64.b64encode(buffer).decode('utf-8')
                
                url_rf = f"https://outline.roboflow.com/{ENDPOINT_ROBOFLOW}"
                params = {"api_key": API_KEY_ROBOFLOW, "confidence": "40"}
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                
                response = requests.post(url_rf, params=params, data=img_64, headers=headers)
                prediction = response.json()
                
                if "predictions" in prediction:
                    preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for p in preds:
                        pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    
                    pix_tumor = np.count_nonzero(mask)
                    porcentaje = (pix_tumor / pix_totales) * 100
                    
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    overlay = img_rgb.copy()
                    overlay[mask > 0] = [255, 0, 0]
                    res_img = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
                    
                    st.write(f"### Análisis de Imagen - {nombre} {a_pat} {a_mat}")
                    st.image(res_img, use_container_width=True)

                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    file_name = f"Analisis_{nombre}_{a_pat}_{expediente}.jpg"
                    drive_id = "No sincronizado"

                    # 2. SINCRONIZACIÓN CON GOOGLE
                    try:
                        if "google_drive_credentials" in st.secrets:
                            # CORRECCIÓN AQUÍ: Limpieza de la estructura de diccionarios
                            creds_dict = dict(st.secrets["google_drive_credentials"])
                            if "private_key" in creds_dict:
                                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
                            
                            creds = Credentials.from_service_account_info(
                                creds_dict,
                                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
                            )
                            
                            # Subir a Google Drive
                            ds = build('drive', 'v3', credentials=creds)
                            cv2.imwrite(file_name, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                            media = MediaFileUpload(file_name, mimetype='image/jpeg')
                            df = ds.files().create(body={'name': file_name}, media_body=media, fields='id').execute()
                            drive_id = df.get('id')
                            if os.path.exists(file_name): os.remove(file_name)

                            # Registrar en Google Sheets
                            gc = gspread.authorize(creds)
                            sh = gc.open(SHEET_NAME).sheet1
                            sh.append_row([
                                now_str, tipo_reg, expediente, nombre, a_pat, a_mat, 
                                pix_totales, pix_tumor, round(porcentaje, 4), drive_id
                            ])
                            st.toast("✅ Datos sincronizados correctamente.")
                        else:
                            st.warning("⚠️ No se encontraron las credenciales en 'Secrets'.")
                    except Exception as e_cloud:
                        st.error(f"Error de Sincronización: {str(e_cloud)}")

                    # 3. REPORTE TÉCNICO
                    st.markdown(f"""
                    <div class="report-container">
                        <div class="report-header">REPORTE TÉCNICO DE SEGMENTACIÓN</div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">PACIENTE</p><b>{nombre} {a_pat} {a_mat}</b></div>
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">EXPEDIENTE</p><b>{expediente}</b></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">PÍXELES TOTALES</p><p style="margin:0;">{pix_totales:,} px</p></div>
                            <div><p style="color: #7f8c8d; margin:0; font-size:12px;">PÍXELES TUMOR</p><p style="margin:0; color: #c23616;">{pix_tumor:,} px</p></div>
                        </div>
                        <div style="background-color: #fff5f0; text-align: center; border: 1px solid #e67e22; padding: 20px; border-radius: 10px;">
                            <p style="color: #e67e22; margin:0; font-weight: bold; font-size: 14px; text-transform: uppercase;">ÁREA DE OCUPACIÓN TUMORAL</p>
                            <h1 style="color: #c23616; margin:0; font-size: 55px;">{porcentaje:.4f} %</h1>
                        </div>
                        <div class="footer-text">
                            Sincronizado con Historial Clínico (Base de Datos Hospitalaria).<br>
                            Imagen de diagnóstico guardada exitosamente en Drive ID: <b>{drive_id}</b><br>
                            Referencia: {file_name} | Fecha: {now_str}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("❌ Error: Roboflow no devolvió predicciones.")

            except Exception as e:
                st.error(f"❌ Error Crítico: {str(e)}")

# --- BOTÓN NUEVA CONSULTA ---
st.write("---")
st.markdown('<div class="btn-nueva">', unsafe_allow_html=True)
if st.button("Limpiar y Nueva Consulta"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
