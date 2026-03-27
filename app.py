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

# --- CONFIGURACIÓN ---
API_KEY_ROBOFLOW = "nOMi9VHi25eRhP420XFn"
ENDPOINT_ROBOFLOW = "segmentacion-tumores-mamografia-sn1wk/5"

# Identificadores de Google (Asegúrate de que estos IDs sean los correctos)
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"
# ID de la carpeta 'Pacientes_Mamografía'
DRIVE_FOLDER_ID = "1S66F3LwaWazDogCbcU8kGJH91vKTuyHb" 
MI_CORREO = "nana.wowrara@gmail.com"

st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# --- DISEÑO ---
st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #1e88e5 !important; color: white !important; }
    .header-box { background-color: #34495e; padding: 25px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; color: white; }
</style>
<div class="header-box">
    <h1 style='margin:0;'>Diagnóstico: Carpeta Pacientes_Mamografía</h1>
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

if st.button("Ejecutar Análisis y Guardar en Carpeta"):
    if not uploader:
        st.warning("⚠️ Cargue una imagen para procesar.")
    else:
        with st.spinner("🔬 Procesando y guardando imagen en Drive..."):
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
                    
                    st.image(res_img, use_container_width=True, caption="Imagen Procesada para Guardado")

                    # 2. GUARDADO PRIORITARIO EN DRIVE
                    drive_id = "No guardado"
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if "service_account_base64" in st.secrets:
                        try:
                            b64_str = st.secrets["service_account_base64"].strip()
                            b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
                            info = json.loads(base64.b64decode(b64_str))
                            info["private_key"] = info["private_key"].replace("\\n", "\n")
                            creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"])
                            
                            # DRIVE: El objetivo principal
                            ds = build('drive', 'v3', credentials=creds)
                            file_name = f"Analisis_{expediente}_{datetime.now().strftime('%H%M%S')}.jpg"
                            cv2.imwrite(file_name, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                            
                            media = MediaFileUpload(file_name, mimetype='image/jpeg', resumable=False)
                            file_metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
                            
                            file_drive = ds.files().create(
                                body=file_metadata, 
                                media_body=media, 
                                fields='id',
                                supportsAllDrives=True
                            ).execute()
                            
                            drive_id = file_drive.get('id')
                            
                            # Compartir para asegurar visibilidad en tu unidad
                            ds.permissions().create(fileId=drive_id, body={'type': 'user', 'role': 'writer', 'emailAddress': MI_CORREO}).execute()
                            
                            if os.path.exists(file_name): os.remove(file_name)
                            st.success(f"✅ Imagen guardada exitosamente en la carpeta. ID: {drive_id}")

                            # 3. REGISTRO EN EXCEL (COMO RESPALDO)
                            gc = gspread.authorize(creds)
                            sh = gc.open_by_key(SHEET_ID).sheet1
                            sh.append_row([
                                now_str, str(tipo_reg), str(expediente), 
                                str(nombre), str(a_pat), str(a_mat), 
                                int(h*w), pix_tumor, round(porcentaje, 4), drive_id
                            ])
                            
                        except Exception as e:
                            st.error(f"Error al intentar guardar en Drive: {e}")
                            st.info("Nota: Revisa si la cuenta de servicio tiene permisos de editor en la carpeta destino.")

                else:
                    st.info("Análisis completado: No se detectaron anomalías.")
            except Exception as e:
                st.error(f"Error técnico general: {e}")

if st.button("Limpiar Pantalla"):
    st.rerun()
