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

# Identificadores (Carpeta Prueba)
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"
DRIVE_FOLDER_ID = "1gQchYqpMWvLBRQ8U-1Kl3yvB8j6uxmzd" 
MI_CORREO = "nana.wowrara@gmail.com"

st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# ... (CSS y Encabezado igual) ...
st.markdown("<h1 style='color: #1e88e5;'>Módulo de Análisis Clínico</h1>", unsafe_allow_html=True)

# Formulario
c1, c2 = st.columns([1, 2])
tipo_reg = c1.selectbox("Registro:", ["Nuevo", "Existente"])
expediente = c2.text_input("Expediente:", value="00478119")
c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_pat = c4.text_input("A. Paterno:", value="Reyes")
a_mat = c5.text_input("A. Materno:", value="Morales")

uploader = st.file_uploader("📤 Subir Imagen Radiográfica", type=["jpg", "png", "jpeg"])

if st.button("Ejecutar y Guardar en Drive"):
    if not uploader:
        st.warning("⚠️ Cargue una imagen.")
    else:
        with st.spinner("🔬 Sincronizando datos e imagen..."):
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

                    # 2. SINCRONIZACIÓN (NUEVA LÓGICA DE SUBIDA)
                    drive_id = "Error de Cuota"
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if "service_account_base64" in st.secrets:
                        b64_str = st.secrets["service_account_base64"].strip()
                        b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
                        info = json.loads(base64.b64decode(b64_str))
                        info["private_key"] = info["private_key"].replace("\\n", "\n")
                        creds = Credentials.from_service_account_info(info, scopes=["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"])
                        
                        # --- DRIVE: SUBIDA DIRECTA ---
                        try:
                            ds = build('drive', 'v3', credentials=creds)
                            fname = f"Analisis_{expediente}_{datetime.now().strftime('%H%M%S')}.jpg"
                            cv2.imwrite(fname, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
                            
                            # METADATOS: Intentamos que el dueño seas tú desde el inicio
                            file_metadata = {
                                'name': fname,
                                'parents': [DRIVE_FOLDER_ID],
                                'description': f'Analisis de {nombre} {a_pat}'
                            }
                            media = MediaFileUpload(fname, mimetype='image/jpeg')
                            
                            # Subida con parámetros de compatibilidad total
                            file_drive = ds.files().create(
                                body=file_metadata,
                                media_body=media,
                                fields='id',
                                supportsAllDrives=True, # Importante para carpetas compartidas
                                ignoreDefaultVisibility=True
                            ).execute()
                            
                            drive_id = file_drive.get('id')
                            
                            # COMPARTIR INMEDIATAMENTE (Esto ayuda a validar la cuota del receptor)
                            ds.permissions().create(
                                fileId=drive_id,
                                body={'type': 'user', 'role': 'owner', 'emailAddress': MI_CORREO},
                                transferOwnership=True, # Intentamos pasarte la propiedad
                                supportsAllDrives=True
                            ).execute()
                            
                            if os.path.exists(fname): os.remove(fname)
                        except Exception as de:
                            # Si falla el 'transferOwnership', intentamos permiso normal
                            drive_id = f"ID: {drive_id} (Limitado)" if 'drive_id' in locals() else f"Error: {str(de)[:30]}"

                        # --- SHEETS: SIEMPRE FUNCIONA ---
                        gc = gspread.authorize(creds)
                        sh = gc.open_by_key(SHEET_ID).sheet1
                        sh.append_row([now_str, str(tipo_reg), str(expediente), str(nombre), str(a_pat), str(a_mat), int(h*w), pix_tumor, round(porcentaje, 4), drive_id])
                        st.toast("✅ Sincronización Realizada")

                    st.success(f"Análisis finalizado: {porcentaje:.4f}% | Registro en Drive: {drive_id}")

            except Exception as e:
                st.error(f"Error: {e}")
