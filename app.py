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
SHEET_ID = "1sdmCsIJmRz84Fu26KtTrE_rTTh7SzoS5womeVctnXQ4"
DRIVE_FOLDER_ID = "1S66F3LwaWazDogCbcU8kGJH91vKTuyHb"

st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# ... (Mantén tus estilos CSS y formulario igual que antes) ...

# --- PARTE DE LA SINCRONIZACIÓN (REEMPLAZA ESTA SECCIÓN) ---
if "service_account_base64" in st.secrets:
    try:
        b64_str = st.secrets["service_account_base64"].strip()
        missing_padding = len(b64_str) % 4
        if missing_padding: b64_str += '=' * (4 - missing_padding)
        
        info = json.loads(base64.b64decode(b64_str))
        info["private_key"] = info["private_key"].replace("\\n", "\n")
        
        creds = Credentials.from_service_account_info(info, 
                scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
        
        # Drive: Subida con soporte para almacenamiento compartido
        ds = build('drive', 'v3', credentials=creds)
        cv2.imwrite(file_name, cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
        
        media = MediaFileUpload(file_name, mimetype='image/jpeg')
        file_metadata = {
            'name': file_name,
            'parents': [DRIVE_FOLDER_ID]
        }
        
        # EL CAMBIO MÁGICO: supportsAllDrives=True y un request de subida corregido
        df = ds.files().create(
            body=file_metadata, 
            media_body=media, 
            fields='id',
            supportsAllDrives=True # <-- Permite usar el espacio de la carpeta compartida
        ).execute()
        
        drive_id = df.get('id')
        if os.path.exists(file_name): os.remove(file_name)

        # Sheets
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(SHEET_ID).sheet1
        sh.append_row([now_str, tipo_reg, expediente, nombre, a_pat, a_mat, h*w, pix_tumor, round(porcentaje, 4), drive_id])
        st.toast("✅ Sincronización Exitosa")
        
    except Exception as e_cloud:
        # Si sigue fallando por cuota, mostramos un mensaje más amigable
        if "storageQuotaExceeded" in str(e_cloud):
            st.error("⚠️ Error de Cuota: Google no permite que la cuenta de servicio use tu espacio aún. Prueba moviendo el Google Sheet a una carpeta diferente y vuelve a compartirla con el correo de la cuenta de servicio.")
        else:
            st.error(f"Error: {str(e_cloud)}")
