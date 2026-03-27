import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
import os

# --- DATOS DE CONEXIÓN MAESTRA (¡NO CAMBIAR!) ---
API_KEY = "nOMi9VHi25eRhP420XFn"
# Usamos la Versión 5 que es la que tiene el cohete en Roboflow
ENDPOINT = "segmentacion-tumores-mamografia-sn1wk/5"

# --- CONFIGURACIÓN DE PÁGINA (WIDE) ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

# --- ESTILOS CSS (Clonación exacta de tu diseño) ---
st.markdown("""
<style>
    /* Estilo para los botones */
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; }
    /* Botón Ejecutar: Azul */
    .btn-ejecutar > div > button { background-color: #3498db !important; color: white !important; border: none; }
    /* Botón Nueva Consulta: Verde */
    .btn-nueva > div > button { background-color: #27ae60 !important; color: white !important; border: none; }
    
    /* Encabezado Principal */
    .header-box { background-color: #2c3e50; padding: 20px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
</style>

<div class="header-box">
    <h1 style="color: white; margin: 0; font-family: sans-serif;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">MÓDULO DE ANÁLISIS CLÍNICO AVANZADO | Ingeniería Biomédica</p>
</div>
""", unsafe_allow_html=True)

st.info("**Gestión Hospitalaria:** Ingrese la filiación completa de la paciente y cargue el estudio para su procesamiento y registro.")

# --- FORMULARIO DE REGISTRO ---
c1, c2 = st.columns([1, 2])
tipo_reg = c1.selectbox("Registro:", ["Nuevo", "Existente"])
expediente = c2.text_input("Expediente:", value="00478119")

c3, c4, c5 = st.columns(3)
nombre = c3.text_input("Nombre(s):", value="Ana")
a_pat = c4.text_input("A. Paterno:", value="Reyes")
a_mat = c5.text_input("A. Materno:", value="Morales")

# --- SECCIÓN DE CARGA ---
uploader = st.file_uploader("📤 Subir Imagen Radiográfica (1)", type=["jpg", "png", "jpeg"])

# --- BOTÓN DE ACCIÓN ---
st.markdown('<div class="btn-ejecutar">', unsafe_allow_html=True)
ejecutar = st.button("Ejecutar Análisis Clínico")
st.markdown('</div>', unsafe_allow_html=True)

# --- PROCESAMIENTO (Motor Blindado) ---
if ejecutar:
    if not uploader:
        st.warning("⚠️ Por favor, cargue una imagen antes de ejecutar.")
    else:
        with st.spinner("🔬 Conectando con el Servidor de IA..."):
            try:
                # 1. Leer imagen localmente
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                
                # 2. Convertir a Base64 para enviarla
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 3. Petición Directa (La que NO falla)
                url = f"https://outline.roboflow.com/{ENDPOINT}"
                params = {"api_key": API_KEY, "confidence": "40"}
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                
                response = requests.post(url, params=params, data=img_base64, headers=headers)
                prediction = response.json()
                
                if "predictions" not in prediction:
                    st.error(f"Error Técnico: {prediction.get('message', 'Servidor no disponible')}")
                else:
                    preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                    
                    # Título de resultados (con tu diseño)
                    st.subheader(f"Resultados de Análisis - {nombre} {a_pat} {a_mat}")
                    
                    if not preds:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        st.success("✅ Análisis Finalizado: No se detectaron hallazgos tumorales.")
                    else:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for p in preds:
                            pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                        
                        tumor_px = np.count_nonzero(mask)
                        porcentaje = (tumor_px / (h * w)) * 100
                        
                        # Imagen con Overlay Rojo (Como en tu diseño)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        overlay = img_rgb.copy()
                        overlay[mask > 0] = [255, 0, 0] # Rojo intenso
                        st.image(cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0), use_container_width=True)

                        # --- REPORTE TÉCNICO (Clonación exacta del cuadro naranja) ---
                        st.markdown(f"""
                        <div style="border: 2px solid #3498db; padding: 20px; border-radius: 10px; background-color: #f8f9fa; font-family: sans-serif;">
                            <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; margin-top: 0; padding-bottom: 5px;">REPORTE TÉCNICO DE SEGMENTACIÓN</h2>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                                <div><p style="color: #7f8c8d; margin:0;">PACIENTE</p><b>{nombre} {a_pat} {a_mat}</b></div>
                                <div><p style="color: #7f8c8d; margin:0;">EXPEDIENTE</p><b>{expediente}</b></div>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                                <div><p style="color: #7f8c8d; margin:0;">PÍXELES TOTALES</p><b>{h*w:,} px</b></div>
                                <div><p style="color: #7f8c8d; margin:0;">PÍXELES TUMOR</p><b style="color: #c23616;">{tumor_px:,} px</b></div>
                            </div>
                            <div style="background-color: #fdf2e9; text-align: center; border: 1px solid #e67e22; padding: 15px; border-radius: 5px; margin-top: 15px;">
                                <p style="color: #e67e22; margin:0; font-weight: bold;">ÁREA DE OCUPACIÓN TUMORAL</p>
                                <h1 style="color: #c23616; margin:0; font-size: 45px;">{porcentaje:.4f} %</h1>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error crítico en el proceso: {str(e)}")

# --- SECCIÓN INFERIOR ---
st.write("---")
st.markdown('<div class="btn-nueva">', unsafe_allow_html=True)
if st.button("Nueva Consulta"):
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
