import streamlit as st
import cv2
import numpy as np
import requests
import base64

# --- DATOS DE CONEXIÓN ---
API_KEY = "nOMi9VHi25eRhP420XFn"
ENDPOINT = "segmentacion-tumores-mamografia-sn1wk/5"

# --- INTERFAZ ---
st.set_page_config(page_title="Plataforma de Diagnóstico Digital", layout="wide")

st.markdown("""
<style>
    .stButton > button { width: 100%; border-radius: 5px; height: 3em; font-weight: bold; background-color: #3498db !important; color: white !important; }
    .header-box { background-color: #2c3e50; padding: 20px; border-radius: 5px; border-left: 10px solid #3498db; margin-bottom: 20px; }
</style>
<div class="header-box">
    <h1 style="color: white; margin: 0; font-family: sans-serif;">Plataforma de Diagnóstico Digital</h1>
    <p style="color: #bdc3c7; margin: 5px 0 0 0;">Análisis Clínico Avanzado | Ingeniería Biomédica</p>
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
ejecutar = st.button("EJECUTAR ANÁLISIS CLÍNICO")

if ejecutar:
    if not uploader:
        st.warning("⚠️ Por favor, cargue una imagen.")
    else:
        with st.spinner("🔬 Analizando tejido..."):
            try:
                # 1. Leer imagen
                file_bytes = np.asarray(bytearray(uploader.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                h, w, _ = img.shape
                
                # 2. Convertir a Base64
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 3. Petición Directa con el Header que nos faltaba
                url = f"https://outline.roboflow.com/{ENDPOINT}"
                params = {"api_key": API_KEY, "confidence": "40"}
                
                # Agregamos el Content-Type para que Roboflow no se queje
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                
                response = requests.post(url, params=params, data=img_base64, headers=headers)
                prediction = response.json()
                
                if "predictions" not in prediction:
                    st.error(f"Error: {prediction.get('message', 'Error en respuesta')}")
                else:
                    preds = [p for p in prediction['predictions'] if p.get('class') == 'tumor']
                    
                    st.subheader(f"Resultados de Análisis - {nombre} {a_pat} {a_mat}")
                    
                    if not preds:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        st.success("✅ Análisis Finalizado: Sin hallazgos tumorales.")
                    else:
                        mask = np.zeros((h, w), dtype=np.uint8)
                        for p in preds:
                            pts = np.array([(int(pt['x']), int(pt['y'])) for pt in p['points']], np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                        
                        tumor_px = np.count_nonzero(mask)
                        porcentaje = (tumor_px / (h * w)) * 100
                        
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        overlay = img_rgb.copy()
                        overlay[mask > 0] = [255, 0, 0]
                        st.image(cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0), use_container_width=True)

                        st.markdown(f"""
                        <div style="border: 2px solid #3498db; padding: 20px; border-radius: 10px; background-color: #f8f9fa;">
                            <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; margin-top: 0;">REPORTE TÉCNICO</h2>
                            <div style="background-color: #fdf2e9; text-align: center; border: 1px solid #e67e22; padding: 15px; border-radius: 5px; margin-top: 15px;">
                                <p style="color: #e67e22; margin:0; font-weight: bold;">ÁREA DE OCUPACIÓN TUMORAL</p>
                                <h1 style="color: #c23616; margin:0; font-size: 45px;">{porcentaje:.4f} %</h1>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

st.write("---")
if st.button("Nueva Consulta"):
    st.rerun()
