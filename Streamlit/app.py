import streamlit as st
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class CodificadorMultietiqueta(BaseEstimator, TransformerMixin):
    def __init__(self, classes=None):
        self.classes = classes

    def fit(self, X, y=None):
        self.codificador_ = MultiLabelBinarizer(classes=self.classes)
        self.codificador_.fit(X)
        return self

    def transform(self, X):
        return self.codificador_.transform(X)

st.set_page_config(page_title="Estimador de Valoraciones GOG", layout="centered")

@st.cache_resource
def cargar_modelo():
    return joblib.load('./modelo_gog_rf.pkl')

modelo = cargar_modelo()

st.title("Estimador de Valoraciones de Videojuegos")
st.markdown("Ingresa las características del videojuego para predecir su valoración promedio.")

with st.form("prediccion_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        developer = st.text_input("Desarrolladora (Developer)", value="CD PROJEKT RED")
        publisher = st.text_input("Publicadora (Publisher)", value="CD PROJEKT RED")
        year = st.number_input("Año de lanzamiento", min_value=1980, max_value=2025, value=2024)
        baseAmount = st.number_input("Precio Base (USD)", min_value=0.0, value=29.99, step=1.0)
        
    with col2:
        ageLimit = st.selectbox("Restricción de edad", options=[0, 12, 16, 18])
        isFree = st.checkbox("¿Es gratuito? (Free to play)")
        
        todos_los_generos = [
            'Adventure', 'Action', 'Strategy', 'Role-playing', 'Fantasy', 'Sci-fi',
            'Simulation', 'Puzzle', 'Point-and-click', 'Turn-based', 'Shooter',
            'FPP', 'Platformer', 'Horror', 'Real-time', 'Historical', 'Managerial',
            'Arcade', 'Tactical', 'Visual Novel', 'Modern', 'Narrative',
            'Detective-mystery', 'Survival', 'Roguelike', 'Combat', 'Building',
            'TPP', 'Fighting', 'Comedy', 'Mystery', 'Stealth', 'Racing',
            'Open World', 'Metroidvania', 'JRPG', 'Sandbox', 'Sports', 'Economic',
            'Card Game', 'Hidden Object', 'Rally', 'Virtual life', 'Naval',
            'Off-road', 'Pinball', 'Espionage', 'Soccer', 'Programming',
            'Educational', 'Team sport', 'Chess', 'Touring'
        ]

        genres = st.multiselect(
            "Géneros", 
            options=todos_los_generos,
            default=['Action', 'Adventure']
        )
        os_support = st.multiselect("Sistemas Operativos Soportados", 
            options=['windows', 'mac', 'linux'],
            default=['windows']
        )

    submit = st.form_submit_button("Predecir Valoración")

if submit:
    if not genres or not os_support:
        st.warning("Debes seleccionar al menos un género y un sistema operativo.")
    else:
        entrada = pd.DataFrame([{
            'developer': developer,
            'publisher': publisher,
            'year': year,
            'baseAmount': baseAmount,
            'ageLimit': ageLimit,
            'isFree': int(isFree), 
            'genres': genres,     
            'supportedOperatingSystems': os_support 
        }])
        
        try:
            prediccion = modelo.predict(entrada)[0]
            st.success(f"### Valoración Estimada: ⭐ {prediccion:.2f} / 5.00")
        except Exception as e:
            st.error(f"Error al procesar los datos: {e}")