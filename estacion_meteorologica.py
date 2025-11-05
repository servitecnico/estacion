"""
Aplicación Unificada de Estación Meteorológica
Combina gráficas temporales y rosa de vientos para análisis completo de datos meteorológicos
"""

import os
import io
import sys
import gc
from datetime import datetime
from typing import List, Optional

# Optimizaciones de memoria para PyInstaller
if hasattr(sys, '_MEIPASS'):
    # Configurar límites de memoria cuando se ejecuta desde PyInstaller
    os.environ['PYTHONOPTIMIZE'] = '2'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    # Limpiar memoria al inicio
    gc.collect()

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Intentar importar librerías opcionales
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False



# Colores para rosa de vientos
COLORES_BLUES = ['#08306b', '#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']

try:
    import plotly.express as px
    COLORES_DISPONIBLES = True
except ImportError:
    px = None
    COLORES_DISPONIBLES = False


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def try_read_csv(file_bytes: bytes, encoding: str) -> pd.DataFrame:
    """Lee CSV usando coma como delimitador fijo."""
    return pd.read_csv(io.BytesIO(file_bytes), sep=",", encoding=encoding)


def try_read_any(file_bytes: bytes, encoding: str, filename: Optional[str] = None) -> pd.DataFrame:
    """Lee CSV o Excel dependiendo de la extensión; intenta ambos si es necesario."""
    name_lower = (filename or "").lower()
    buffer = io.BytesIO(file_bytes)
    if name_lower.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(buffer, engine="openpyxl")
        except Exception:
            buffer.seek(0)
            return pd.read_excel(buffer)
    # Intentar CSV primero
    try:
        buffer.seek(0)
        return pd.read_csv(buffer, sep=",", encoding=encoding)
    except Exception:
        # Intentar Excel como respaldo
        buffer.seek(0)
        try:
            return pd.read_excel(buffer, engine="openpyxl")
        except Exception:
            buffer.seek(0)
            return pd.read_excel(buffer)


def download_csv_from_url(url: str, encoding: str = "utf-8-sig") -> Optional[pd.DataFrame]:
    """Descarga y lee un CSV desde una URL web."""
    if not REQUESTS_AVAILABLE:
        return None
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        file_bytes = response.content
        df = try_read_csv(file_bytes, encoding)
        return df
    except Exception as e:  # noqa: BLE001
        return None


def normalize_datetime_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Normaliza una columna de fecha a tipo datetime."""
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def get_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Obtiene todas las columnas numéricas."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def get_column_unit(column_name: str) -> str:
    """Obtiene la unidad de medida basándose en el nombre de la columna."""
    col_lower = column_name.lower().strip()
    
    if "uv index" in col_lower:
        return ''
    
    unit_map = {
        'air temperature': '°C', 'temperature': '°C', 'temp': '°C',
        'air humidity': '%RH', 'humidity': '%RH',
        'air pressure': 'hPa', 'pressure': 'hPa',
        'soil humidity': '%RH', 'soil moisture': '%RH',
        'wind direction': '°', 'direction': '°',
        'uv radiation': 'W/m²', 'uv': 'W/m²',
        'wind speed': 'm/s', 'speed': 'm/s',
        'soil temperature': '°C', 'soil temp': '°C',
        'pyranometer': 'W/m²',
        'co2': 'ppm', 'carbon dioxide': 'ppm',
        'evaporation': 'mm',
        'ph': '',
        'ec': 'mS/cm', 'electrical conductivity': 'mS/cm',
        'salinity': 'ppm',
    }
    
    for key, unit in unit_map.items():
        if key in col_lower:
            return unit
    
    return ''


def format_column_label_with_unit(column_name: str) -> str:
    """Formatea el nombre de la columna agregando su unidad si existe."""
    unit = get_column_unit(column_name)
    if unit:
        return f"{column_name} ({unit})"
    return column_name


def export_dataframe(
    df: pd.DataFrame,
    target_dir: str,
    filename: str = "datos_filtrados.csv",
    file_format: str = "csv",
) -> str:
    """Exporta un DataFrame a archivo."""
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, filename)
    if file_format == "csv":
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
    elif file_format == "xlsx":
        df.to_excel(out_path, index=False, engine="openpyxl")
    else:
        raise ValueError("Formato no soportado")
    return out_path


    


# ============================================================================
# FUNCIONES DE ROSA DE VIENTOS
# ============================================================================

def crear_rosa_vientos(
    direcciones: pd.Series,
    velocidades: pd.Series = None,
    numero_sectores: int = 16,
    titulo: str = "Rosa de Vientos",
    mostrar_velocidades: bool = True
) -> go.Figure:
    """
    Crea una gráfica de rosa de vientos (wind rose).
    
    Parámetros:
    -----------
    direcciones : pd.Series
        Serie con direcciones del viento en grados (0-360°)
    velocidades : pd.Series, opcional
        Serie con velocidades del viento (m/s)
    numero_sectores : int
        Número de sectores direccionales (8, 16, 32)
    titulo : str
        Título de la gráfica
    mostrar_velocidades : bool
        Si True, colorea por velocidades; si False, solo muestra frecuencia
        
    Retorna:
    --------
    fig : plotly.graph_objects.Figure
        Figura de Plotly con la rosa de vientos
    """
    # Limpiar datos
    df = pd.DataFrame({
        'direccion': direcciones,
        'velocidad': velocidades if velocidades is not None else pd.Series([1] * len(direcciones))
    })
    df = df.dropna(subset=['direccion'])
    
    if df.empty:
        raise ValueError("No hay datos válidos de dirección del viento")
    
    # Asegurar que las direcciones estén en el rango 0-360
    df['direccion'] = df['direccion'] % 360
    
    # Si no hay velocidades, usar valor constante
    if velocidades is None or not mostrar_velocidades:
        df['velocidad'] = 1
        mostrar_velocidades = False
    
    # Definir rangos de velocidad para colorear
    if mostrar_velocidades:
        velocidades_validas = df['velocidad'].dropna()
        if not velocidades_validas.empty:
            v_min = velocidades_validas.min()
            v_max = velocidades_validas.max()
            
            if v_max > v_min:
                bins = np.linspace(v_min, v_max, 6)
                labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f} m/s" for i in range(len(bins)-1)]
                df['categoria_velocidad'] = pd.cut(df['velocidad'], bins=bins, labels=labels, include_lowest=True)
            else:
                df['categoria_velocidad'] = f"{v_min:.1f} m/s"
        else:
            mostrar_velocidades = False
            df['velocidad'] = 1
    
    # Crear sectores direccionales
    angulo_sector = 360 / numero_sectores
    sectores = []
    for i in range(numero_sectores):
        inicio = i * angulo_sector
        fin = (i + 1) * angulo_sector
        sectores.append({
            'inicio': inicio,
            'fin': fin,
            'centro': (inicio + fin) / 2
        })
    
    # Contar frecuencias por sector y categoría de velocidad
    datos_grafica = []
    
    if mostrar_velocidades:
        categorias = sorted(df['categoria_velocidad'].unique())
        if COLORES_DISPONIBLES and px is not None:
            colores = px.colors.sequential.Blues_r[:len(categorias)] if len(categorias) <= 9 else px.colors.sequential.Blues_r
        else:
            num_colores = len(categorias)
            colores = COLORES_BLUES[:num_colores] if num_colores <= len(COLORES_BLUES) else COLORES_BLUES
        
        for categoria in categorias:
            df_cat = df[df['categoria_velocidad'] == categoria]
            for sector in sectores:
                mask = (df_cat['direccion'] >= sector['inicio']) & (df_cat['direccion'] < sector['fin'])
                if sector['fin'] == 360:
                    mask = mask | (df_cat['direccion'] == 360)
                
                frecuencia = mask.sum()
                if frecuencia > 0:
                    datos_grafica.append({
                        'sector': sector['centro'],
                        'frecuencia': frecuencia,
                        'categoria': categoria,
                        'color': colores[list(categorias).index(categoria)]
                    })
    else:
        for sector in sectores:
            mask = (df['direccion'] >= sector['inicio']) & (df['direccion'] < sector['fin'])
            if sector['fin'] == 360:
                mask = mask | (df['direccion'] == 360)
            
            frecuencia = mask.sum()
            if frecuencia > 0:
                datos_grafica.append({
                    'sector': sector['centro'],
                    'frecuencia': frecuencia,
                    'categoria': 'Todos',
                    'color': '#1f77b4'
                })
    
    # Crear la figura polar
    fig = go.Figure()
    
    if mostrar_velocidades:
        categorias = sorted(df['categoria_velocidad'].unique())
        for categoria in categorias:
            datos_cat = [d for d in datos_grafica if d['categoria'] == categoria]
            if datos_cat:
                angulos = [d['sector'] for d in datos_cat]
                frecuencias = [d['frecuencia'] for d in datos_cat]
                nombre_categoria = str(categoria) if "m/s" in str(categoria) else f"{categoria} m/s"
                
                fig.add_trace(go.Barpolar(
                    r=frecuencias,
                    theta=angulos,
                    name=nombre_categoria,
                    marker_color=datos_cat[0]['color'],
                    marker_line_color='white',
                    marker_line_width=1,
                    hovertemplate='<b>Dirección:</b> %{theta}°<br>' +
                                  '<b>Frecuencia:</b> %{r}<br>' +
                                  '<b>Velocidad:</b> ' + nombre_categoria + '<extra></extra>',
                ))
    else:
        angulos = [d['sector'] for d in datos_grafica]
        frecuencias = [d['frecuencia'] for d in datos_grafica]
        
        fig.add_trace(go.Barpolar(
            r=frecuencias,
            theta=angulos,
            name='Frecuencia',
            marker_color='#1f77b4',
            marker_line_color='white',
            marker_line_width=1,
            hovertemplate='<b>Dirección:</b> %{theta}°<br>' +
                          '<b>Frecuencia:</b> %{r}<extra></extra>',
        ))
    
    # Configurar el layout polar
    fig.update_layout(
        title={
            'text': titulo,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        font_size=12,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.1
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max([d['frecuencia'] for d in datos_grafica]) * 1.1] if datos_grafica else [0, 1],
                tickfont_size=10,
                showticklabels=True,
                tickangle=0
            ),
            angularaxis=dict(
                tickmode='array',
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'],
                direction='clockwise',
                rotation=90,
                tickfont_size=12
            )
        ),
        showlegend=mostrar_velocidades,
        height=600,
        width=700
    )
    
    return fig


# ============================================================================
# APLICACIÓN PRINCIPAL
# ============================================================================

def main() -> None:
    """Función principal de la aplicación."""
    st.set_page_config(
        page_title="Estación Meteorológica Universidad Huamanga - Análisis Completo",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌤️ Estación Meteorológica Universidad Huamanga - Análisis Completo")
    st.markdown("**Visualización de datos meteorológicos con gráficas temporales y rosa de vientos**")
    
    # Inicializar estado de sesión
    if "df_from_url" not in st.session_state:
        st.session_state["df_from_url"] = None
    
    # Sidebar - Carga de archivos
    with st.sidebar:
        st.header("📁 Carga de Datos")
        
        # Descarga desde URL
        st.subheader("🌐 Descargar desde URL")
        url_input = st.text_input(
            "URL del CSV",
            value=st.session_state.get("csv_url", ""),
            placeholder="https://cloud.rikacloud.com/data/2042",
            label_visibility="collapsed"
        )
        
        if st.button("⬇️ Descargar desde URL", disabled=not REQUESTS_AVAILABLE, use_container_width=True):
            if url_input:
                with st.spinner("Descargando..."):
                    try:
                        df_from_url = download_csv_from_url(url_input, encoding="utf-8-sig")
                        if df_from_url is not None and not df_from_url.empty:
                            st.session_state["csv_url"] = url_input
                            st.session_state["df_from_url"] = df_from_url
                            st.success(f"✅ {len(df_from_url)} filas")
                            st.rerun()
                        else:
                            st.error("❌ No se pudo descargar")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Ingresa una URL")
        
        if not REQUESTS_AVAILABLE:
            st.caption("⚠️ Instala 'requests' para descargar desde URL")
        
        st.divider()
        
        # Selección de archivo local
        url_data = st.session_state.get("df_from_url")
        
        if url_data is not None:
            st.success(f"✅ Datos desde URL")
            if st.button("🔄 Cambiar fuente", use_container_width=True):
                st.session_state["df_from_url"] = None
                st.session_state["csv_url"] = ""
                st.rerun()
        
        
        uploaded = st.file_uploader("O sube archivo aquí", type=["csv", "xlsx", "xls"])
        
        encoding = st.selectbox(
            "Codificación",
            options=["utf-8-sig", "utf-8", "latin-1", "cp1252"],
            index=0,
            help="Si ves caracteres raros, cambia la codificación"
        )
        
        st.divider()
        
        # Configuración de rosa de vientos
        st.header("⚙️ Configuración Rosa de Vientos")
        numero_sectores = st.selectbox(
            "Número de sectores",
            options=[8, 16, 32],
            index=1,
            help="Mayor número = mayor resolución direccional"
        )
        
        colorear_por_velocidad = st.checkbox(
            "Colorear por velocidad",
            value=True,
            help="Mostrar diferentes colores según la velocidad del viento"
        )
    
    # Cargar datos
    df: Optional[pd.DataFrame] = None
    file_source_desc = None
    
    url_data = st.session_state.get("df_from_url")
    if url_data is not None:
        df = url_data
        file_source_desc = f"URL: {st.session_state.get('csv_url', 'N/A')}"
    elif uploaded is not None:
        try:
            file_bytes = uploaded.read()
            df = try_read_any(file_bytes, encoding, filename=uploaded.name)
            file_source_desc = uploaded.name
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo leer el CSV: {e}")
            df = None
    
    
    if df is None:
        st.info("👆 Sube un archivo CSV o Excel (o usa URL) para comenzar")
        return
    
    # Validar columnas
    if len(df.columns) > 20:
        st.warning(f"⚠️ El archivo tiene {len(df.columns)} columnas. Se procesarán las primeras 20.")
        df = df.iloc[:, :20]
    
    st.success(f"✅ Archivo cargado: {file_source_desc} ({len(df.columns)} columnas, {len(df)} filas)")
    
    # Detectar columna de fecha
    date_col = df.columns[0]
    st.info(f"📅 Columna de fecha: **{date_col}**")
    
    # Convertir fecha
    df_dt = normalize_datetime_column(df, date_col)
    valid_dates = df_dt[date_col].dropna()
    
    if valid_dates.empty:
        st.error(f"❌ No se pudieron convertir las fechas en '{date_col}'. Verifica el formato.")
        st.dataframe(df.head(10))
        st.stop()
    
    min_date = valid_dates.min().date()
    max_date = valid_dates.max().date()
    
    # Filtros de fecha
    st.subheader("📅 Filtros de Fecha")
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Fecha inicio", value=min_date, min_value=min_date, max_value=max_date)
    with col_b:
        end_date = st.date_input("Fecha fin", value=max_date, min_value=min_date, max_value=max_date)
    
    if start_date > end_date:
        st.error("❌ La fecha de inicio no puede ser mayor que la fecha fin.")
        st.stop()
    
    mask = (df_dt[date_col] >= pd.to_datetime(start_date)) & (df_dt[date_col] <= pd.to_datetime(end_date))
    df_filtered = df_dt.loc[mask].copy()
    
    st.success(f"✅ Filas después de filtrar: {len(df_filtered):,} de {len(df):,} totales")
    
    # Crear índice UV si existe
    try:
        uv_col = next((c for c in df_filtered.columns if "uv radiation" in c.lower()), None)
        if uv_col is not None:
            uv_index_series = (df_filtered[uv_col].astype(float) / 200.0 * 15.0).round()
            uv_index_series = uv_index_series.clip(lower=0)
            df_filtered["UV index"] = uv_index_series.astype("Int64")
    except Exception:
        pass
    
    # Crear pestañas para las diferentes visualizaciones
    tab1, tab2 = st.tabs(["📈 Gráficas Temporales", "🌹 Rosa de Vientos"])
    
    # ========================================================================
    # PESTAÑA 1: GRÁFICAS TEMPORALES
    # ========================================================================
    with tab1:
        st.subheader("📈 Gráficas Temporales - Estación Meteorológica")
        
        numeric_cols = [col for col in df_filtered.columns[1:] if pd.api.types.is_numeric_dtype(df_filtered[col])]
        
        if not numeric_cols:
            st.warning("⚠️ No se encontraron columnas numéricas para graficar.")
            st.dataframe(df_filtered.head(10))
        else:
            st.caption(f"Se graficarán automáticamente {len(numeric_cols)} variables meteorológicas")
            
            plot_df = df_filtered[[date_col] + numeric_cols].copy()
            long_df = plot_df.melt(id_vars=[date_col], value_vars=numeric_cols, var_name="Serie", value_name="Valor")
            long_df['Serie_con_unidad'] = long_df['Serie'].apply(format_column_label_with_unit)
            
            fig = px.line(
                long_df,
                x=date_col,
                y="Valor",
                color="Serie_con_unidad",
                title=f"📊 Datos de Estación Meteorológica - {len(numeric_cols)} variables",
                labels={
                    date_col: "Fecha",
                    "Valor": "Valor",
                    "Serie_con_unidad": "Variable"
                }
            )
            fig.update_layout(
                legend_title_text="Variables Meteorológicas",
                height=600,
                hovermode='x unified',
                xaxis_title="Fecha",
                yaxis_title="Valor",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PESTAÑA 2: ROSA DE VIENTOS
    # ========================================================================
    with tab2:
        st.subheader("🌹 Rosa de Vientos - Dirección del Viento")
        
        # Buscar columnas de dirección y velocidad automáticamente
        posibles_direcciones = []
        posibles_velocidades = []
        
        for col in df_filtered.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['direccion', 'direction', 'wind direction', 'wind_dir']):
                posibles_direcciones.append(col)
            if any(term in col_lower for term in ['velocidad', 'speed', 'wind speed', 'wind_speed']):
                posibles_velocidades.append(col)
        
        # Si no se encuentran automáticamente, usar todas las columnas numéricas (excepto fecha)
        if not posibles_direcciones:
            posibles_direcciones = [c for c in df_filtered.columns[1:] if pd.api.types.is_numeric_dtype(df_filtered[c])]
        if not posibles_velocidades:
            posibles_velocidades = [c for c in df_filtered.columns[1:] if pd.api.types.is_numeric_dtype(df_filtered[c])]
        
        col1, col2 = st.columns(2)
        
        with col1:
            columna_direccion = st.selectbox(
                "Columna de Dirección del Viento (°)",
                options=posibles_direcciones,
                help="Selecciona la columna con dirección del viento (0-360°)"
            )
        
        with col2:
            usar_velocidad = st.checkbox("Usar velocidad del viento", value=len(posibles_velocidades) > 0)
            if usar_velocidad:
                columna_velocidad = st.selectbox(
                    "Columna de Velocidad del Viento (m/s)",
                    options=posibles_velocidades,
                    help="Selecciona la columna con velocidad del viento en m/s"
                )
            else:
                columna_velocidad = None
        
        # Validar y procesar datos
        direcciones = df_filtered[columna_direccion].dropna().copy()
        
        if direcciones.empty:
            st.error("❌ No hay datos válidos de dirección del viento")
        else:
            # Aplicar rotación de 180° automáticamente
            # Rotar 180°: (dirección - 180) % 360
            # Esto maneja automáticamente los valores negativos sumándoles 360°
            direcciones = (direcciones - 180) % 360
            st.info(f"🔄 Direcciones rotadas automáticamente 180°. Ejemplo: 90° → {(90 - 180) % 360}°, 270° → {(270 - 180) % 360}°")
            
            velocidades = None
            if usar_velocidad and columna_velocidad:
                velocidades = df_filtered[columna_velocidad]
            
            # Estadísticas
            st.subheader("📊 Estadísticas")
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("Registros válidos", len(direcciones))
            with col4:
                st.metric("Dirección promedio", f"{direcciones.mean():.1f}°")
            with col5:
                if velocidades is not None:
                    velocidades_validas = velocidades.dropna()
                    if not velocidades_validas.empty:
                        st.metric("Velocidad promedio", f"{velocidades_validas.mean():.2f} m/s")
            
            # Crear rosa de vientos
            try:
                titulo_grafica = "Rosa de Vientos - Estación Meteorológica"
                if len(df_filtered) < len(df):
                    titulo_grafica += f"\n({start_date.strftime('%d/%m/%Y')} - {end_date.strftime('%d/%m/%Y')})"
                
                fig_rosa = crear_rosa_vientos(
                    direcciones=direcciones,
                    velocidades=velocidades if usar_velocidad and columna_velocidad else None,
                    numero_sectores=numero_sectores,
                    titulo=titulo_grafica,
                    mostrar_velocidades=colorear_por_velocidad and (velocidades is not None)
                )
                
                st.plotly_chart(fig_rosa, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error al crear la rosa de vientos: {str(e)}")
                st.exception(e)
    
    # ========================================================================
    # SECCIÓN DE DATOS Y EXPORTACIÓN
    # ========================================================================
    st.divider()
    
    with st.expander("📋 Ver datos filtrados completos", expanded=False):
        st.dataframe(df_filtered, use_container_width=True)
        st.caption(f"Total: {len(df_filtered)} filas × {len(df_filtered.columns)} columnas")
    
    st.subheader("💾 Descargar Datos Filtrados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df_filtered.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="📥 Descargar CSV",
            data=csv_data,
            file_name="datos_filtrados.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        try:
            excel_buffer = io.BytesIO()
            df_filtered.to_excel(excel_buffer, index=False, engine="openpyxl")
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Descargar Excel",
                data=excel_buffer,
                file_name="datos_filtrados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        except Exception as e:  # noqa: BLE001
            st.error(f"Error al preparar Excel: {e}")


if __name__ == "__main__":
    main()

