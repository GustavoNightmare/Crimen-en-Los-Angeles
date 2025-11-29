from numpy import dtypes
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# 1) Cargar dataset crudo y revisar duplicados (opcional)
# -----------------------------------------------------------------------------
cri = pd.read_csv('./dataset/datasetcrudo.csv')
print(cri.head(5))

cri.duplicated().any()
cri.duplicated().sum()
print("Duplicados completos:")
print(cri[cri.duplicated()])

duplicados = cri[cri.duplicated(subset='DR_NO', keep=False)]
print("Duplicados por DR_NO:")
print(duplicados)

print("-------------------------------------primera parte-------------------------------------------")
print("IMPORTACIONES ")
print("-------------------------------------primera parte-------------------------------------------")

# Limpieza ligera de Date Rptd (NO tocamos DATE OCC)
cri['Date Rptd'] = cri['Date Rptd'].astype(
    str).str.replace("12:00:00 AM", "", regex=False)

print("-------------------------------------segunda parte-------------------------------------------")
print("Horas ")
print("-------------------------------------segunda parte-------------------------------------------")

# Nos quedamos con las columnas relevantes
columns_to_keep = ['DATE OCC', 'TIME OCC', 'LAT', 'LON', 'Crm Cd Desc']
cri_procesado = cri[columns_to_keep].copy()
print("Dimensiones después de eliminar columnas:", cri_procesado.shape)
print("Columnas en dataset procesado:", cri_procesado.columns.tolist())

# -----------------------------------------------------------------------------
# 2) Mapeo de categorías
# -----------------------------------------------------------------------------
crime_mapping = {
    'Homicidio': ['HOMICIDE', 'MANSLAUGHTER'],
    'Violación': ['RAPE', 'SEXUAL PENETRATION', 'ORAL COPULATION', 'SODOMY'],
    'Robo': ['ROBBERY', 'CARJACKING', 'THEFT', 'SHOPLIFTING', 'PICKPOCKET',
             'PURSE SNATCHING', 'BICYCLE THEFT', 'LARCENY'],
    'Asalto Agravado': ['ASSAULT', 'ADW', 'CHILD BEATING', 'SPOUSAL BEATING',
                        'SHOTS FIRED', 'BRANDISHING'],
    'Incendio Provocado': ['ARSON'],
    'Allanamiento de Morada': ['BURGLARY'],
    'Robo de Vehículo': ['STOLEN VEHICLE', 'GRAND THEFT AUTO', 'AUTO THEFT'],
    'Vandalismo': ['VANDALISM', 'GRAFFITI'],
}

print("Mapeo de categorías definido correctamente")


def map_crime_category(description):
    if pd.isna(description):
        return 'Otros'
    description_upper = str(description).upper()
    for category, keywords in crime_mapping.items():
        if any(keyword in description_upper for keyword in keywords):
            return category
    return 'Otros'


print(cri_procesado.head(5))
print("Función de mapeo definida correctamente")

print("Aplicando mapeo de categorías...")
cri_procesado['crime_category'] = cri_procesado['Crm Cd Desc'].apply(
    map_crime_category)
print("Nueva columna 'crime_category' creada")
print("Valores únicos en crime_category:",
      cri_procesado['crime_category'].unique())

# Ya no necesitamos Crm Cd Desc
cri_procesado = cri_procesado.drop(columns=['Crm Cd Desc'])

# -----------------------------------------------------------------------------
# 3) Filtrar coordenadas válidas
# -----------------------------------------------------------------------------
print("Filtrando coordenadas válidas...")
filas_antes = len(cri_procesado)
cri_procesado = cri_procesado[(
    cri_procesado['LAT'] != 0) & (cri_procesado['LON'] != 0)]
cri_procesado = cri_procesado.dropna(subset=['LAT', 'LON'])
filas_despues = len(cri_procesado)

print(f"Filas antes del filtrado: {filas_antes}")
print(f"Filas después del filtrado: {filas_despues}")
print(f"Filas eliminadas: {filas_antes - filas_despues}")

print("-------------------------------------tercera  parte-------------------------------------------")
print("mapeo de categorias")
print("-------------------------------------tercera  parte-------------------------------------------")

print("\nDistribución de categorías de crimen:")
print(cri_procesado['crime_category'].value_counts())

print('Aqui es el dataset ANTES de convertir DATE OCC:')
print(cri_procesado.head(5))

# Reemplazamos cadenas vacías por NaN
cri_procesado = cri_procesado.replace('', np.nan)

print("Datos nulos por columna ANTES de convertir fechas:")
print(cri_procesado.isnull().sum())

print("-------------------------------------cuarta  parte-------------------------------------------")
print("revision de fechas y horas")
print("-------------------------------------cuarta  parte-------------------------------------------")

# -----------------------------------------------------------------------------
# 4) Conversión de fechas y creación de hora (solo interna)
# -----------------------------------------------------------------------------
# Convertir DATE OCC a datetime
cri_procesado['DATE OCC'] = pd.to_datetime(
    cri_procesado['DATE OCC'].astype(str).str.strip(),
    errors='coerce'
)

print("Después de convertir DATE OCC a datetime:")
print(cri_procesado[['DATE OCC']].head())
print(cri_procesado['DATE OCC'].describe())

# Aseguramos TIME OCC numérico (24h military time)
cri_procesado['TIME OCC'] = pd.to_numeric(
    cri_procesado['TIME OCC'], errors='coerce')

# Hora cruda (0–23, sacada del TIME OCC) SOLO para la corrección
cri_procesado['hour'] = (cri_procesado['TIME OCC'] // 100).astype('Int64')

# -----------------------------------------------------------------------------
# 5) Filtro de fechas buenas usando SOLO DATE OCC
# -----------------------------------------------------------------------------
cutoff = pd.Timestamp("2024-02-28")

mask_good = (cri_procesado['DATE OCC'] >= "2020-01-01") & \
            (cri_procesado['DATE OCC'] < cutoff)

cri_clean = cri_procesado[mask_good].copy()

print("Registros desde 2020 hasta 2024-02-28:", len(cri_clean))

# -----------------------------------------------------------------------------
# 6) Corrección de horas 12:00 por categoría (interno)
# -----------------------------------------------------------------------------
cri_clean['hour_corr'] = cri_clean['hour']

mask_1200_global = (cri_clean['TIME OCC'] == 1200)
candidate_hours = [10, 11, 13, 14]

np.random.seed(42)

for cat in cri_clean['crime_category'].unique():
    mask_cat = (cri_clean['crime_category'] == cat)
    counts = cri_clean.loc[mask_cat, 'hour'].value_counts()
    n12_total = counts.get(12, 0)
    if n12_total == 0:
        continue

    mask_cat_1200 = mask_cat & mask_1200_global
    n12_default = cri_clean.loc[mask_cat_1200 &
                                (cri_clean['hour'] == 12)].shape[0]
    n12_real = n12_total - n12_default

    neighbor_counts = counts.reindex(candidate_hours).fillna(0)
    if neighbor_counts.sum() == 0:
        continue

    target_12 = int(round(neighbor_counts.mean()))
    target_12 = max(target_12, n12_real)

    keep_default = max(0, min(n12_default, target_12 - n12_real))
    n_move = n12_default - keep_default

    if n_move <= 0:
        continue

    neigh_for_probs = neighbor_counts.copy()
    neigh_for_probs[neigh_for_probs <= 0] = 1
    probs = neigh_for_probs / neigh_for_probs.sum()

    idx_12_default = cri_clean.loc[mask_cat_1200 & (
        cri_clean['hour'] == 12)].index
    idx_to_move = np.random.choice(idx_12_default, size=n_move, replace=False)
    new_hours = np.random.choice(probs.index, size=n_move, p=probs.values)

    cri_clean.loc[idx_to_move, 'hour_corr'] = new_hours

print("Corrección aplicada por categoría.")

# -----------------------------------------------------------------------------
# 7) Aplicar la corrección a TIME OCC y preparar dataset FINAL
# -----------------------------------------------------------------------------

# Usamos hour_corr como hora final corregida
cri_clean['hour_final'] = cri_clean['hour_corr'].astype('Int64')

# Donde TIME OCC era 1200 (hora por defecto), lo sustituimos por la hora corregida
mask_def = cri_clean['TIME OCC'] == 1200
cri_clean.loc[mask_def, 'TIME OCC'] = cri_clean.loc[mask_def,
                                                    'hour_final'] * 100  # <<<

# Ahora ya no necesitamos las columnas auxiliares de hora
cri_clean = cri_clean.drop(columns=['hour', 'hour_corr', 'hour_final'])

final_cols = ['DATE OCC', 'TIME OCC', 'LAT', 'LON', 'crime_category']
cri_final = cri_clean[final_cols].copy()

print("HEAD del dataset limpio FINAL:")
print(cri_final.head(10))

# -----------------------------------------------------------------------------
# 8) Guardar dataset limpio
# -----------------------------------------------------------------------------
cri_final.to_csv('./dataset/dataset_procesado.csv', index=False)
print("Dataset limpio guardado en ./dataset/dataset_procesado.csv")
