from matplotlib.widgets import Slider, RadioButtons
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, Dropdown

# Si vienes del CSV limpio:
df = pd.read_csv('./dataset/dataset_procesado.csv')

df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])
df['TIME OCC'] = pd.to_numeric(df['TIME OCC'], errors='coerce')

# columnas de tiempo
df['year'] = df['DATE OCC'].dt.year
df['month_num'] = df['DATE OCC'].dt.month
df['day'] = df['DATE OCC'].dt.day
df['dow_num'] = df['DATE OCC'].dt.dayofweek          # 0=lunes
df['hour'] = (df['TIME OCC'] // 100).astype(int)

# Si tuvieras hour_corr en este df, usarla; si no, usamos hour normal
df['hour_plot'] = df.get('hour_corr', df['hour'])

# etiquetas ordenadas
day_labels = ['Lunes', 'Martes', 'Miércoles',
              'Jueves', 'Viernes', 'Sábado', 'Domingo']
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


# 1. Mapa de calor: crímenes por día de la semana y hora
day_hour = (
    df
    .groupby(['dow_num', 'hour_plot'])
    .size()
    .unstack(fill_value=0)
    .reindex(index=range(7), columns=range(24))
)

plt.figure(figsize=(12, 6))
plt.imshow(day_hour, aspect='auto')
plt.colorbar(label='Número de crímenes')
plt.xticks(range(24), range(24))
plt.yticks(range(7), day_labels)
plt.xlabel('Hora del día')
plt.ylabel('Día de la semana')
plt.title('Mapa de calor: crímenes por día de la semana y hora')
plt.tight_layout()
plt.show()


# 2. Mapa de calor: crímenes por año y mes (total)

year_month_pivot = (
    df
    .groupby(['year', 'month_num'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(1, 13))
)

plt.figure(figsize=(12, 6))
plt.imshow(year_month_pivot, aspect='auto')
plt.colorbar(label='Número de crímenes')
plt.xticks(range(12), month_labels)
plt.yticks(range(len(year_month_pivot.index)), year_month_pivot.index)
plt.xlabel('Mes')
plt.ylabel('Año')
plt.title('Mapa de calor: crímenes por año y mes')
plt.tight_layout()
plt.show()


# 3. Crímenes totales por mes (serie temporal)

df['year_month'] = df['DATE OCC'].dt.to_period('M')

counts_month = df['year_month'].value_counts().sort_index()

plt.figure(figsize=(12, 4))
counts_month.plot(kind='line', marker='o')
plt.xlabel('Año-Mes')
plt.ylabel('Número de crímenes')
plt.title('Crímenes totales por mes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 4. Crímenes por hora del día (total y por categoría)

hour_counts = (
    df.groupby('hour_plot')
      .size()
      .reindex(range(24), fill_value=0)
)

plt.figure(figsize=(10, 4))
hour_counts.plot(kind='bar')
plt.xlabel('Hora del día')
plt.ylabel('Número de crímenes')
plt.title('Crímenes totales por hora del día')
plt.tight_layout()
plt.show()

# Por categoría (una gráfica por categoría)

cats = sorted(df['crime_category'].unique())

for cat in cats:
    sub = df[df['crime_category'] == cat]
    counts = (
        sub.groupby('hour_plot')
        .size()
        .reindex(range(24), fill_value=0)
    )

    plt.figure(figsize=(10, 4))
    counts.plot(kind='bar')
    plt.xlabel('Hora del día')
    plt.ylabel('Número de crímenes')
    plt.title(f'Crímenes por hora – {cat}')
    plt.tight_layout()
    plt.show()

# 5. Crímenes por categoría y mes (mapa de calor)

cat_month = (
    df
    .groupby(['crime_category', 'month_num'])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(1, 13))
)

plt.figure(figsize=(14, max(6, len(cat_month)*0.35)))
plt.imshow(cat_month, aspect='auto')
plt.colorbar(label='Número de crímenes')
plt.xticks(range(12), month_labels, rotation=45, ha='right')
plt.yticks(range(len(cat_month.index)), cat_month.index)
plt.xlabel('Mes')
plt.ylabel('Categoría de crimen')
plt.title('Crímenes por categoría y mes')
plt.tight_layout()
plt.show()

# Mapa de calor separado para cada categoría

for cat in cats:
    sub = df[df['crime_category'] == cat]
    pivot = (
        sub
        .groupby('month_num')
        .size()
        .reindex(range(1, 13), fill_value=0)
        .to_frame(cat)
        .T
    )

    plt.figure(figsize=(8, 2))
    plt.imshow(pivot, aspect='auto')
    plt.colorbar(label='Número de crímenes')
    plt.xticks(range(12), month_labels, rotation=45, ha='right')
    plt.yticks([0], [cat])
    plt.xlabel('Mes')
    plt.ylabel('Categoría')
    plt.title(f'Crímenes por mes – {cat}')
    plt.tight_layout()
    plt.show()

# 6. Mapa (plano cartesiano) LAT vs LON con sliders

# límites para mantener siempre el mismo cuadro

# ==================  CONFIGURACIÓN BÁSICA  ==================

lon_min, lon_max = df['LON'].min(), df['LON'].max()
lat_min, lat_max = df['LAT'].min(), df['LAT'].max()

years = sorted(df['year'].unique())
cats = ['Todas'] + sorted(df['crime_category'].unique())

# Valores iniciales
init_year = years[0]
init_month = 0   # 0 = todos
init_day = 0   # 0 = todos
init_hour = 0   # 0 = todas
init_cat = 'Todas'

max_points = 8000


def filtrar_df(categoria, year, month, day, hour):
    sub = df[df['year'] == year].copy()

    if categoria != 'Todas':
        sub = sub[sub['crime_category'] == categoria]
    if month != 0:
        sub = sub[sub['month_num'] == month]
    if day != 0:
        sub = sub[sub['day'] == day]
    if hour != 0:
        sub = sub[sub['hour_plot'] == hour]

    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=0)

    return sub

# ==================  FIGURA Y CONTROLES  ==================


fig, ax = plt.subplots()
# Dejamos espacio abajo e izquierda para sliders y radio buttons
plt.subplots_adjust(left=0.28, bottom=0.30)

# Datos iniciales
sub0 = filtrar_df(init_cat, init_year, init_month, init_day, init_hour)
scat = ax.scatter(sub0['LON'], sub0['LAT'], s=3, alpha=0.5)

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.set_title(f'Crímenes – {init_cat} – {init_year}')

# ----- Sliders (todos en la parte de abajo) -----
ax_year = plt.axes([0.30, 0.22, 0.60, 0.03])
ax_month = plt.axes([0.30, 0.17, 0.60, 0.03])
ax_day = plt.axes([0.30, 0.12, 0.60, 0.03])
ax_hour = plt.axes([0.30, 0.07, 0.60, 0.03])

s_year = Slider(ax_year,  'Año',
                years[0], years[-1], valinit=init_year,  valstep=1)
s_month = Slider(ax_month, 'Mes (0=Todos)', 0, 12,
                 valinit=init_month, valstep=1)
s_day = Slider(ax_day,   'Día (0=Todos)', 0, 31, valinit=init_day,   valstep=1)
s_hour = Slider(ax_hour,  'Hora (0=Todas)', 0,
                23, valinit=init_hour,  valstep=1)

# ----- RadioButtons para categoría (lado izquierdo) -----
rax = plt.axes([0.02, 0.35, 0.20, 0.60])
radio = RadioButtons(rax, cats, active=0)
rax.set_title("Categoría")

# ==================  CALLBACK DE ACTUALIZACIÓN  ==================


def update(val=None):
    year = int(s_year.val)
    month = int(s_month.val)
    day = int(s_day.val)
    hour = int(s_hour.val)
    cat = radio.value_selected

    sub = filtrar_df(cat, year, month, day, hour)

    if len(sub) > 0:
        coords = np.column_stack([sub['LON'].values, sub['LAT'].values])
    else:
        coords = np.empty((0, 2))

    scat.set_offsets(coords)

    titulo = f'Crímenes – {cat if cat != "Todas" else "Todas las categorías"} – {year}'
    if month != 0:
        titulo += f', mes {month}'
    if day != 0:
        titulo += f', día {day}'
    if hour != 0:
        titulo += f', hora {hour}'
    ax.set_title(titulo)

    fig.canvas.draw_idle()


# Conectamos sliders y radio al callback
s_year.on_changed(update)
s_month.on_changed(update)
s_day.on_changed(update)
s_hour.on_changed(update)
radio.on_clicked(update)

plt.show()

# 7. Mapa de densidad espacial con los mismos sliders

# bordes fijos del histograma

fig2, ax2 = plt.subplots()
plt.subplots_adjust(left=0.28, bottom=0.30)

bins = 60
xedges = np.linspace(lon_min, lon_max, bins+1)
yedges = np.linspace(lat_min, lat_max, bins+1)


def densidad_filtrada(cat, year, month, day, hour):
    sub = filtrar_df(cat, year, month, day, hour)
    H, _, _ = np.histogram2d(sub['LON'], sub['LAT'],
                             bins=[xedges, yedges])
    return H


H0 = densidad_filtrada(init_cat, init_year, init_month, init_day, init_hour)
img = ax2.imshow(H0.T, origin='lower',
                 extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                 aspect='auto')
cbar = plt.colorbar(img, ax=ax2, label='N° de crímenes')

ax2.set_xlabel('Longitud')
ax2.set_ylabel('Latitud')
ax2.set_title(f'Densidad espacial – {init_cat} – {init_year}')

ax2.set_xlim(lon_min, lon_max)
ax2.set_ylim(lat_min, lat_max)

# Sliders segunda figura
ax2_year = plt.axes([0.30, 0.22, 0.60, 0.03])
ax2_month = plt.axes([0.30, 0.17, 0.60, 0.03])
ax2_day = plt.axes([0.30, 0.12, 0.60, 0.03])
ax2_hour = plt.axes([0.30, 0.07, 0.60, 0.03])

s2_year = Slider(ax2_year,  'Año',
                 years[0], years[-1], valinit=init_year,  valstep=1)
s2_month = Slider(ax2_month, 'Mes (0=Todos)', 0,
                  12, valinit=init_month, valstep=1)
s2_day = Slider(ax2_day,   'Día (0=Todos)', 0,
                31, valinit=init_day,   valstep=1)
s2_hour = Slider(ax2_hour,  'Hora (0=Todas)', 0,
                 23, valinit=init_hour,  valstep=1)

rax2 = plt.axes([0.02, 0.35, 0.20, 0.60])
radio2 = RadioButtons(rax2, cats, active=0)
rax2.set_title("Categoría")


def update2(val=None):
    year = int(s2_year.val)
    month = int(s2_month.val)
    day = int(s2_day.val)
    hour = int(s2_hour.val)
    cat = radio2.value_selected

    H = densidad_filtrada(cat, year, month, day, hour)
    img.set_data(H.T)
    img.set_clim(vmin=0, vmax=H.max() if H.max() > 0 else 1)

    titulo = f'Densidad espacial – {cat if cat != "Todas" else "Todas las categorías"} – {year}'
    if month != 0:
        titulo += f', mes {month}'
    if day != 0:
        titulo += f', día {day}'
    if hour != 0:
        titulo += f', hora {hour}'
    ax2.set_title(titulo)

    fig2.canvas.draw_idle()


s2_year.on_changed(update2)
s2_month.on_changed(update2)
s2_day.on_changed(update2)
s2_hour.on_changed(update2)
radio2.on_clicked(update2)

plt.show()
