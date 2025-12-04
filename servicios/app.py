import joblib
from matplotlib import colors
import matplotlib.pyplot as plt
import os
import time

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, url_for

import matplotlib
matplotlib.use("Agg")  # backend sin GUI para la Jetson


# ==========================================================
#   0) Flask
# ==========================================================
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)


# ==========================================================
#   1) Cargar datos y modelos Random Forest
# ==========================================================
def load_data(path_csv="./dataset/dataset_procesado.csv",
              max_years=4,
              cut_2024=True):
    """
    Carga el dataset procesado y crea la columna 'date'.
    Usamos el mismo corte que en el entrenamiento.
    """
    df = pd.read_csv(path_csv, parse_dates=["DATE OCC"])
    df["date"] = df["DATE OCC"].dt.normalize()

    if cut_2024:
        df = df[df["date"] < pd.Timestamp("2024-03-07")].copy()

    if max_years is not None:
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=365 * max_years)
        df = df[df["date"] >= cutoff].copy()
        print(f"Usando datos desde {cutoff.date()} hasta {max_date.date()}")

    print("Shape datos crudos:", df.shape)
    return df


print("Cargando datos y modelos de Random Forest...")

# 1) Dataset
df_raw = load_data("./dataset/dataset_procesado.csv")
last_date = df_raw["date"].max()
print("Última fecha en el dataset:", last_date.date())

# 2) KMeans de zonas (mismo que en entrenamiento)
kmeans_path = "./modelos/kmeans_zonas.joblib"
if not os.path.exists(kmeans_path):
    raise FileNotFoundError(
        f"No se encontró {kmeans_path}. Ejecuta primero pruebasrandom.py"
    )

kmeans = joblib.load(kmeans_path)

# Asignar zona a cada crimen
df_raw["zone"] = kmeans.predict(df_raw[["LAT", "LON"]].values)

# Centros de zona (para pintar el mapa)
centers = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=["zone_lat", "zone_lon"]
)
centers["zone"] = centers.index

# 3) Conteo diario por zona y categoría
daily = (
    df_raw.groupby(["crime_category", "zone", "date"])
    .size()
    .reset_index(name="count")
)
print("Shape daily (día x zona x categoría con >=1 crimen):", daily.shape)

# 4) Modelos Random Forest por categoría
rf_models_path = "./modelos/random_forests_por_categoria.joblib"
if not os.path.exists(rf_models_path):
    raise FileNotFoundError(
        f"No se encontró {rf_models_path}. Ejecuta primero pruebasrandom.py"
    )

# dict: cat -> {"model", "feature_cols"}
rf_models = joblib.load(rf_models_path)

crime_categories = sorted(daily["crime_category"].unique())
print("Categorías:", crime_categories)

# Para no recalcular siempre, cacheamos la serie completa por categoría
full_cat_cache = {}


# ==========================================================
#   2) Funciones auxiliares (mismas que en entrenamiento)
# ==========================================================
def make_supervised_for_category(daily_cat):
    """
    Construye la serie diaria por zona para UNA categoría:
    columnas: date, zone, count, dow, month, lag1, lag7, roll7
    (rellenando días sin crímenes con 0).
    """
    all_dates = pd.date_range(
        daily_cat["date"].min(),
        daily_cat["date"].max(),
        freq="D"
    )

    pieces = []
    for zone, g in daily_cat.groupby("zone"):
        g = g.set_index("date")["count"] \
             .reindex(all_dates, fill_value=0) \
             .rename("count") \
             .to_frame()

        g = g.rename_axis("date").reset_index()
        g["zone"] = zone
        pieces.append(g)

    full = pd.concat(pieces, ignore_index=True)

    # Calendario
    full["dow"] = full["date"].dt.dayofweek
    full["month"] = full["date"].dt.month

    # Lags por zona
    full = full.sort_values(["zone", "date"])
    grp = full.groupby("zone", group_keys=False)

    full["lag1"] = grp["count"].shift(1)
    full["lag7"] = grp["count"].shift(7)
    full["roll7"] = grp["count"].shift(1).rolling(7, min_periods=1).mean()

    # Quitamos filas sin lag1
    full = full.dropna(subset=["lag1"]).copy()

    # Tipos pequeños para ahorrar RAM
    for col in ["count", "lag1", "lag7", "roll7"]:
        full[col] = full[col].astype("float32")
    full["zone"] = full["zone"].astype("int16")
    full["dow"] = full["dow"].astype("int8")
    full["month"] = full["month"].astype("int8")

    return full


def predict_next_day_category(model, feature_cols, full_cat):
    """
    A partir de la serie completa de UNA categoría, construye las
    features para el día siguiente en cada zona y aplica el modelo.
    """
    full_cat = full_cat.sort_values(["zone", "date"])
    last_date_local = full_cat["date"].max()
    next_date = last_date_local + pd.Timedelta(days=1)

    rows = []
    for zone, g in full_cat.groupby("zone"):
        g = g.sort_values("date")
        last = g.iloc[-1]

        lag1 = last["count"]
        lag7 = g["count"].iloc[-7] if len(g) >= 7 else last["count"]
        roll7 = g["count"].iloc[-7:].mean()

        rows.append({
            "date": next_date,
            "zone": int(zone),
            "lag1": float(lag1),
            "lag7": float(lag7),
            "roll7": float(roll7),
            "dow": next_date.dayofweek,
            "month": next_date.month
        })

    df_next = pd.DataFrame(rows)
    X_next = df_next[feature_cols].values
    df_next["pred_count"] = model.predict(X_next)
    return df_next


def get_next_predictions_for_category(cat):
    """
    Devuelve un DataFrame con:
      date, zone, pred_count, crime_category, zone_lat, zone_lon
    para la categoría 'cat' predicha para el día siguiente.
    """
    if cat not in rf_models:
        raise ValueError(f"No hay modelo entrenado para categoría: {cat}")

    # Serie completa cacheada o se genera una vez
    if cat not in full_cat_cache:
        daily_cat = daily[daily["crime_category"] == cat].copy()
        full_cat_cache[cat] = make_supervised_for_category(daily_cat)

    full_cat = full_cat_cache[cat]
    model_info = rf_models[cat]
    model = model_info["model"]
    feature_cols = model_info["feature_cols"]

    df_next = predict_next_day_category(model, feature_cols, full_cat)
    df_next["crime_category"] = cat

    # Añadir coordenadas de la zona
    df_next = df_next.merge(
        centers[["zone", "zone_lat", "zone_lon"]],
        on="zone",
        how="left"
    )
    return df_next


# ==========================================================
#   3) Heatmap helpers
# ==========================================================
def build_heatmap_df(df_next, top_percent=0.05):
    """
    A partir de un DF de predicciones de una categoría:
    añade prob_1plus, is_hotspot y hotspot_threshold.
    """
    df = df_next.copy()
    df["pred_count"] = df["pred_count"].clip(lower=0)
    df["prob_1plus"] = 1.0 - np.exp(-df["pred_count"])

    if len(df) > 0:
        threshold = df["pred_count"].quantile(1 - top_percent)
    else:
        threshold = 0.0

    df["is_hotspot"] = df["pred_count"] >= threshold
    df["hotspot_threshold"] = threshold
    return df


def save_heatmap_image(df_cat, crime_category, metric="prob_1plus"):
    """
    Dibuja un scatter de las zonas sobre un mapa real de LA,
    coloreado por la probabilidad en PORCENTAJE (0–100),
    y guarda el resultado en static/heatmap.png
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import contextily as cx

    if df_cat.empty:
        return None

    # Pasar de probabilidad [0,1] a porcentaje [0,100]
    values = 100.0 * df_cat[metric].values  # ej. 0.73 -> 73 %

    fig, ax = plt.subplots(figsize=(8, 6))

    sc = ax.scatter(
        df_cat["zone_lon"],
        df_cat["zone_lat"],
        c=values,
        s=40,
        alpha=0.9,
        cmap="inferno",   # escala lineal
        edgecolor="black",
        linewidth=0.3,
    )

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Prob. ≥1 crimen (%)")

    # Límites del mapa con un pequeño margen
    margin_lon = 0.02
    margin_lat = 0.02
    ax.set_xlim(df_cat["zone_lon"].min() - margin_lon,
                df_cat["zone_lon"].max() + margin_lon)
    ax.set_ylim(df_cat["zone_lat"].min() - margin_lat,
                df_cat["zone_lat"].max() + margin_lat)

    # Fondo de mapa (OpenStreetMap / Carto)
    # CRS 4326 porque estamos usando lat/lon
    cx.add_basemap(
        ax,
        crs="EPSG:4326",
        source=cx.providers.CartoDB.Positron  # mapa clarito
    )

    ax.set_xlabel("Longitud")
    ax.set_ylabel("Latitud")
    ax.set_title(f"Heatmap predicho - {crime_category}")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()

    static_dir = app.static_folder
    os.makedirs(static_dir, exist_ok=True)
    filepath = os.path.join(static_dir, "heatmap.png")

    plt.savefig(filepath, bbox_inches="tight", dpi=120)
    plt.close()

    return "heatmap.png"

# ==========================================================
#   4) Rutas Flask
# ==========================================================


@app.route("/", methods=["GET", "POST"])
def index():
    selected_category = crime_categories[0] if crime_categories else ""
    img_url = None
    hotspot_threshold = None
    total_zones = None
    pred_date = None

    if request.method == "POST":
        selected_category = request.form.get("category", selected_category)

        # 1) Predicciones Random Forest para ESA categoría (día siguiente)
        df_next = get_next_predictions_for_category(selected_category)
        if not df_next.empty:
            pred_date = df_next["date"].iloc[0].date()

        # 2) Preparar DF para heatmap
        df_heat = build_heatmap_df(df_next, top_percent=0.05)
        total_zones = len(df_heat)
        hotspot_threshold = (
            df_heat["hotspot_threshold"].iloc[0]
            if not df_heat.empty
            else None
        )

        # 3) Guardar imagen
        img_file = save_heatmap_image(
            df_heat,
            selected_category,
            metric="prob_1plus"
        )
        if img_file:
            img_url = url_for("static", filename=img_file) + \
                f"?v={int(time.time())}"

    return render_template(
        "index.html",
        crime_categories=crime_categories,
        selected_category=selected_category,
        img_url=img_url,
        last_date=last_date.date(),
        pred_date=pred_date,
        hotspot_threshold=hotspot_threshold,
        total_zones=total_zones,
    )


if __name__ == "__main__":
    # Servidor de desarrollo
    app.run(host="0.0.0.0", port=5000, debug=True)
