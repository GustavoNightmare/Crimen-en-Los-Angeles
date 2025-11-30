import torch.nn as nn
import torch
import os
import time
import random
import numpy as np
import pandas as pd

from flask import Flask, render_template, request, url_for

import matplotlib
matplotlib.use("Agg")  # <-- añade esta línea

# Si quieres también usar XGBoost más adelante:
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ==========================================================
#   0) Config y semillas
# ==========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
)

# ==========================================================
#   1) Funciones de datos (mismas que en entrenamiento)
# ==========================================================


def load_data(path_csv="./dataset/dataset_procesado.csv"):
    df = pd.read_csv(path_csv, parse_dates=['DATE OCC'])
    df['date'] = df['DATE OCC'].dt.normalize()
    return df


def add_grid_cells(df, cell_size_lat=0.02, cell_size_lon=0.02):
    lat_min, lat_max = df['LAT'].min(), df['LAT'].max()
    lon_min, lon_max = df['LON'].min(), df['LON'].max()

    df['cell_y'] = ((df['LAT'] - lat_min) / cell_size_lat).astype(int)
    df['cell_x'] = ((df['LON'] - lon_min) / cell_size_lon).astype(int)
    df['cell_id'] = df['cell_y'] * 10_000 + df['cell_x']

    return df, (lat_min, lon_min, cell_size_lat, cell_size_lon)


def build_daily_counts(df):
    daily = (
        df.groupby(['date', 'cell_id', 'crime_category'])
        .size()
        .reset_index(name='count')
    )
    return daily


def make_full_grid(daily):
    all_dates = pd.date_range(daily['date'].min(),
                              daily['date'].max(),
                              freq='D')
    cells = daily['cell_id'].unique()
    cats = daily['crime_category'].unique()

    idx = pd.MultiIndex.from_product(
        [all_dates, cells, cats],
        names=['date', 'cell_id', 'crime_category']
    )

    daily_full = (
        daily.set_index(['date', 'cell_id', 'crime_category'])
             .reindex(idx, fill_value=0)
             .reset_index()
    )

    return daily_full


def add_time_features(daily_full, max_lag=7):
    df = daily_full.sort_values(['cell_id', 'crime_category', 'date']).copy()

    g = df.groupby(['cell_id', 'crime_category'], group_keys=False)

    df['lag1'] = g['count'].shift(1)
    df['lag2'] = g['count'].shift(2)
    df['lag7'] = g['count'].shift(7)

    df['roll7'] = g['count'].shift(1).rolling(window=7, min_periods=1).mean()

    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['cell_x'] = df['cell_id'] % 10000
    df['cell_y'] = df['cell_id'] // 10000

    df = df.dropna(subset=['lag1']).copy()

    df['lag2'] = df['lag2'].fillna(0)
    df['lag7'] = df['lag7'].fillna(0)
    df['roll7'] = df['roll7'].fillna(0)

    base = df[['date', 'crime_category', 'cell_x', 'cell_y', 'lag1']].copy()

    frames = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            tmp = base.copy()
            tmp['cell_x'] += dx
            tmp['cell_y'] += dy
            frames.append(tmp)

    neigh_all = pd.concat(frames, ignore_index=True)

    neigh_agg = (
        neigh_all
        .groupby(['date', 'crime_category', 'cell_x', 'cell_y'])['lag1']
        .sum()
        .reset_index()
        .rename(columns={'lag1': 'lag1_neigh'})
    )

    df = df.merge(
        neigh_agg,
        on=['date', 'crime_category', 'cell_x', 'cell_y'],
        how='left'
    )
    df['lag1_neigh'] = df['lag1_neigh'].fillna(0)

    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    return df


# ==========================================================
#   2) Modelo MLP (igual que en entrenamiento afinado)
# ==========================================================
class CrimeMLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class TorchRegressorWrapper:
    def __init__(self, net, device, feature_cols, means, stds):
        self.net = net
        self.device = device
        self.feature_cols = feature_cols
        self.means = means.astype(np.float32)
        self.stds = stds.astype(np.float32)

    def predict(self, X_df):
        self.net.eval()
        X = X_df[self.feature_cols].values.astype(np.float32)
        X_norm = (X - self.means) / self.stds
        with torch.no_grad():
            X_t = torch.from_numpy(X_norm).to(self.device)
            y_pred = self.net(X_t).cpu().numpy().ravel()
        return y_pred


# ==========================================================
#   3) Cargar datos y modelos AL ARRANCAR EL SERVIDOR
# ==========================================================
print("Cargando datos y preparando modelo...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo PyTorch:", device)

# 1) Datos base
df_raw = load_data("./dataset/dataset_procesado.csv")
df_raw, grid_info = add_grid_cells(
    df_raw, cell_size_lat=0.02, cell_size_lon=0.02)
lat_min, lon_min, dlat, dlon = grid_info

daily = build_daily_counts(df_raw)
daily_full = make_full_grid(daily)
df_feat = add_time_features(daily_full)

last_date = df_feat['date'].max()  # "hoy" = último día
print("Última fecha en el dataset:", last_date.date())

# 2) Features / normalización como en entrenamiento
feature_cols = [
    'lag1', 'lag2', 'lag7', 'roll7',
    'lag1_neigh',
    'dow', 'month',
    'cell_x', 'cell_y'
]

test_days = 60
cutoff = last_date - pd.Timedelta(days=test_days)
train_feat = df_feat[df_feat['date'] < cutoff]

X_train = train_feat[feature_cols].values.astype(np.float32)
means = X_train.mean(axis=0)
stds = X_train.std(axis=0)
stds[stds == 0] = 1.0

# 3) Cargar modelo MLP entrenado
mlp = CrimeMLP(in_features=len(feature_cols)).to(device)
state_dict_path = "./modelos/modelo_mlp_crimen.pt"
if not os.path.exists(state_dict_path):
    raise FileNotFoundError(
        f"No se encontró {state_dict_path}. Entrena primero el modelo MLP.")

state_dict = torch.load(state_dict_path, map_location=device)
mlp.load_state_dict(state_dict)
mlp_wrapper = TorchRegressorWrapper(mlp, device, feature_cols, means, stds)

# 4) Si quieres también XGBoost (opcional)
xgb_model = None
if HAS_XGB and os.path.exists("./modelos/modelo_xgboost_crimen.json"):
    xgb_model = XGBRegressor()
    xgb_model.load_model("./modelos/modelo_xgboost_crimen.json")
    print("Modelo XGBoost cargado.")
else:
    print("XGBoost no disponible o modelo no encontrado; solo se usará MLP.")

# 5) Lista de categorías
crime_categories = sorted(df_feat['crime_category'].unique().tolist())


# ==========================================================
#   4) Helpers para predicciones y heatmap
# ==========================================================
def build_pred_for_day(model_name="mlp", day_choice="tomorrow"):
    """
    model_name: 'mlp' o 'xgb'
    day_choice: 'today' o 'tomorrow'
    Devuelve un DataFrame con columnas:
    [date, cell_id, crime_category, ...features..., pred_count]
    """
    global df_feat, last_date, feature_cols

    if model_name == "xgb" and xgb_model is None:
        raise ValueError("Modelo XGBoost no disponible en este servidor.")

    if day_choice == "today":
        # Usamos las features ya calculadas para el último día
        day = last_date
        df_day = df_feat[df_feat['date'] == day].copy()

        if model_name == "mlp":
            df_day['pred_count'] = mlp_wrapper.predict(df_day)
        else:
            X_day = df_day[feature_cols].values.astype(np.float32)
            df_day['pred_count'] = xgb_model.predict(X_day)

        return df_day

    elif day_choice == "tomorrow":
        # Construimos features para el día siguiente (igual que predict_next_day)
        df = df_feat.sort_values(['cell_id', 'crime_category', 'date']).copy()

        last_date_local = df['date'].max()
        next_date = last_date_local + pd.Timedelta(days=1)

        # Para vecinos (lag1_neigh) usamos el último día real
        last_mask = df['date'] == last_date_local
        last_day = df.loc[last_mask, ['cell_id', 'crime_category',
                                      'cell_x', 'cell_y', 'count']].copy()

        base = last_day[['crime_category', 'cell_x', 'cell_y', 'count']].copy()

        frames = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                tmp = base.copy()
                tmp['cell_x'] += dx
                tmp['cell_y'] += dy
                frames.append(tmp)

        neigh_all = pd.concat(frames, ignore_index=True)

        neigh_agg = (
            neigh_all
            .groupby(['crime_category', 'cell_x', 'cell_y'])['count']
            .sum()
            .reset_index()
            .rename(columns={'count': 'lag1_neigh'})
        )

        last_day = last_day.merge(
            neigh_agg,
            on=['crime_category', 'cell_x', 'cell_y'],
            how='left'
        )
        last_day['lag1_neigh'] = last_day['lag1_neigh'].fillna(0)

        neigh_map = {
            (row.cell_id, row.crime_category): row.lag1_neigh
            for row in last_day.itertuples()
        }

        rows = []
        for (cell, cat), g in df.groupby(['cell_id', 'crime_category']):
            g = g.sort_values('date')

            last_row = g.iloc[-1]

            lag1 = last_row['count']
            lag2 = g['count'].iloc[-2] if len(g) >= 2 else 0
            lag7 = g['count'].iloc[-7] if len(g) >= 7 else 0
            roll7 = g['count'].iloc[-7:].mean()

            dow = next_date.dayofweek
            month = next_date.month

            cell_x = cell % 10000
            cell_y = cell // 10000

            lag1_neigh = neigh_map.get((cell, cat), 0.0)

            rows.append({
                'date': next_date,
                'cell_id': cell,
                'crime_category': cat,
                'lag1': lag1,
                'lag2': lag2,
                'lag7': lag7,
                'roll7': roll7,
                'dow': dow,
                'month': month,
                'cell_x': cell_x,
                'cell_y': cell_y,
                'lag1_neigh': lag1_neigh
            })

        next_df = pd.DataFrame(rows)

        if model_name == "mlp":
            next_df['pred_count'] = mlp_wrapper.predict(next_df)
        else:
            X_next = next_df[feature_cols].values.astype(np.float32)
            next_df['pred_count'] = xgb_model.predict(X_next)

        return next_df

    else:
        raise ValueError("day_choice debe ser 'today' o 'tomorrow'")


def build_heatmap_df(full_pred_df, crime_category, top_percent=0.05):
    """
    A partir de un DF de predicciones (todas las categorías),
    devuelve solo la categoría dada, con columnas:
    [lat_center, lon_center, pred_count, prob_1plus, is_hotspot, ...]
    """
    df_cat = full_pred_df[full_pred_df['crime_category']
                          == crime_category].copy()
    if df_cat.empty:
        return df_cat

    if "cell_x" not in df_cat.columns or "cell_y" not in df_cat.columns:
        df_cat["cell_x"] = df_cat["cell_id"] % 10000
        df_cat["cell_y"] = df_cat["cell_id"] // 10000

    lat_min, lon_min, dlat, dlon = grid_info
    df_cat["lat_center"] = lat_min + (df_cat["cell_y"] + 0.5) * dlat
    df_cat["lon_center"] = lon_min + (df_cat["cell_x"] + 0.5) * dlon

    df_cat["pred_count"] = df_cat["pred_count"].clip(lower=0)
    df_cat["prob_1plus"] = 1.0 - np.exp(-df_cat["pred_count"])

    if len(df_cat) > 0:
        threshold = df_cat["pred_count"].quantile(1 - top_percent)
    else:
        threshold = 0.0

    df_cat["is_hotspot"] = df_cat["pred_count"] >= threshold
    df_cat["hotspot_threshold"] = threshold

    return df_cat


def save_heatmap_image(df_cat, crime_category, metric="prob_1plus"):
    import matplotlib.pyplot as plt

    if df_cat.empty:
        return None

    metric_col = metric

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        df_cat["lon_center"],
        df_cat["lat_center"],
        c=df_cat[metric_col],
        s=12,
        alpha=0.9
    )
    plt.colorbar(sc, label=metric_col)
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.title(f"Heatmap predicho - {crime_category} ({metric_col})")
    plt.tight_layout()

    static_dir = app.static_folder
    os.makedirs(static_dir, exist_ok=True)
    filepath = os.path.join(static_dir, "heatmap.png")

    plt.savefig(filepath, bbox_inches="tight", dpi=120)
    plt.close()

    return "heatmap.png"


# ==========================================================
#   5) Rutas Flask
# ==========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    selected_model = "mlp"
    selected_day = "tomorrow"       # por defecto: "mañana"
    selected_category = crime_categories[0] if crime_categories else ""
    img_url = None
    hotspot_threshold = None
    total_cells = None

    if request.method == "POST":
        selected_model = request.form.get("model", "mlp")
        selected_day = request.form.get("day", "tomorrow")
        selected_category = request.form.get("category", selected_category)

        # 1) Predicciones para ese día
        pred_df = build_pred_for_day(
            model_name=selected_model,
            day_choice=selected_day
        )

        # 2) Heatmap para esa categoría
        df_heat = build_heatmap_df(
            pred_df,
            selected_category,
            top_percent=0.05
        )

        total_cells = len(df_heat)
        hotspot_threshold = (
            df_heat["hotspot_threshold"].iloc[0]
            if not df_heat.empty
            else None
        )

        # 3) Guardar imagen y construir URL
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
        selected_model=selected_model,
        selected_day=selected_day,
        img_url=img_url,
        has_xgb=(xgb_model is not None),
        last_date=last_date.date(),
        hotspot_threshold=hotspot_threshold,
        total_cells=total_cells,
    )


if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    app.run(debug=True)
