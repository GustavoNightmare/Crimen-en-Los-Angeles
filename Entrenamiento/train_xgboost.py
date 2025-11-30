import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ----------------------------------------------------------
# 1) Cargar dataset limpio
# ----------------------------------------------------------
def load_data(path_csv="./dataset/dataset_procesado.csv"):
    df = pd.read_csv(path_csv, parse_dates=['DATE OCC'])
    # Solo necesitamos fecha (sin hora) para este modelo diario
    df['date'] = df['DATE OCC'].dt.normalize()
    return df


# ----------------------------------------------------------
# 2) Crear cuadrícula espacial (celdas)
# ----------------------------------------------------------
def add_grid_cells(df, cell_size_lat=0.02, cell_size_lon=0.02):
    """
    cell_size_lat/lon ~ tamaño de celda en grados.
    0.02 es aprox ~2 km, puedes afinarlo luego.
    """
    lat_min, lat_max = df['LAT'].min(), df['LAT'].max()
    lon_min, lon_max = df['LON'].min(), df['LON'].max()

    # indice entero de la celda
    df['cell_y'] = ((df['LAT'] - lat_min) / cell_size_lat).astype(int)
    df['cell_x'] = ((df['LON'] - lon_min) / cell_size_lon).astype(int)

    # id único por celda (número)
    df['cell_id'] = df['cell_y'] * 10_000 + \
        df['cell_x']   # truco para combinarlos

    return df, (lat_min, lon_min, cell_size_lat, cell_size_lon)


# ----------------------------------------------------------
# 3) Agregar por día / celda / categoría
# ----------------------------------------------------------
def build_daily_counts(df):
    daily = (
        df.groupby(['date', 'cell_id', 'crime_category'])
        .size()
        .reset_index(name='count')
    )
    return daily


def make_full_grid(daily):
    """
    Creamos todas las combinaciones date x cell_id x category
    y rellenamos faltantes con 0.
    OJO: si la cuadrícula es muy fina esto se puede poner pesado:
    puedes subir el cell_size_lat/lon para reducir número de celdas.
    """
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


# ----------------------------------------------------------
# 4) Features temporales (lags, medias móviles, calendario, vecinos)
# ----------------------------------------------------------
def add_time_features(daily_full, max_lag=7):
    df = daily_full.sort_values(['cell_id', 'crime_category', 'date']).copy()

    g = df.groupby(['cell_id', 'crime_category'], group_keys=False)

    # Lags simples
    df['lag1'] = g['count'].shift(1)
    df['lag2'] = g['count'].shift(2)
    df['lag7'] = g['count'].shift(7)

    # Media móvil de los últimos 7 días (sin incluir hoy)
    df['roll7'] = g['count'].shift(1).rolling(window=7, min_periods=1).mean()

    # Calendario
    df['dow'] = df['date'].dt.dayofweek  # 0=lunes
    df['month'] = df['date'].dt.month

    # Coordenadas de la celda a partir del cell_id
    df['cell_x'] = df['cell_id'] % 10000
    df['cell_y'] = df['cell_id'] // 10000

    # Quitamos filas sin lag1 (principio de cada serie)
    df = df.dropna(subset=['lag1']).copy()

    # --------- NUEVO: feature de "crimen en celdas vecinas ayer" ---------
    # Usamos lag1 (crímenes de ayer) y lo sumamos en un vecindario 3x3
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

    return df


# ----------------------------------------------------------
# 5) Train / test split temporal + entrenamiento XGBoost GPU
# ----------------------------------------------------------
def train_xgboost_gpu(df_features, test_days=60):
    df_features = df_features.sort_values('date')

    cutoff = df_features['date'].max() - pd.Timedelta(days=test_days)

    train = df_features[df_features['date'] < cutoff]
    test = df_features[df_features['date'] >= cutoff]

    feature_cols = [
        'lag1', 'lag2', 'lag7', 'roll7',
        'lag1_neigh',
        'dow', 'month',
        'cell_x', 'cell_y'
    ]

    X_train = train[feature_cols]
    y_train = train['count']

    X_test = test[feature_cols]
    y_test = test['count']

    # Modelo XGBoost usando GPU
    # Si te da error con "device", prueba:
    #   - actualizar xgboost: pip install -U xgboost
    #   - o cambiar device="cuda" por tree_method="gpu_hist"
    model = XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",   # junto con device="cuda" usa GPU
        device="cuda",        # <- aquí le dices "usa la GPU"
        n_jobs=-1,
        random_state=42,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        verbose=False,
    )

    pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    mse = mean_squared_error(y_test, pred_test)
    rmse = np.sqrt(mse)

    print("Evaluación XGBoost GPU (últimos días como test):")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    return model, feature_cols, df_features


# ----------------------------------------------------------
# 6) Predecir el día siguiente para TODAS las celdas/categorías
#    (incluyendo vecinos)
# ----------------------------------------------------------
def predict_next_day(model, feature_cols, df_features):
    df = df_features.sort_values(['cell_id', 'crime_category', 'date']).copy()

    last_date = df['date'].max()
    next_date = last_date + pd.Timedelta(days=1)

    # Calculamos lag1_neigh para el último día disponible
    last_mask = df['date'] == last_date
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

    # Diccionario (cell_id, categoria) -> lag1_neigh
    neigh_map = {
        (row.cell_id, row.crime_category): row.lag1_neigh
        for row in last_day.itertuples()
    }

    # Construimos las features para el día siguiente
    rows = []
    for (cell, cat), g in df.groupby(['cell_id', 'crime_category']):
        g = g.sort_values('date')

        last = g.iloc[-1]

        # crímenes ayer en esta celda
        lag1 = last['count']
        lag2 = g['count'].iloc[-2] if len(g) >= 2 else 0
        lag7 = g['count'].iloc[-7] if len(g) >= 7 else 0
        roll7 = g['count'].iloc[-7:].mean()

        dow = next_date.dayofweek
        month = next_date.month

        cell_x = cell % 10000
        cell_y = cell // 10000

        lag1_neigh = neigh_map.get((cell, cat), 0.0)  # vecinos ayer

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

    X_next = next_df[feature_cols]
    next_df['pred_count'] = model.predict(X_next)

    return next_df


# ----------------------------------------------------------
# 7) Main
# ----------------------------------------------------------
def main():
    # 1) Cargar
    df = load_data()

    # 2) Cuadrícula
    df, grid_info = add_grid_cells(df, cell_size_lat=0.02, cell_size_lon=0.02)
    print("Grid info (lat_min, lon_min, dlat, dlon):", grid_info)

    # 3) Agregar por día/celda/categoría
    daily = build_daily_counts(df)

    # 3b) Rellenar grid completo con 0
    daily_full = make_full_grid(daily)
    print("Shape daily_full:", daily_full.shape)

    # 4) Features temporales + espaciales
    df_feat = add_time_features(daily_full)
    print("Shape df_feat:", df_feat.shape)

    # 5) Entrenar modelo (XGBoost en GPU)
    model, feature_cols, df_feat_trained = train_xgboost_gpu(
        df_feat, test_days=60
    )

    # 6) Predicción para el día siguiente
    next_day_pred = predict_next_day(model, feature_cols, df_feat_trained)
    print("\nEjemplo de predicción para el día siguiente:")
    print(next_day_pred.head(20))

    # 7) Guardar predicciones
    next_day_pred.to_csv(
        "./dataset/predicciones_siguiente_dia_xgb.csv", index=False
    )
    print("\nPredicciones guardadas en ./dataset/predicciones_siguiente_dia_xgb.csv")


if __name__ == "__main__":
    main()
