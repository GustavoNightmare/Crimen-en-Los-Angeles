import os
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


# ----------------------------------------------------------
# 1) Cargar dataset limpio
# ----------------------------------------------------------
def load_data(path_csv="./dataset/dataset_procesado.csv"):
    df = pd.read_csv(path_csv, parse_dates=['DATE OCC'])
    df['date'] = df['DATE OCC'].dt.normalize()
    return df


# ----------------------------------------------------------
# 2) Crear cuadrícula espacial (celdas)
# ----------------------------------------------------------
def add_grid_cells(df, cell_size_lat=0.02, cell_size_lon=0.02):
    lat_min, lat_max = df['LAT'].min(), df['LAT'].max()
    lon_min, lon_max = df['LON'].min(), df['LON'].max()

    df['cell_y'] = ((df['LAT'] - lat_min) / cell_size_lat).astype(int)
    df['cell_x'] = ((df['LON'] - lon_min) / cell_size_lon).astype(int)
    df['cell_id'] = df['cell_y'] * 10_000 + df['cell_x']

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

    # Lags
    df['lag1'] = g['count'].shift(1)
    df['lag2'] = g['count'].shift(2)
    df['lag7'] = g['count'].shift(7)

    # Media móvil de los últimos 7 días (sin incluir hoy)
    df['roll7'] = g['count'].shift(1).rolling(window=7, min_periods=1).mean()

    # Calendario
    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Coordenadas de celda
    df['cell_x'] = df['cell_id'] % 10000
    df['cell_y'] = df['cell_id'] // 10000

    # Al menos tener lag1 (si no, no hay historial)
    df = df.dropna(subset=['lag1']).copy()

    # Rellenar lags/roll para que NO queden NaNs
    df['lag2'] = df['lag2'].fillna(0)
    df['lag7'] = df['lag7'].fillna(0)
    df['roll7'] = df['roll7'].fillna(0)

    # --------- vecinos (3x3) usando lag1 (ayer) ---------
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

    # Seguridad extra: quitar cualquier NaN/inf residual
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    return df


# ----------------------------------------------------------
# 5) Train / test split temporal + entrenamiento XGBoost
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

    # Esta build de XGBoost no tiene GPU, así que usamos "hist" en CPU.
    model = XGBRegressor(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",      # hist en CPU
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    pred_test = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    mse = mean_squared_error(y_test, pred_test)
    rmse = np.sqrt(mse)

    print("Evaluación XGBoost (últimos días como test):")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    # Guardar modelo en ./modelos
    os.makedirs("./modelos", exist_ok=True)
    model_path = "./modelos/modelo_xgboost_crimen.json"
    model.save_model(model_path)
    print(f"Modelo XGBoost guardado en {model_path}")

    return model, feature_cols, df_features


# ----------------------------------------------------------
# 6) Predicción del día siguiente
# ----------------------------------------------------------
def predict_next_day(model, feature_cols, df_features):
    df = df_features.sort_values(['cell_id', 'crime_category', 'date']).copy()

    last_date = df['date'].max()
    next_date = last_date + pd.Timedelta(days=1)

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

    neigh_map = {
        (row.cell_id, row.crime_category): row.lag1_neigh
        for row in last_day.itertuples()
    }

    rows = []
    for (cell, cat), g in df.groupby(['cell_id', 'crime_category']):
        g = g.sort_values('date')

        last = g.iloc[-1]

        lag1 = last['count']
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

    X_next = next_df[feature_cols]
    next_df['pred_count'] = model.predict(X_next)

    return next_df


# ----------------------------------------------------------
# 7) Main
# ----------------------------------------------------------
def main():
    df = load_data()

    df, grid_info = add_grid_cells(df, cell_size_lat=0.02, cell_size_lon=0.02)
    print("Grid info (lat_min, lon_min, dlat, dlon):", grid_info)

    daily = build_daily_counts(df)

    daily_full = make_full_grid(daily)
    print("Shape daily_full:", daily_full.shape)

    df_feat = add_time_features(daily_full)
    print("Shape df_feat:", df_feat.shape)

    model, feature_cols, df_feat_trained = train_xgboost_gpu(
        df_feat, test_days=60
    )

    next_day_pred = predict_next_day(model, feature_cols, df_feat_trained)
    print("\nEjemplo de predicción para el día siguiente:")
    print(next_day_pred.head(20))

    os.makedirs("./dataset", exist_ok=True)
    out_path = "./dataset/predicciones_siguiente_dia_xgb.csv"
    next_day_pred.to_csv(out_path, index=False)
    print(f"\nPredicciones guardadas en {out_path}")


if __name__ == "__main__":
    main()
