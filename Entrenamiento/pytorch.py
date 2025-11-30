import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


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

    df['lag1'] = g['count'].shift(1)
    df['lag2'] = g['count'].shift(2)
    df['lag7'] = g['count'].shift(7)

    df['roll7'] = g['count'].shift(1).rolling(window=7, min_periods=1).mean()

    df['dow'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    df['cell_x'] = df['cell_id'] % 10000
    df['cell_y'] = df['cell_id'] // 10000

    df = df.dropna(subset=['lag1']).copy()

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
# 5) Dataset y modelo PyTorch
# ----------------------------------------------------------
class CrimeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
    """
    Wrapper para que el modelo de PyTorch tenga .predict(X)
    como un modelo de sklearn/xgboost.
    """

    def __init__(self, net, device):
        self.net = net
        self.device = device

    def predict(self, X_df):
        self.net.eval()
        X = X_df.values.astype(np.float32)
        with torch.no_grad():
            X_t = torch.from_numpy(X).to(self.device)
            y_pred = self.net(X_t).cpu().numpy().ravel()
        return y_pred


# ----------------------------------------------------------
# 6) Entrenamiento con PyTorch (GPU si hay)
# ----------------------------------------------------------
def train_pytorch_model(df_features, test_days=60,
                        epochs=5, batch_size=4096, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

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

    X_train = train[feature_cols].values
    y_train = train['count'].values

    X_test = test[feature_cols].values
    y_test = test['count'].values

    train_dataset = CrimeDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    net = CrimeMLP(in_features=len(feature_cols)).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        net.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = net(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch}/{epochs} - Train MSE: {epoch_loss:.4f}")

    # Evaluar en test
    net.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
        y_pred_t = net(X_test_t).cpu().numpy().ravel()

    mae = mean_absolute_error(y_test, y_pred_t)
    mse = mean_squared_error(y_test, y_pred_t)
    rmse = np.sqrt(mse)

    print("Evaluación MLP PyTorch (últimos días como test):")
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")

    wrapper = TorchRegressorWrapper(net, device)
    return wrapper, feature_cols, df_features


# ----------------------------------------------------------
# 7) Predicción día siguiente (reutilizamos la lógica)
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
# 8) Main
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

    # Entrenar modelo PyTorch (GPU si hay)
    model, feature_cols, df_feat_trained = train_pytorch_model(
        df_feat,
        test_days=60,
        epochs=5,
        batch_size=4096,
        lr=1e-3
    )

    next_day_pred = predict_next_day(model, feature_cols, df_feat_trained)
    print("\nEjemplo de predicción para el día siguiente:")
    print(next_day_pred.head(20))

    next_day_pred.to_csv(
        "./dataset/predicciones_siguiente_dia_pytorch.csv", index=False
    )
    print("\nPredicciones guardadas en ./dataset/predicciones_siguiente_dia_pytorch.csv")


if __name__ == "__main__":
    main()
