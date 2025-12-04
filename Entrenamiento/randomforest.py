import matplotlib.pyplot as plt
import os
import re

import numpy as np
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

import matplotlib
matplotlib.use("Agg")  # para que funcione en la Orin sin GUI


# -------------------------------------------------------------
# 1) Cargar dataset
# -------------------------------------------------------------
def load_data(path_csv="./dataset/dataset_procesado.csv",
              max_years=4,
              cut_2024=True):
    """
    Carga el CSV procesado con columnas:
    DATE OCC, TIME OCC, LAT, LON, crime_category
    """

    df = pd.read_csv(path_csv, parse_dates=["DATE OCC"])

    # Solo columnas que necesitamos para ahorrar RAM
    df = df[["DATE OCC", "TIME OCC", "LAT", "LON", "crime_category"]].copy()

    # Fecha diaria (sin hora)
    df["date"] = df["DATE OCC"].dt.normalize()

    # (Opcional) cortar al periodo estable del dataset
    if cut_2024:
        df = df[df["date"] < pd.Timestamp("2024-03-07")].copy()

    # (Opcional) limitar a últimos N años para que sea aún más ligero
    if max_years is not None:
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=365 * max_years)
        df = df[df["date"] >= cutoff].copy()
        print(f"Usando datos desde {cutoff.date()} hasta {max_date.date()}")

    print("Shape datos crudos:", df.shape)
    return df


# -------------------------------------------------------------
# 2) Definir zonas espaciales con K-Means ligero
# -------------------------------------------------------------
def fit_zones(df, n_zones=40, random_state=0, max_points=200_000):
    """
    Crea 'n_zones' zonas espaciales usando LAT/LON.
    Usa MiniBatchKMeans para ahorrar memoria y tiempo.
    """

    coords = df[["LAT", "LON"]].values

    # Submuestreo para entrenar el k-means si hay demasiados puntos
    if coords.shape[0] > max_points:
        idx = np.random.choice(coords.shape[0], max_points, replace=False)
        coords_sample = coords[idx]
    else:
        coords_sample = coords

    print(f"Entrenando K-Means con {coords_sample.shape[0]} puntos y "
          f"{n_zones} zonas...")

    kmeans = MiniBatchKMeans(
        n_clusters=n_zones,
        batch_size=2048,
        random_state=random_state
    )
    kmeans.fit(coords_sample)

    # Asignar zona a cada delito
    df["zone"] = kmeans.predict(coords)

    # Centros de cada zona (para mapa después)
    centers = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=["zone_lat", "zone_lon"]
    )
    centers["zone"] = centers.index

    print("Zonas creadas. Ejemplo de centros:")
    print(centers.head())

    return df, kmeans, centers


# -------------------------------------------------------------
# 3) Agregar por día / zona / categoría
# -------------------------------------------------------------
def build_daily_counts(df):
    """
    Devuelve un DataFrame con columnas:
    crime_category, zone, date, count
    """
    daily = (
        df.groupby(["crime_category", "zone", "date"])
        .size()
        .reset_index(name="count")
    )
    print("Shape daily (solo días con >=1 crimen):", daily.shape)
    return daily


# -------------------------------------------------------------
# 4) Serie temporal día × zona para UNA categoría
# -------------------------------------------------------------
def make_supervised_for_category(daily_cat):
    """
    daily_cat: columnas ['crime_category','zone','date','count'] filtrado
               a una categoría concreta.
    Devuelve un DataFrame con:
    date, zone, count, dow, month, lag1, lag7, roll7
    (rellenando días sin crímenes con 0).
    """

    cat = daily_cat["crime_category"].iloc[0]
    print(f"  Preparando serie temporal para categoría: {cat}")

    all_dates = pd.date_range(
        daily_cat["date"].min(),
        daily_cat["date"].max(),
        freq="D"
    )

    pieces = []
    # Para cada zona rellenamos todos los días
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

    # Quitamos los primeros días sin lag1
    full = full.dropna(subset=["lag1"]).copy()

    # Reducir tipos para ahorrar RAM
    for col in ["count", "lag1", "lag7", "roll7"]:
        full[col] = full[col].astype("float32")
    full["zone"] = full["zone"].astype("int16")
    full["dow"] = full["dow"].astype("int8")
    full["month"] = full["month"].astype("int8")

    print("  Shape serie completa categoría:", full.shape)
    return full


# -------------------------------------------------------------
# 5) Tuning + entrenamiento RandomForest para UNA categoría
# -------------------------------------------------------------
def train_category_model(full_cat,
                         category,
                         param_grid,
                         split=(0.7, 0.2, 0.1)):
    """
    full_cat: DataFrame con columnas:
      date, zone, count, dow, month, lag1, lag7, roll7

    split: (train, val, test) en proporciones (suma 1.0).
    Se respeta el orden temporal.
    """

    full_cat = full_cat.sort_values("date").reset_index(drop=True)

    n = len(full_cat)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    n_test = n - n_train - n_val

    train = full_cat.iloc[:n_train]
    val = full_cat.iloc[n_train:n_train + n_val]
    test = full_cat.iloc[n_train + n_val:]

    print(f"  Split temporal: train={len(train)}, val={len(val)}, "
          f"test={len(test)}")

    feature_cols = ["lag1", "lag7", "roll7", "dow", "month", "zone"]

    results = []

    # ------------------ BÚSQUEDA DE HIPERPARÁMETROS ------------------
    for params in param_grid:
        n_estimators = params["n_estimators"]
        max_depth = params["max_depth"]

        print(f"    Probando n_estimators={n_estimators}, "
              f"max_depth={max_depth} ...")

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=4,
            random_state=0,
        )
        model.fit(train[feature_cols], train["count"])

        pred_val = model.predict(val[feature_cols])
        mae_val = mean_absolute_error(val["count"], pred_val)
        mse_val = mean_squared_error(val["count"], pred_val)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(val["count"], pred_val)

        results.append({
            "crime_category": category,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "mae_val": mae_val,
            "rmse_val": rmse_val,
            "r2_val": r2_val,
        })

        print(f"      -> MAE_val={mae_val:.3f}, "
              f"RMSE_val={rmse_val:.3f}, R2_val={r2_val:.3f}")

    results_df = pd.DataFrame(results)

    # Elegimos el mejor por RMSE de validación (más bajo)
    best_row = results_df.sort_values("rmse_val").iloc[0]
    best_params = {
        "n_estimators": int(best_row["n_estimators"]),
        "max_depth": int(best_row["max_depth"]),
    }

    print(f"  Mejor combinación según validación: "
          f"n_estimators={best_params['n_estimators']}, "
          f"max_depth={best_params['max_depth']}, "
          f"RMSE_val={best_row['rmse_val']:.3f}, "
          f"MAE_val={best_row['mae_val']:.3f}, "
          f"R2_val={best_row['r2_val']:.3f}")

    # ------------------ RE-ENTRENAR CON TRAIN+VAL ------------------
    train_val = pd.concat([train, val], ignore_index=True)

    best_model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        n_jobs=4,
        random_state=0,
    )
    best_model.fit(train_val[feature_cols], train_val["count"])

    # Evaluar en TEST (10% final)
    pred_test = best_model.predict(test[feature_cols])
    mae_test = mean_absolute_error(test["count"], pred_test)
    mse_test = mean_squared_error(test["count"], pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(test["count"], pred_test)

    print(f"  Rendimiento en TEST (10% final): "
          f"MAE={mae_test:.3f}, RMSE={rmse_test:.3f}, R2={r2_test:.3f}")

    test_metrics = {
        "mae_test": mae_test,
        "rmse_test": rmse_test,
        "r2_test": r2_test,
    }

    return best_model, feature_cols, full_cat, results_df, test_metrics


# -------------------------------------------------------------
# 5b) Gráfica de resultados de tuning
# -------------------------------------------------------------
def plot_tuning_results(results_df, category, out_dir="./graficas"):
    """
    Dibuja RMSE de validación vs n_estimators para cada max_depth.
    Guarda la imagen como PNG.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))

    depths = sorted(results_df["max_depth"].unique())
    for d in depths:
        sub = results_df[results_df["max_depth"] == d] \
            .sort_values("n_estimators")
        plt.plot(
            sub["n_estimators"],
            sub["rmse_val"],
            marker="o",
            label=f"max_depth={d}"
        )

    plt.xlabel("n_estimators")
    plt.ylabel("RMSE validación")
    plt.title(f"Tuning RandomForest - {category}")
    plt.grid(alpha=0.3)
    plt.legend()

    cat_safe = re.sub(r"[^A-Za-z0-9_]+", "_", category)
    filepath = os.path.join(out_dir, f"rf_tuning_{cat_safe}.png")

    plt.tight_layout()
    plt.savefig(filepath, dpi=120)
    plt.close()

    print(f"  Gráfica de tuning guardada en: {filepath}")


# -------------------------------------------------------------
# 6) Predecir el día siguiente para UNA categoría
# -------------------------------------------------------------
def predict_next_day_category(model, feature_cols, full_cat):
    full_cat = full_cat.sort_values(["zone", "date"])
    last_date = full_cat["date"].max()
    next_date = last_date + pd.Timedelta(days=1)

    rows = []
    for zone, g in full_cat.groupby("zone"):
        g = g.sort_values("date")
        last = g.iloc[-1]

        lag1 = last["count"]
        # Si no hay 7 días usamos el último valor
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
    X_next = df_next[feature_cols]
    df_next["pred_count"] = model.predict(X_next)
    return df_next


# -------------------------------------------------------------
# 7) Main: pipeline completo
# -------------------------------------------------------------
def main():
    os.makedirs("./modelos", exist_ok=True)
    os.makedirs("./dataset", exist_ok=True)
    os.makedirs("./graficas", exist_ok=True)

    # 1) Cargar datos
    df = load_data()

    # 2) Zonas espaciales (ajusta n_zones según veas)
    df, kmeans, centers = fit_zones(df, n_zones=40)

    # 3) Daily counts por categoría y zona
    daily = build_daily_counts(df)

    categories = sorted(daily["crime_category"].unique())
    print("Categorías:", categories)

    all_next_preds = []
    models = {}
    tuning_results_all = []

    # Grid de hiperparámetros a probar
    param_grid = [
        {"n_estimators": 40, "max_depth": 8},
        {"n_estimators": 80, "max_depth": 8},
        {"n_estimators": 120, "max_depth": 8},
        {"n_estimators": 40, "max_depth": 12},
        {"n_estimators": 80, "max_depth": 12},
        {"n_estimators": 120, "max_depth": 12},
    ]

    for cat in categories:
        print("\n============================")
        print(f"Categoría: {cat}")
        print("============================")

        daily_cat = daily[daily["crime_category"] == cat].copy()
        full_cat = make_supervised_for_category(daily_cat)

        model, feature_cols, full_cat, results_cat, test_metrics = \
            train_category_model(full_cat, cat, param_grid)

        # Guardamos los resultados de tuning (con métricas de test duplicadas)
        results_cat = results_cat.copy()
        results_cat["mae_test"] = test_metrics["mae_test"]
        results_cat["rmse_test"] = test_metrics["rmse_test"]
        results_cat["r2_test"] = test_metrics["r2_test"]
        tuning_results_all.append(results_cat)

        # Gráfica de tuning
        plot_tuning_results(results_cat, cat, out_dir="./graficas")

        # Guardamos el mejor modelo y sus columnas de features
        models[cat] = {
            "model": model,
            "feature_cols": feature_cols
        }

        # Predicción para el día siguiente con el mejor modelo
        next_cat = predict_next_day_category(model, feature_cols, full_cat)
        next_cat["crime_category"] = cat
        all_next_preds.append(next_cat)

    # 4) Unir predicciones de todas las categorías
    next_all = pd.concat(all_next_preds, ignore_index=True)

    # Añadimos coordenadas de la zona para poder mapear
    next_all = next_all.merge(centers[["zone", "zone_lat", "zone_lon"]],
                              on="zone",
                              how="left")

    out_csv = "./dataset/prediccion_mapa_siguiente_dia_por_categoria.csv"
    next_all.to_csv(out_csv, index=False)
    print(f"\nPredicciones guardadas en: {out_csv}")

    # 5) Guardar modelos y k-means para reutilizar
    joblib.dump(kmeans, "./modelos/kmeans_zonas.joblib")
    joblib.dump(models, "./modelos/random_forests_por_categoria.joblib")
    print("Modelos guardados en carpeta ./modelos")

    # 6) Guardar resultados de tuning
    tuning_df = pd.concat(tuning_results_all, ignore_index=True)
    tuning_out = "./dataset/rf_tuning_resultados_por_categoria.csv"
    tuning_df.to_csv(tuning_out, index=False)
    print(f"Resultados de tuning guardados en: {tuning_out}")


if __name__ == "__main__":
    main()
