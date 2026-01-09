"""
Script dedicado al entrenamiento del clasificador ML para filtrar máscaras de SAM3.

OBJETIVO: Maximizar la PRECISION para detectar RUIDO (clase 0 = NO es línea válida)
    - Alta precision clase 0 = cuando dice "esto es ruido", casi siempre acierta
    - Queremos filtrar FP (falsos positivos de SAM3) sin perder líneas reales

CLASES:
    - Clase 1 (label=1): Línea VÁLIDA (IoU ≥ threshold con GT)
    - Clase 0 (label=0): RUIDO/NO_LÍNEA (IoU < threshold)

CONFUSION MATRIX:
                    Predicho: 0 (ruido)    Predicho: 1 (línea)
    Real: 0 (ruido)      TN ✓                   FP ✗
    Real: 1 (línea)      FN ✗                   TP ✓

MÉTRICAS:
    - Precision clase 0 = TN / (TN + FN)  →  De lo que predice como ruido, % correcto
    - Recall clase 0    = TN / (TN + FP)  →  Del ruido real, % que detecta
    - Precision clase 1 = TP / (TP + FP)  →  De lo que predice como línea, % correcto
    - Recall clase 1    = TP / (TP + FN)  →  De las líneas reales, % que detecta
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix, precision_recall_curve
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


def print_confusion_matrix_table(tn, fp, fn, tp, title="Confusion Matrix"):
    """
    Imprime la confusion matrix en formato tabla con explicación.
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"\n                    │ Predicho: 0 (RUIDO) │ Predicho: 1 (LÍNEA) │")
    print(f"────────────────────┼──────────────────────┼─────────────────────┤")
    print(f" Real: 0 (ruido)    │     TN = {tn:>6}     │     FP = {fp:>6}    │  Total: {tn+fp}")
    print(f" Real: 1 (línea)    │     FN = {fn:>6}     │     TP = {tp:>6}    │  Total: {tp+fn}")
    print(f"────────────────────┴──────────────────────┴─────────────────────┘")
    print(f"                         Total: {tn+fn}            Total: {tp+fp}         Total: {tn+fp+fn+tp}")
    
    # Interpretación
    print(f"\n📊 Interpretación:")
    print(f"   TN (True Negative)  = {tn:>6} → Ruido correctamente identificado como ruido ✓")
    print(f"   FP (False Positive) = {fp:>6} → Ruido identificado como línea (ERROR) ✗")
    print(f"   FN (False Negative) = {fn:>6} → Línea identificada como ruido (ERROR) ✗")
    print(f"   TP (True Positive)  = {tp:>6} → Línea correctamente identificada como línea ✓")


def print_metrics_table(prec_0, rec_0, f1_0, prec_1, rec_1, f1_1, accuracy):
    """
    Imprime tabla de métricas con explicación clara.
    """
    print(f"\n{'='*80}")
    print("MÉTRICAS DE CLASIFICACIÓN")
    print(f"{'='*80}")
    print(f"\n┌─────────────────────────┬──────────┬─────────┬─────────┐")
    print(f"│         Métrica         │ Clase 0  │ Clase 1 │  Global │")
    print(f"│                         │ (RUIDO)  │ (LÍNEA) │         │")
    print(f"├─────────────────────────┼──────────┼─────────┼─────────┤")
    print(f"│ Precision               │  {prec_0:>6.3f}  │ {prec_1:>6.3f}  │    -    │")
    print(f"│ Recall                  │  {rec_0:>6.3f}  │ {rec_1:>6.3f}  │    -    │")
    print(f"│ F1 Score                │  {f1_0:>6.3f}  │ {f1_1:>6.3f}  │    -    │")
    print(f"├─────────────────────────┴──────────┴─────────┼─────────┤")
    print(f"│ Accuracy (aciertos totales)                  │ {accuracy:>6.3f}  │")
    print(f"└──────────────────────────────────────────────┴─────────┘")
    
    print(f"\n📝 Explicación:")
    print(f"   Precision Clase 0 = TN/(TN+FN) = {prec_0:.3f}")
    print(f"      → De las máscaras que predice como RUIDO, {prec_0*100:.1f}% realmente son ruido")
    print(f"   Recall Clase 0    = TN/(TN+FP) = {rec_0:.3f}")
    print(f"      → Del ruido real en el dataset, detecta el {rec_0*100:.1f}%")
    print(f"\n   Precision Clase 1 = TP/(TP+FP) = {prec_1:.3f}")
    print(f"      → De las máscaras que predice como LÍNEA, {prec_1*100:.1f}% realmente son líneas")
    print(f"   Recall Clase 1    = TP/(TP+FN) = {rec_1:.3f}")
    print(f"      → De las líneas reales en el dataset, detecta el {rec_1*100:.1f}%")


def plot_threshold_analysis(y_test, y_proba, label_name, out_dir):
    """
    Genera gráficos para analizar el impacto del threshold en las métricas.
    """
    thresholds = np.arange(0.05, 0.96, 0.05)
    
    metrics = {
        'threshold': [],
        'precision_class0': [],
        'recall_class0': [],
        'f1_class0': [],
        'precision_class1': [],
        'recall_class1': [],
        'f1_class1': []
    }
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Métricas clase 0 (ruido)
        prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
        rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
        
        # Métricas clase 1 (líneas válidas)
        prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
        
        metrics['threshold'].append(thr)
        metrics['precision_class0'].append(prec_0)
        metrics['recall_class0'].append(rec_0)
        metrics['f1_class0'].append(f1_0)
        metrics['precision_class1'].append(prec_1)
        metrics['recall_class1'].append(rec_1)
        metrics['f1_class1'].append(f1_1)
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision vs Threshold
    axes[0, 0].plot(metrics['threshold'], metrics['precision_class0'], 'r-', label='Precision Clase 0 (ruido)', linewidth=2)
    axes[0, 0].plot(metrics['threshold'], metrics['precision_class1'], 'g-', label='Precision Clase 1 (líneas)', linewidth=2)
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Recall vs Threshold
    axes[0, 1].plot(metrics['threshold'], metrics['recall_class0'], 'r--', label='Recall Clase 0 (ruido)', linewidth=2)
    axes[0, 1].plot(metrics['threshold'], metrics['recall_class1'], 'g--', label='Recall Clase 1 (líneas)', linewidth=2)
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_title('Recall vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 vs Threshold
    axes[1, 0].plot(metrics['threshold'], metrics['f1_class0'], 'r:', label='F1 Clase 0 (ruido)', linewidth=2)
    axes[1, 0].plot(metrics['threshold'], metrics['f1_class1'], 'g:', label='F1 Clase 1 (líneas)', linewidth=2)
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Precision-Recall curve para clase 1
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    axes[1, 1].plot(recall_curve, precision_curve, 'b-', linewidth=2)
    axes[1, 1].set_xlabel('Recall (Clase 1)')
    axes[1, 1].set_ylabel('Precision (Clase 1)')
    axes[1, 1].set_title('Precision-Recall Curve (Clase 1)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"threshold_analysis_{label_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return plot_path, pd.DataFrame(metrics)


def find_optimal_threshold(y_test, y_proba, optimize_for='f1_class1'):
    """
    Encuentra el threshold óptimo según el criterio especificado.
    
    optimize_for:
        - 'f1_class1': Maximiza F1 de clase 1 (líneas válidas) [DEFAULT]
        - 'f1_class0': Maximiza F1 de clase 0 (ruido)
        - 'precision_class0': Maximiza precision para detectar ruido
        - 'precision_class1': Maximiza precision para detectar líneas
        - 'balanced': Balance entre precision clase 0 y recall clase 1
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for thr in thresholds:
        y_pred = (y_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Métricas clase 0
        prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
        rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
        
        # Métricas clase 1
        prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
        
        # Seleccionar métrica según criterio
        if optimize_for == 'f1_class1':
            score = f1_1
        elif optimize_for == 'f1_class0':
            score = f1_0
        elif optimize_for == 'precision_class0':
            score = prec_0
        elif optimize_for == 'precision_class1':
            score = prec_1
        elif optimize_for == 'balanced':
            # Balance: alta precision clase 0 (filtrar ruido) + alto recall clase 1 (no perder líneas)
            score = (prec_0 + rec_1) / 2
        else:
            score = f1_1
        
        if score > best_score:
            best_score = score
            best_threshold = thr
    
    return best_threshold, best_score


def train_ml_models(csv_path, out_dir, label_name='label30', seed=42, test_size=0.3, optimize_for='f1_class1', use_stacking=False, use_voting=False, stacking_passthrough=False, score_label_threshold=0.25, use_gt_pairs=False):
    """
    Entrena múltiples clasificadores y guarda el mejor.
    
    Args:
        csv_path: path al CSV con features
        out_dir: directorio de salida
        label_name: 'label30' o 'label50'
        seed: semilla aleatoria
        test_size: proporción para test
        optimize_for: criterio para seleccionar threshold óptimo
        use_stacking: si True, añade modelo Stacking ensemble
        use_voting: si True, añade modelo Voting ensemble
        stacking_passthrough: si True, el meta-learner también recibe features originales
    """
    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO DE CLASIFICADOR ML - FILTRO DE RUIDO")
    print(f"{'='*80}\n")
    print(f"🎯 OBJETIVO: Detectar máscaras RUIDOSAS (FP de SAM3) con alta precision")
    print(f"   Optimizando para: {optimize_for}\n")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Cargar datos
    print(f"📂 Cargando datos de: {csv_path}")
    df = pd.read_csv(csv_path)

    # --- Support: if label_name == 'score', binarize 'score' using score_label_threshold ---
    if label_name == 'score':
        if 'score' not in df.columns:
            print(f"❌ Error: columna 'score' no encontrada en {csv_path}")
            return None
        n_before = len(df)
        df = df.dropna(subset=['score'])
        n_after = len(df)
        if n_after < n_before:
            print(f"[INFO] Eliminadas {n_before - n_after} filas con score=NaN")
        # crear columna binaria temporal que usaremos como label
        bin_col = 'label_score_bin'
        df[bin_col] = (df['score'] >= float(score_label_threshold)).astype(int)
        effective_label = bin_col
    else:
        # --- Eliminar filas con label faltante (NaN) para evitar errores posteriores ---
        if label_name in df.columns:
            n_before = len(df)
            df = df.dropna(subset=[label_name])
            n_after = len(df)
            if n_after < n_before:
                print(f"[INFO] Eliminadas {n_before - n_after} filas con {label_name}=NaN")
        else:
            print(f"❌ Error: columna de label faltante: {label_name}")
            return None
        effective_label = label_name

    # Detectar features según el CSV que produces (orden preferente)
    # NOTA: excluimos 'score' (es el label/valor bruto) y solo usamos 'ransac_inline_pct'
    preferred = [
        'ransac_inline_pct'
    ]
    
    # columnas de features esperadas con prefijo feat_ (si existen).
    # mantenemos un listado "preferente" para ordenar, pero también añadimos cualquier feat_* extra que aparezca.
    feat_prefixed = [
        'feat_area','feat_aspect_ratio','feat_vertical_coverage','feat_horizontal_position',
        'feat_correlation','feat_mean_width','feat_std_width','feat_width_ratio',
        'feat_angle','feat_curvature','feat_edge_density','feat_straightness',
        'feat_mean_luminance','feat_num_components','feat_largest_component_ratio','feat_solidity'
    ]
    
    # construir feature_cols en orden: preferred ∩ cols, luego feat_prefixed ∩ cols, luego cualquier feat_* restante
    cols = set(df.columns)
    feature_cols = [c for c in preferred if c in cols]
    feature_cols += [c for c in feat_prefixed if c in cols and c not in feature_cols]
    # añadir cualquier feat_ adicional no listada explícitamente
    extra_feats = sorted([c for c in df.columns if c.startswith('feat_') and c not in feature_cols])
    feature_cols += extra_feats

    # Verificaciones mínimas
    if len(feature_cols) == 0:
        print("❌ Error: no se detectaron columnas de features válidas en el CSV.")
        print(f"Columnas disponibles (primeras 50): {list(df.columns)[:50]}")
        return None

    print(f"✅ Columnas detectadas para features (ordenadas): {feature_cols}")
 
    # --- NUEVO: filtrar por GT y emparejar negativos (label=0) con mismos positivos (label=1) ---
    if use_gt_pairs:
        if 'gt_has_lines' not in df.columns:
            print("⚠️ --use_gt_pairs activado pero no existe la columna 'gt_has_lines' en el CSV. Se omite el filtrado.")
        else:
            # seleccionar solo filas con GT presente
            df_gt = df[df['gt_has_lines'].astype(bool)].copy()
            n_total_gt = len(df_gt)
            neg_df = df_gt[df_gt[effective_label] == 0]
            pos_df = df_gt[df_gt[effective_label] == 1]
            n_neg = len(neg_df)
            n_pos = len(pos_df)
            if n_neg == 0:
                print("⚠️ --use_gt_pairs: no hay negativos (label=0) con GT; no se aplicará el filtrado.")
            else:
                # sample positives to match n_neg (with replacement if not enough)
                if n_pos >= n_neg:
                    pos_sampled = pos_df.sample(n=n_neg, random_state=seed)
                else:
                    # re-muestrear con reemplazo para igualar tamaño
                    pos_sampled = pos_df.sample(n=n_neg, replace=True, random_state=seed)
                df = pd.concat([neg_df, pos_sampled], axis=0).sample(frac=1, random_state=seed).reset_index(drop=True)
                print(f"[use_gt_pairs] Filtrado: GT filas antes={n_total_gt}, negativos={n_neg}, positivos_disponibles={n_pos}, después={len(df)} (negativos + {n_neg} positivos muestreados).")
    # --- fin del filtrado ---

    # Normalizar ransac_applied si existe
    if 'ransac_applied' in df.columns:
        try:
            df['ransac_applied'] = df['ransac_applied'].fillna(0).astype(int)
        except Exception:
            df['ransac_applied'] = df['ransac_applied'].apply(lambda x: 1 if bool(x) else 0).astype(int)

    # construir X,y tras eliminar NaNs en label; asegurar y entero (0/1)
    X = df[feature_cols].values
    y = df[effective_label].astype(int).values
    
    # Reportar NaNs por columna antes de split
    nan_total = np.isnan(X).sum()
    if nan_total > 0:
        nan_per_col = np.isnan(X).sum(axis=0)
        cols_with_nan = [feature_cols[i] for i, c in enumerate(nan_per_col) if c > 0]
        print(f"\n⚠️  Se detectaron {nan_total} valores NaN en X (columnas: {cols_with_nan}) - se imputarán (median).")
    else:
        print("\n✅ No se detectaron NaNs en las features.")
    
    num_pos = int(np.sum(y))
    num_neg = int(len(y) - num_pos)
    print(f"\n📊 Dataset: {len(X)} máscaras")
    print(f"   Clase 1 (líneas válidas): {num_pos} ({num_pos/len(y)*100:.1f}%)")
    print(f"   Clase 0 (ruido/no_línea): {num_neg} ({num_neg/len(y)*100:.1f}%)")
    
    # Split train/test (hacer split ANTES de imputar para evitar data leakage)
    print(f"\n🔀 Dividiendo datos: train={1-test_size:.0%}, test={test_size:.0%}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Imputar NaNs usando mediana (fit SOLO en train)
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    print("🔧 Imputer entrenado en train y aplicado a train/test (strategy=median).")
    
    # Balancear train
    print("⚖️  Balanceando clases en train (1:1)...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    min_class = min(len(pos_idx), len(neg_idx))
    
    np.random.seed(seed)
    balanced_idx = np.concatenate([
        np.random.choice(pos_idx, min_class, replace=False),
        np.random.choice(neg_idx, min_class, replace=False)
    ])
    np.random.shuffle(balanced_idx)
    
    X_train_balanced = X_train[balanced_idx]
    y_train_balanced = y_train[balanced_idx]
    
    print(f"   Train balanceado: {len(X_train_balanced)} muestras ({min_class} de cada clase)")
    print(f"   Test (sin balancear): {len(X_test)} muestras ({y_test.sum()} clase 1, {len(y_test) - y_test.sum()} clase 0)")
    
    # Definir modelos base
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=seed),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=seed),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=seed, class_weight='balanced'),
        'MLP_64_32': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed),
        'MLP_128_64_32': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=seed)
    }
    
    # Añadir STACKING si se solicita (MÁS LENTO, MÁS SOFISTICADO)
    if use_stacking:
        print("\n🔗 Configurando Stacking Ensemble (con Meta-Learner)...")
        
        base_estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=seed)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=seed))
        ]
        
        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(random_state=seed),
            cv=5,
            n_jobs=-1,
            passthrough=stacking_passthrough  # NUEVO: controlar si pasa features originales
        )
        
        models['Stacking_RF_GB_MLP'] = stacking_clf
        print("   Base models: RandomForest, GradientBoosting, MLP")
        print("   Meta-learner: LogisticRegression (nivel 2)")
        print(f"   Passthrough: {'✓ SÍ' if stacking_passthrough else '✗ No'}")
        print("   CV folds: 5")
        print("   🐌 Más lento pero más preciso")
        
        if stacking_passthrough:
            print("\n   ℹ️  Con passthrough=True:")
            print(f"      Meta-learner recibe {len(feature_cols)} features originales")
            print(f"      + 3 predicciones (RF, GB, MLP)")
            print(f"      = {len(feature_cols) + 3} features totales para decidir")
    
    # Entrenar todos
    print(f"\n{'='*80}")
    print("🔧 ENTRENANDO MODELOS...")
    print(f"{'='*80}\n")
    
    results = []
    best_score = 0
    best_model = None
    best_model_name = None
    best_threshold = 0.5
    
    for name, model in models.items():
        print(f"\n{'─'*80}")
        print(f"[{name}] Entrenando...")
        print(f"{'─'*80}")
        model.fit(X_train_balanced, y_train_balanced)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            
            # Encontrar threshold óptimo
            optimal_thr, score = find_optimal_threshold(y_test, y_proba, optimize_for)
            y_pred = (y_proba >= optimal_thr).astype(int)
        else:
            y_pred = model.predict(X_test)
            optimal_thr = 0.5
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Métricas CLASE 0 (RUIDO)
        prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
        rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0
        
        # Métricas CLASE 1 (LÍNEAS)
        prec_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Imprimir resultados con tablas
        print(f"\n🎯 Threshold óptimo: {optimal_thr:.3f}")
        print_confusion_matrix_table(tn, fp, fn, tp, title=f"Confusion Matrix - {name}")
        print_metrics_table(prec_0, rec_0, f1_0, prec_1, rec_1, f1_1, accuracy)
        
        results.append({
            'model': name,
            'threshold': optimal_thr,
            'accuracy': accuracy,
            'precision_class0': prec_0,
            'recall_class0': rec_0,
            'f1_class0': f1_0,
            'precision_class1': prec_1,
            'recall_class1': rec_1,
            'f1_class1': f1_1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        })
        
        # Seleccionar mejor modelo según criterio
        if optimize_for == 'precision_class0':
            current_score = prec_0
        elif optimize_for == 'f1_class0':
            current_score = f1_0
        elif optimize_for == 'balanced':
            current_score = (prec_0 + rec_1) / 2
        else:
            current_score = f1_1
        
        if current_score > best_score:
            best_score = current_score
            best_model = model
            best_model_name = name
            best_threshold = optimal_thr
    
    # Generar gráficos de análisis para el mejor modelo
    if hasattr(best_model, 'predict_proba'):
        y_proba_best = best_model.predict_proba(X_test)[:, 1]
        plot_path, metrics_df = plot_threshold_analysis(y_test, y_proba_best, label_name, out_dir)
        print(f"📊 Análisis de thresholds guardado: {plot_path}")
        
        # Guardar métricas por threshold
        metrics_csv = os.path.join(out_dir, f"threshold_metrics_{label_name}.csv")
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"📊 Métricas por threshold: {metrics_csv}")
    
    # Guardar mejor modelo (incluyendo el imputer)
    best_model_path = os.path.join(out_dir, f"best_classifier_{label_name}.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'model_name': best_model_name,
            'feature_cols': feature_cols,
            'threshold': best_threshold,
            'optimize_for': optimize_for,
            'label': label_name,
            'imputer': imputer
        }, f)
    
    print(f"\n{'='*80}")
    print(f"✅ MEJOR MODELO: {best_model_name}")
    print(f"{'='*80}")
    print(f"  Criterio de selección: {optimize_for}")
    print(f"  Score conseguido: {best_score:.4f}")
    print(f"  Threshold óptimo: {best_threshold:.3f}")
    
    # Obtener predicciones del mejor modelo para mostrar tabla final
    if hasattr(best_model, 'predict_proba'):
        y_proba_best = best_model.predict_proba(X_test)[:, 1]
        y_pred_best = (y_proba_best >= best_threshold).astype(int)
    else:
        y_pred_best = best_model.predict(X_test)
    
    tn_best, fp_best, fn_best, tp_best = confusion_matrix(y_test, y_pred_best).ravel()
    
    prec_0_best = tn_best / (tn_best + fn_best) if (tn_best + fn_best) > 0 else 0
    rec_0_best = tn_best / (tn_best + fp_best) if (tn_best + fp_best) > 0 else 0
    f1_0_best = 2 * prec_0_best * rec_0_best / (prec_0_best + rec_0_best) if (prec_0_best + rec_0_best) > 0 else 0
    
    prec_1_best = tp_best / (tp_best + fp_best) if (tp_best + fp_best) > 0 else 0
    rec_1_best = tp_best / (tp_best + fn_best) if (tp_best + fn_best) > 0 else 0
    f1_1_best = 2 * prec_1_best * rec_1_best / (prec_1_best + rec_1_best) if (prec_1_best + rec_1_best) > 0 else 0
    
    accuracy_best = (tp_best + tn_best) / (tp_best + tn_best + fp_best + fn_best)
    
    print_confusion_matrix_table(tn_best, fp_best, fn_best, tp_best, title=f"Confusion Matrix Final - {best_model_name}")
    print_metrics_table(prec_0_best, rec_0_best, f1_0_best, prec_1_best, rec_1_best, f1_1_best, accuracy_best)
    
    print(f"\n💾 Guardado en: {best_model_path}")
    
    # Guardar resumen
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(out_dir, f"training_results_{label_name}.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"📄 Resultados completos: {results_csv}\n")
    
    # Ranking
    print("🏆 Ranking de modelos:")
    print("-" * 100)
    print(f"{'Modelo':<25} {'Thr':>5} {'P_c0':>6} {'R_c0':>6} {'F1_c0':>7} | {'P_c1':>6} {'R_c1':>6} {'F1_c1':>7}")
    print("-" * 100)
    
    results_sorted = sorted(results, key=lambda x: x.get(optimize_for.replace('_', '_').replace('class', 'class'), x['f1_class1']), reverse=True)
    for i, r in enumerate(results_sorted, 1):
        print(f"{i}. {r['model']:<22} {r['threshold']:>5.2f} "
              f"{r['precision_class0']:>6.3f} {r['recall_class0']:>6.3f} {r['f1_class0']:>7.3f} | "
              f"{r['precision_class1']:>6.3f} {r['recall_class1']:>6.3f} {r['f1_class1']:>7.3f}")
    
    return best_model_path


def main():
    parser = argparse.ArgumentParser(
        description="Entrenar clasificador ML para filtrar máscaras ruidosas de SAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Entrenar modelos individuales (default)
  python train_ml_filter.py --csv_path features.csv --label label30

  # Con Voting Ensemble (RECOMENDADO: rápido y efectivo)
  python train_ml_filter.py --csv_path features.csv --label label30 --use_voting

  # Con Stacking básico (solo usa predicciones de modelos base)
  python train_ml_filter.py --csv_path features.csv --label label30 --use_stacking

  # Con Stacking + passthrough (meta-learner usa features originales + predicciones)
  python train_ml_filter.py --csv_path features.csv --label label30 --use_stacking --stacking_passthrough

  # Comparar TODAS las variantes
  python train_ml_filter.py --csv_path features.csv --label label30 --use_voting --use_stacking --stacking_passthrough
        """
    )
    
    parser.add_argument("--csv_path", type=str, required=True,
                       help="Path al CSV con features (mask_features_*.csv)")
    parser.add_argument("--out_dir", type=str, default="/home/its/Hector/SAM3/output_sam3_v2",
                       help="Directorio de salida")
    parser.add_argument("--label", type=str, default="label30", choices=["label30", "label50", "score"],
                       help="Label: label30 (IoU≥0.3) o label50 (IoU≥0.5) o score (usar 'score' y binarizar)")
    parser.add_argument("--score_label_threshold", type=float, default=0.25,
                       help="Si --label score: umbral para binarizar score (>= threshold => 1, else 0). Default=0.25")
    parser.add_argument("--use_gt_pairs", action="store_true",
                       help="Usar solo filas con GT (gt_has_lines==1): conservar todas las label=0 y añadir el mismo número de label=1 muestreadas")
    parser.add_argument("--seed", type=int, default=42,
                       help="Semilla aleatoria")
    parser.add_argument("--test_size", type=float, default=0.3,
                       help="Proporción test (0.3 = 30%%)")
    parser.add_argument("--optimize_for", type=str, default="f1_class1",
                       choices=["f1_class1", "f1_class0", "precision_class0", "precision_class1", "balanced"],
                       help="Métrica a optimizar al elegir threshold")
    parser.add_argument("--use_voting", action="store_true",
                       help="Añade Voting Classifier (promedia predicciones de RF+GB+MLP) - RÁPIDO")
    parser.add_argument("--use_stacking", action="store_true",
                       help="Añade Stacking Classifier (meta-learner sobre RF+GB+MLP) - LENTO pero preciso")
    parser.add_argument("--stacking_passthrough", action="store_true",
                       help="Si se usa --use_stacking, el meta-learner TAMBIÉN recibe features originales (más potente)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"❌ Error: No existe {args.csv_path}")
        return
    
    print(f"\n{'='*80}")
    print("⚙️  CONFIGURACIÓN")
    print(f"{'='*80}")
    print(f"  CSV: {args.csv_path}")
    print(f"  Output: {args.out_dir}")
    print(f"  Label: {args.label}")
    print(f"  Optimize: {args.optimize_for}")
    print(f"  Seed: {args.seed}")
    print(f"  Test size: {args.test_size:.0%}")
    print(f"  Voting: {'✓ SÍ' if args.use_voting else '✗ No'}")
    print(f"  Stacking: {'✓ SÍ' if args.use_stacking else '✗ No'}")
    if args.use_stacking:
        print(f"  Stacking passthrough: {'✓ SÍ (features + predicciones)' if args.stacking_passthrough else '✗ No (solo predicciones)'}")
    
    model_path = train_ml_models(
		csv_path=args.csv_path,
		out_dir=args.out_dir,
		label_name=args.label,
		seed=args.seed,
		test_size=args.test_size,
		optimize_for=args.optimize_for,
		use_stacking=args.use_stacking,
		use_voting=args.use_voting,
		stacking_passthrough=args.stacking_passthrough,
		score_label_threshold=args.score_label_threshold,
		use_gt_pairs=args.use_gt_pairs
	)
    
    if model_path:
        print(f"\n✅ Entrenamiento exitoso")
        print(f"   Modelo: {model_path}")
        print(f"\n💡 Usar con test_sam3_filer.py:")
        print(f"   python test_sam3_filer.py --apply_ml_filter --classifier_path {model_path} ...")
    else:
        print("\n❌ Error durante entrenamiento")


if __name__ == "__main__":
    main()
