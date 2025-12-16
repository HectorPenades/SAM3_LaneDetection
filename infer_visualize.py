import os
import glob
import argparse
from datetime import datetime
import pickle

import torch
import numpy as np
from PIL import Image
import cv2
import pandas as pd
from skimage.morphology import skeletonize
from transformers import Sam3Processor, Sam3Model
from sklearn.linear_model import RANSACRegressor, LinearRegression


# --- Copia local de las utilidades necesarias (autónomo) ---
def load_image(path):
    return Image.open(path).convert("RGB")


def culane_lines_to_mask(lines_txt_path, img_w, img_h, thickness=5):
    """
    Convierte un fichero CULane *.lines.txt en una máscara binaria.
    Cada línea del .txt = una lane: x1 y1 x2 y2 x3 y3 ...
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if not os.path.exists(lines_txt_path):
        return mask

    with open(lines_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            coords = list(map(float, parts))
            pts = []
            for i in range(0, len(coords), 2):
                x, y = coords[i], coords[i + 1]
                if x < 0 or y < 0:  # CULane usa -2 como dummy a veces
                    continue
                pts.append((int(round(x)), int(round(y))))
            if len(pts) < 2:
                continue

            for p1, p2 in zip(pts[:-1], pts[1:]):
                cv2.line(mask, p1, p2, color=255, thickness=thickness)

    return mask


def compute_iou_with_dilation(pred_mask, gt_mask, dilation_radius=15):
    """
    Calcula IoU entre predicción y GT, ambas dilatadas por dilation_radius píxeles.
    
    IMPORTANTE: Solo compara en la región donde existe PREDICCIÓN.
    Así no penaliza si la predicción es un cachito pero está bien alineado.
    
    pred_mask: máscara predicha (0/255 o 0/1)
    gt_mask: máscara GT (0/255 o 0/1)
    dilation_radius: radio de dilatación a ambos lados (15 por defecto - estándar CULane)
    """
    # Binarizar
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    # Dilatar ambas máscaras
    if dilation_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_radius + 1, 2 * dilation_radius + 1))
        pred_dilated = cv2.dilate(pred, kernel)
        gt_dilated = cv2.dilate(gt, kernel)
    else:
        pred_dilated = pred
        gt_dilated = gt
    
    # Región válida = zona donde existe PREDICCIÓN dilatada
    # No penaliza por GT que no se detectó
    valid_region = pred_dilated.astype(np.uint8)
    
    # Intersección y unión SOLO en la región donde hay predicción
    inter = np.logical_and(pred_dilated, gt_dilated).sum()
    union = valid_region.sum()  # Solo píxeles donde hay predicción
    
    if union == 0:
        return 0.0
    
    return inter / union


def _mask_area(mask_np):
    """Área en píxeles de una máscara binaria (0/255 o 0/1)."""
    return int((mask_np > 0).sum())


def _mask_bbox_aspect(mask_np):
    """Aspect ratio (max(w/h, h/w)) de la bbox de la máscara."""
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0:
        return 0.0
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w = max(1, x2 - x1 + 1)
    h = max(1, y2 - y1 + 1)
    return max(w / h, h / w)


def _skeleton_length(mask_np):
    """Longitud (número de píxeles) del esqueleto de la máscara."""
    if mask_np.max() == 0:
        return 0
    skel = skeletonize(mask_np > 0)
    return int(skel.sum())


def filter_instance_masks(masks_tensor, scores, args):
    """
    Filtra máscaras por instancia (acepta tensores PyTorch).
    Devuelve lista de máscaras binarias (0/1 np.uint8) que pasan el filtro.
    """
    if masks_tensor is None or len(masks_tensor) == 0:
        return []
    kept = []
    for i in range(len(masks_tensor)):
        m = masks_tensor[i]
        # binarizar según umbral
        m_np = (m > args.mask_threshold).cpu().numpy().astype(np.uint8) if torch.is_tensor(m) else (np.array(m) > args.mask_threshold).astype(np.uint8)
        if _mask_area(m_np) < args.min_area:
            continue
        if _mask_bbox_aspect(m_np) < args.min_aspect_ratio:
            continue
        if _skeleton_length(m_np) < args.min_skel_length:
            continue
        if scores is not None and getattr(args, 'min_score', 0) > 0:
            try:
                score = float(scores[i].item()) if torch.is_tensor(scores[i]) else float(scores[i])
            except Exception:
                score = 0.0
            if score < args.min_score:
                continue
        kept.append(m_np)
    # resumen opcional corto (evitar mucho ruido aquí)
    return kept


def mask_to_centerline(mask_np):
    """
    Convierte una máscara binaria (np.uint8 0/255) en su línea central
    mediante skeletonize.

    Devuelve máscara 0/255 con grosor ~1 px.
    """
    if mask_np.max() == 0:
        return mask_np  # máscara vacía, no hay nada que esqueletonizar

    skel = skeletonize(mask_np > 0)
    return (skel.astype(np.uint8)) * 255


def dilate_mask(mask_np, radius=0):
    """
    (Opcional) Dilata la línea central para que tenga grosor similar al GT.
    """
    if radius <= 0:
        return mask_np
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask_np, kernel)
# --- fin utilidades locales ---

# --------------------------------------------------------
# compute_mask_features (copiado/adaptado de test_sam3_filer.py)
def compute_mask_features(mask_np, image_np=None):
    features = {}
    features['area'] = int((mask_np > 0).sum())
    if features['area'] == 0:
        return {k: 0.0 for k in ['correlation', 'mean_width', 'std_width', 'width_ratio',
                                  'vertical_coverage', 'area', 'angle', 'curvature',
                                  'aspect_ratio', 'horizontal_position', 'edge_density',
                                  'straightness', 'mean_luminance', 'num_components',
                                  'largest_component_ratio', 'solidity']}
    h, w = mask_np.shape
    ys, xs = np.where(mask_np > 0)
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    bbox_w = max(1, x2 - x1 + 1)
    bbox_h = max(1, y2 - y1 + 1)
    features['aspect_ratio'] = max(bbox_w / bbox_h, bbox_h / bbox_w)
    features['vertical_coverage'] = bbox_h / h
    center_x = (x1 + x2) / 2
    features['horizontal_position'] = center_x / w
    # correlation
    if len(ys) > 1:
        try:
            if len(np.unique(xs)) > 1 and len(np.unique(ys)) > 1:
                from scipy.stats import pearsonr
                corr, _ = pearsonr(xs, ys)
                features['correlation'] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                features['correlation'] = 0.0
        except:
            features['correlation'] = 0.0
    else:
        features['correlation'] = 0.0
    # widths
    widths = []
    for y in range(y1, y2 + 1):
        row_xs = xs[ys == y]
        if len(row_xs) > 0:
            width = row_xs.max() - row_xs.min() + 1
            widths.append(width)
    if widths:
        features['mean_width'] = float(np.mean(widths))
        features['std_width'] = float(np.std(widths))
        features['width_ratio'] = float(np.max(widths) / max(1, np.min(widths)))
    else:
        features['mean_width'] = 0.0
        features['std_width'] = 0.0
        features['width_ratio'] = 1.0
    # PCA angle
    try:
        coords = np.column_stack([xs, ys])
        coords_centered = coords - coords.mean(axis=0)
        cov = np.cov(coords_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
        angle = np.arctan2(principal_axis[1], principal_axis[0])
        features['angle'] = abs(np.degrees(angle))
    except:
        features['angle'] = 0.0
    # curvature
    try:
        skel = skeletonize(mask_np > 0)
        skel_coords = np.column_stack(np.where(skel))
        if len(skel_coords) > 2:
            angles_list = []
            for i in range(1, len(skel_coords) - 1):
                v1 = skel_coords[i] - skel_coords[i-1]
                v2 = skel_coords[i+1] - skel_coords[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle_change = np.arccos(cos_angle)
                    angles_list.append(angle_change)
            features['curvature'] = float(np.mean(angles_list)) if angles_list else 0.0
        else:
            features['curvature'] = 0.0
    except:
        features['curvature'] = 0.0
    # edge density
    try:
        edges = cv2.Canny((mask_np * 255).astype(np.uint8), 50, 150)
        edge_pixels = (edges > 0).sum()
        features['edge_density'] = float(edge_pixels / features['area']) if features['area'] > 0 else 0.0
    except:
        features['edge_density'] = 0.0
    # straightness
    try:
        skel_length = _skeleton_length(mask_np)
        features['straightness'] = float(features['area'] / skel_length) if skel_length > 0 else 0.0
    except:
        features['straightness'] = 0.0
    # mean luminance
    if image_np is not None:
        try:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            masked_pixels = gray[mask_np > 0]
            features['mean_luminance'] = float(masked_pixels.mean()) if len(masked_pixels) > 0 else 0.0
        except:
            features['mean_luminance'] = 0.0
    else:
        features['mean_luminance'] = 0.0
    # components
    try:
        bin_mask = (mask_np > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        features['num_components'] = int(num_labels - 1)
        if features['num_components'] > 1:
            component_areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
            max_component_area = max(component_areas) if component_areas else 0
            features['largest_component_ratio'] = float(max_component_area / features['area']) if features['area'] > 0 else 0.0
        else:
            features['largest_component_ratio'] = 1.0
    except:
        features['num_components'] = 0
        features['largest_component_ratio'] = 0.0
    # solidity
    try:
        bin_mask = (mask_np > 0).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            total_hull_area = 0
            for contour in contours:
                if len(contour) >= 3:
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    total_hull_area += hull_area
            features['solidity'] = float(features['area'] / total_hull_area) if total_hull_area > 0 else 0.0
        else:
            features['solidity'] = 1.0
    except:
        features['solidity'] = 0.0
    return features
# --------------------------------------------------------

def blend_color_on_mask(base, mask_idx, color_rgb, alpha=0.6):
    """Pinta color RGB sobre base (HxWx3 uint8) donde mask_idx es True."""
    if not mask_idx.any():
        return
    color_arr = np.array(color_rgb, dtype=np.float32).reshape(1,1,3)
    base[mask_idx] = (base[mask_idx].astype(np.float32)*(1-alpha) + color_arr*alpha).astype(np.uint8)


def compute_ransac_stats(mask_np, image_np=None, ransac_tol=3.0, min_samples=10):
    """
    Fit RANSAC line to mask pixels. Devuelve diccionario con:
      - area, num_points, ransac_applied (bool)
      - inliers, inline_pct, outline_pct, coef, intercept, res_std, swap
    swap=True indica que se ajustó x ~ y en vez de y ~ x (para pendientes verticales).
    """
    bin_mask = (mask_np > 0).astype(np.uint8)
    area = int(bin_mask.sum())
    ys, xs = np.where(bin_mask)
    n = len(xs)
    if area == 0 or n < min_samples:
        return {'area': area, 'num_points': n, 'ransac_applied': False}

    # Escoger orientación (fit y ~ x o x ~ y) según varianza
    swap = False
    if xs.var() >= ys.var():
        X = xs.reshape(-1, 1).astype(float)
        y = ys.astype(float)
    else:
        X = ys.reshape(-1, 1).astype(float)
        y = xs.astype(float)
        swap = True

    try:
        base = LinearRegression()
        # Compatibilidad con diferentes versiones de sklearn:
        # - versiones modernas usan "estimator="
        # - versiones antiguas usan "base_estimator="
        try:
            ransac = RANSACRegressor(estimator=base, residual_threshold=ransac_tol, min_samples=min_samples)
        except TypeError:
            ransac = RANSACRegressor(base_estimator=base, residual_threshold=ransac_tol, min_samples=min_samples)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        inliers = int(np.sum(inlier_mask))
        inline_pct = float(inliers) / area if area > 0 else 0.0
        outline_pct = 1.0 - inline_pct
        # obtener coeficientes de la regresión robusta
        est = ransac.estimator_
        coef = float(est.coef_[0]) if hasattr(est, 'coef_') else 0.0
        intercept = float(est.intercept_) if hasattr(est, 'intercept_') else 0.0
        residuals = y[inlier_mask] - ransac.predict(X[inlier_mask]) if inliers > 0 else np.array([])
        res_std = float(np.std(residuals)) if residuals.size > 0 else 0.0
        return {
            'area': area,
            'num_points': n,
            'ransac_applied': True,
            'inliers': inliers,
            'inline_pct': inline_pct,
            'outline_pct': outline_pct,
            'coef': coef,
            'intercept': intercept,
            'res_std': res_std,
            'swap': swap
        }
    except Exception as e:
        return {'area': area, 'num_points': n, 'ransac_applied': False, 'error': str(e)}


def masks_to_culane_lines(pred_mask_list, subsample_k=1, resample_n=0):
	"""
	Convierte lista de máscaras binarias (0/1) en lista de líneas CULane.
	- subsample_k: tomar 1 de cada K puntos del esqueleto (K>=1).
	- resample_n: si >0, re-muestrear cada polyline a N puntos (sobrescribe subsample_k).
	Devuelve lista por máscara: cada elemento es lista de componentes; cada componente = list[(x,y),...].
	"""
	def resample_polyline(pts, n_points):
		pts = np.array(pts, dtype=float)
		if len(pts) == 0:
			return []
		if len(pts) == 1:
			return [tuple(pts[0])]*n_points
		seg = np.sqrt(((pts[1:] - pts[:-1])**2).sum(axis=1))
		dist = np.concatenate([[0.0], np.cumsum(seg)])
		if dist[-1] == 0:
			return [tuple(pts[0])] * n_points
		target = np.linspace(0, dist[-1], n_points)
		xs = np.interp(target, dist, pts[:,0])
		ys = np.interp(target, dist, pts[:,1])
		return list(zip(xs, ys))

	lines_per_mask = []
	for m in pred_mask_list:
		skel = skeletonize(m > 0)
		if skel.sum() == 0:
			lines_per_mask.append([])
			continue
		num_labels, labels = cv2.connectedComponents(skel.astype(np.uint8), connectivity=8)
		mask_lines = []
		for lab in range(1, num_labels):
			ys, xs = np.where(labels == lab)
			if len(xs) == 0:
				continue
			order = np.lexsort((xs, ys))
			xs_o = xs[order]
			ys_o = ys[order]
			pts = list(zip(xs_o.tolist(), ys_o.tolist()))
			# simplificar: eliminar puntos repetidos consecutivos
			simple_pts = [pts[0]]
			for p in pts[1:]:
				if p != simple_pts[-1]:
					simple_pts.append(p)
			# aplicar resample o subsample
			if resample_n and resample_n > 0:
				simple_pts = resample_polyline(simple_pts, resample_n)
			elif subsample_k and subsample_k > 1:
				simple_pts = simple_pts[::subsample_k]
			mask_lines.append(simple_pts)
		lines_per_mask.append(mask_lines)
	return lines_per_mask

def process_folder(args):
	# ...existing code...

	# Helper para salvar CSVs (resumen + features)
	def save_csvs(summary_rows_local, csv_rows_local, out_dir, ts_base):
		os.makedirs(out_dir, exist_ok=True)
		summary_path = os.path.join(out_dir, f"infer_summary_{ts_base}.csv")
		pd.DataFrame(summary_rows_local).to_csv(summary_path, index=False)
		print(f"[SAVE_CSV] Resumen guardado en: {os.path.abspath(summary_path)}")
		if len(csv_rows_local) > 0:
			masks_path = os.path.join(out_dir, f"mask_features_{ts_base}.csv")
			pd.DataFrame(csv_rows_local).to_csv(masks_path, index=False)
			print(f"[SAVE_CSV] Features por máscara guardado en: {os.path.abspath(masks_path)}")
		else:
			print(f"[SAVE_CSV] No hay filas para mask_features (no se guardó).")

	# Si se pidió save_csv, precomputar timestamp base (usado para saves periódicos)
	save_ts_base = None
	if getattr(args, 'save_csv', False):
		save_ts_base = datetime.now().strftime("%Y%m%d_%H%M%S")
		print(f"[SAVE_CSV] CSV base timestamp: {save_ts_base} (se usará para los archivos resultantes)")

	# Inicializar variables del clasificador ML y cargar si se pidió --apply_ml_filter
	ml_model = None
	feature_cols = None
	optimal_threshold = getattr(args, 'optimal_threshold', 0.5)
	if getattr(args, 'apply_ml_filter', False):
		if not args.classifier_path or not os.path.exists(args.classifier_path):
			print(f"⚠️ classifier_path no encontrado: {args.classifier_path}. Continuando SIN ML.")
			ml_model = None
			feature_cols = None
		else:
			try:
				with open(args.classifier_path, 'rb') as f:
					ml_data = pickle.load(f)
				ml_model = ml_data.get('model') or ml_data.get('classifier') or ml_data.get('clf') or ml_data.get('pipeline')
				feature_cols = ml_data.get('feature_columns') or ml_data.get('feature_cols') or ml_data.get('features')
				if feature_cols is None:
					feature_cols = ['correlation','mean_width','std_width','width_ratio',
									'vertical_coverage','area','angle','curvature',
									'aspect_ratio','horizontal_position','edge_density',
									'straightness','mean_luminance','num_components',
									'largest_component_ratio','solidity','score']
				optimal_threshold = float(ml_data.get('optimal_threshold', ml_data.get('threshold', optimal_threshold)))
				print(f"✓ Clasificador ML cargado: {type(ml_model).__name__ if ml_model is not None else 'None'}, umbral_opt={optimal_threshold:.3f}")
			except Exception as e:
				print(f"⚠️ Error cargando clasificador ML: {e}. Continuando SIN ML.")
				ml_model = None
				feature_cols = None

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Device: {device}")
	# Mostrar umbral usado para considerar una máscara 'buena' frente al GT (verde vs amarillo)
	print(f"Usando iou_match_threshold = {getattr(args, 'iou_match_threshold', 0.3):.3f} (mascara considerada 'buena' si IoU >= umbral)")

	model = Sam3Model.from_pretrained(args.model_id).to(device)
	processor = Sam3Processor.from_pretrained(args.model_id)

	os.makedirs(args.out_dir, exist_ok=True)
	overlays_dir = os.path.join(args.out_dir, "overlays_infer")
	os.makedirs(overlays_dir, exist_ok=True)
	# Mostrar de forma clara dónde se guardarán los CSV si --save_csv está activado
	if getattr(args, 'save_csv', False):
		out_dir_abs = os.path.abspath(args.out_dir)
		print(f"[SAVE_CSV] Los CSV se guardarán en: {out_dir_abs}")
		print(f"[SAVE_CSV] Patrón resumen: {os.path.join(out_dir_abs, 'infer_summary_<YYYYMMDD_HHMMSS>.csv')}")
		print(f"[SAVE_CSV] Patrón features por máscara: {os.path.join(out_dir_abs, 'mask_features_<YYYYMMDD_HHMMSS>.csv')}")
	# preparar args-like para filter_instance_masks
	from types import SimpleNamespace
	f_args = SimpleNamespace(
		mask_threshold=args.mask_threshold,
		min_area=args.min_area,
		min_aspect_ratio=args.min_aspect_ratio,
		min_skel_length=args.min_skel_length,
		min_score=args.min_score
	)

	# image_paths puede venir de dos fuentes:
	# 1) --image_list: fichero con rutas (relativas o absolutas) — se resolverán con --image_root o --image_dir si son relativas.
	# 2) --image_dir (legacy): glob de la carpeta
	image_paths = []
	if getattr(args, 'image_list', None):
		list_path = args.image_list
		if not os.path.exists(list_path):
			raise FileNotFoundError(f"image_list no encontrado: {list_path}")
		with open(list_path, 'r') as f:
			lines = [l.strip() for l in f if l.strip()]
		missing = []
		for rel in lines:
			# Probar varias opciones y usar la primera que exista:
			# 1) la ruta tal cual (puede ser absoluta)
			# 2) prefijada con image_root (si se dio)
			# 3) prefijada con image_dir (si se dio)
			# 4) finalmente la ruta tal cual como fallback
			candidates = []
			candidates.append(rel)
			if getattr(args, 'image_root', None):
				candidates.append(os.path.join(args.image_root, rel.lstrip(os.sep)))
			if getattr(args, 'image_dir', None):
				candidates.append(os.path.join(args.image_dir, rel.lstrip(os.sep)))
			# normalizar y buscar el primero existente
			found = None
			for c in candidates:
				try:
					cn = os.path.normpath(c)
				except Exception:
					cn = c
				if os.path.exists(cn):
					found = cn
					break
			if found is not None:
				image_paths.append(found)
			else:
				# Si no existe ningún candidato, registrar missing.
				# Mostrar como ejemplo la ruta prefijada con image_root si está disponible
				if getattr(args, 'image_root', None):
					prefixed = os.path.join(args.image_root, rel.lstrip(os.sep))
					missing.append(prefixed)
				else:
					missing.append(rel)
		image_paths = sorted(image_paths)
		print(f"[IMAGE_LIST] Leídas {len(lines)} rutas desde {list_path}, encontradas válidas={len(image_paths)}, missing={len(missing)}")
		if missing:
			print(f"[IMAGE_LIST] Rutas no existentes (muestras): {missing[:5]}")
	else:
		if not getattr(args, 'image_dir', None):
			raise ValueError("Debe indicar --image_dir o --image_list")
		image_paths = sorted(
			glob.glob(os.path.join(args.image_dir, "*.jpg")) +
			glob.glob(os.path.join(args.image_dir, "*.png")) +
			glob.glob(os.path.join(args.image_dir, "*.jpeg"))
		)
		print(f"[IMAGE_DIR] Usando carpeta {args.image_dir}: imágenes encontradas={len(image_paths)}")

	# Preparar directorios para opciones opcionales
	if getattr(args, 'save_mask', False):
		saved_masks_dir = os.path.join(args.out_dir, "saved_masks")
		os.makedirs(saved_masks_dir, exist_ok=True)
	if getattr(args, 'create_txt_line', False):
		lines_out_dir = os.path.join(args.out_dir, "generated_lines")
		os.makedirs(lines_out_dir, exist_ok=True)
	# Aplicar start_index si se solicita
	start_idx = max(0, int(getattr(args, "start_index", 0)))
	if start_idx >= len(image_paths):
		print(f"⚠️ start_index={start_idx} >= número de imágenes ({len(image_paths)}). No hay imágenes para procesar.")
		image_paths = []
	elif start_idx > 0:
		print(f"🔁 Empezando en índice {start_idx} -> {os.path.basename(image_paths[start_idx])}")
		image_paths = image_paths[start_idx:]

	summary_rows = []
	# Acumulador para CSV de features por máscara (usado para entrenamiento)
	csv_rows = []
	processed_count = 0
	for img_path in image_paths:
		processed_count += 1
		stem = os.path.splitext(os.path.basename(img_path))[0]
		# build GT path next to the image file (so .lines.txt is detected even if images come from image_list or other folders)
		gt_path = os.path.splitext(img_path)[0] + ".lines.txt"

		image = load_image(img_path)
		image_np = np.array(image)  # RGB HxWx3
		h, w = image_np.shape[:2]

		# -----------------------------
		# NUEVO: calcular ruta relativa de la imagen para el CSV (image_rel)
		rel_path = None
		try:
			if getattr(args, 'image_root', None):
				rel_path = os.path.relpath(img_path, args.image_root)
			elif getattr(args, 'image_dir', None):
				rel_path = os.path.relpath(img_path, args.image_dir)
		except Exception:
			rel_path = None
		if rel_path is None:
			rel_path = os.path.basename(img_path)
		# -----------------------------

		# Inferencia
		inputs = processor(images=image, text=args.prompt, return_tensors="pt").to(device)
		with torch.no_grad():
			outputs = model(**inputs)
		results = processor.post_process_instance_segmentation(
			outputs,
			threshold=args.score_threshold,
			mask_threshold=args.mask_threshold,
			target_sizes=inputs.get("original_sizes").tolist()
		)[0]

		masks = results.get("masks", None)
		scores = results.get("scores", None)

		# Preparar lista original de máscaras (np.uint8 0/1)
		pred_mask_list = []
		if masks is not None and len(masks) > 0:
			for m in masks:
				pred_mask_list.append((m > args.mask_threshold).cpu().numpy().astype(np.uint8))
		else:
			pred_mask_list = []

		# -----------------------------
		# NUEVO: determinar existencia de fichero GT y si tiene líneas
		gt_file_exists = os.path.exists(gt_path)
		if gt_file_exists:
			try:
				gt_mask = culane_lines_to_mask(gt_path, w, h, thickness=args.gt_thickness)
				gt_has_lines = bool(gt_mask.max() > 0)
			except Exception:
				gt_mask = np.zeros((h, w), dtype=np.uint8)
				gt_has_lines = False
		else:
			gt_mask = np.zeros((h, w), dtype=np.uint8)
			gt_has_lines = False
		# -----------------------------

		# --- SAVE MASKS (opcional) ---
		if getattr(args, 'save_mask', False) and len(pred_mask_list) > 0:
			combined = np.zeros((h, w), dtype=np.uint8)
			for i_m, mm in enumerate(pred_mask_list):
				path_i = os.path.join(saved_masks_dir, f"{stem}_mask_{i_m:03d}.png")
				Image.fromarray((mm * 255).astype(np.uint8)).save(path_i)
				combined = np.logical_or(combined, mm).astype(np.uint8)
			combined_path = os.path.join(saved_masks_dir, f"{stem}_mask_combined.png")
			Image.fromarray((combined * 255).astype(np.uint8)).save(combined_path)
			print(f"   [save_mask] máscaras guardadas en {saved_masks_dir}")

		# --- CREATE TXT LINES (opcional) ---
		if getattr(args, 'create_txt_line', False):
			try:
				lines_per_mask = masks_to_culane_lines(pred_mask_list, subsample_k=args.line_subsample_k, resample_n=args.line_resample_n)
				out_lines = []
				for mask_lines in lines_per_mask:
					for comp in mask_lines:
						if len(comp) < 2:
							continue
						# Intentar ajustar RANSAC sobre los puntos del componente para reducir ruido
						try:
							pts = np.array(comp, dtype=float)
							xs = pts[:,0]
							ys = pts[:,1]
							# Si hay pocos puntos, escribir directamente
							if len(xs) < max(2, args.ransac_min_samples):
								raise ValueError("pocos puntos para RANSAC")
							# Elegir orientación según varianza (como en compute_ransac_stats)
							swap = False
							if xs.var() >= ys.var():
								X = xs.reshape(-1,1)
								y = ys
							else:
								X = ys.reshape(-1,1)
								y = xs
								swap = True
							# Ajuste RANSAC robusto
							try:
								ransac = RANSACRegressor(estimator=LinearRegression(), residual_threshold=args.ransac_tol, min_samples=max(2, args.ransac_min_samples))
							except TypeError:
								ransac = RANSACRegressor(base_estimator=LinearRegression(), residual_threshold=args.ransac_tol, min_samples=max(2, args.ransac_min_samples))
							ransac.fit(X, y)
							# Muestrear la recta ajustada en el rango de la componente
							minv = int(np.floor(X.min()))
							maxv = int(np.ceil(X.max()))
							if maxv - minv <= 1:
								# pocos valores, usar puntos originales predichos
								sample_coords = X.flatten()
							else:
								# tomar  max(10, n_points) muestras uniformes
								n_samp = max(10, len(X))
								sample_coords = np.linspace(minv, maxv, n_samp)
							pred = ransac.predict(sample_coords.reshape(-1,1))
							if swap:
								line_pts = [(float(pred_i), float(sc)) for sc, pred_i in zip(sample_coords, pred)]
							else:
								line_pts = [(float(sc), float(pred_i)) for sc, pred_i in zip(sample_coords, pred)]
							# Formatear como CULane line (x y x y ...)
							parts = []
							for (xv, yv) in line_pts:
								parts.append(f"{float(xv):.1f}")
								parts.append(f"{float(yv):.1f}")
							out_lines.append(" ".join(parts))
						except Exception:
							# Fallback: escribir los puntos originales del skeleton si RANSAC no funciona
							parts = []
							for (x, y) in comp:
								parts.append(f"{float(x):.1f}")
								parts.append(f"{float(y):.1f}")
							out_lines.append(" ".join(parts))
				out_txt_path = os.path.join(lines_out_dir, f"{stem}.lines.txt")
				# escribir las líneas (crea archivo aunque esté vacío para trazabilidad)
				with open(out_txt_path, "w") as f:
					for L in out_lines:
						f.write(L + "\n")
				if out_lines:
					print(f"   [create_txt_line] generado: {out_txt_path} ({len(out_lines)} lines)")
				else:
					print(f"   [create_txt_line] creado archivo vacío: {out_txt_path} (sin líneas detectadas)")
			except Exception as e:
				print(f"   ⚠️ [create_txt_line] fallo para {stem}: {e}")

		# --- RECOGER FEATURES por MÁSCARA (para CSV de entrenamiento) ---
		if getattr(args, 'save_csv', False):
			if len(pred_mask_list) > 0:
				# ...existing per-mask code (sin cambios)
				# Mantener exactamente el bloque anterior que itera por cada máscara y añade filas por máscara.
				# (Se deja aquí como "existing per-mask code" para no duplicar; en su fichero real debe quedar el bloque original.)
				for i_mask, pm in enumerate(pred_mask_list):
					# básico
					m_np = (pm > 0).astype(np.uint8)
					# score (si existe)
					try:
						score_val = float(scores[i_mask].item()) if (scores is not None and len(scores) > i_mask) else None
					except Exception:
						score_val = None
					# ML prob (si calculada)
					ml_p = ml_probs.get(i_mask, None) if 'ml_probs' in locals() else None
					# calcular features (robusto)
					try:
						feat = compute_mask_features(m_np, image_np)
					except Exception as e:
						feat = {}
					# IoU vs GT: si existe fichero .lines.txt calcular IoU (incluso si está vacío)
					if gt_file_exists:
						iou_val = compute_iou_with_dilation(m_np * 255, gt_mask, dilation_radius=args.dilation_iou)
					else:
						iou_val = None
					# labels: label30 / label50 solo si hay fichero GT; si no hay fichero GT se omiten (vacío)
					if iou_val is None:
						label30 = ""
						label50 = ""
					else:
						label30 = 1 if iou_val >= 0.3 else 0
						label50 = 1 if iou_val >= 0.5 else 0

					# RANSAC stats (solo si está activado)
					ransac_stats = None
					if getattr(args, 'debug_ransac', False):
						try:
							ransac_stats = compute_ransac_stats(m_np, image_np=image_np, ransac_tol=args.ransac_tol, min_samples=args.ransac_min_samples)
						except Exception:
							ransac_stats = None

					# Construir fila única y consistente
					row = {
						"image_rel": rel_path,
						"image": stem,
						"mask_id": i_mask,
						"gt_file_exists": int(gt_file_exists),
						"gt_has_lines": int(gt_has_lines),
						"gt_present": int(gt_has_lines),
						"iou_to_gt": (None if iou_val is None else float(iou_val)),
						"label30": label30,
						"label50": label50,
						"score": (None if score_val is None else float(score_val)),
						"ml_prob": (None if ml_p is None else float(ml_p)),
						"is_kept": int(i_mask in kept_indices) if 'kept_indices' in locals() else 0
					}

					# Adjuntar campos RANSAC si calculados
					if ransac_stats:
						row.update({
							"ransac_applied": bool(ransac_stats.get('ransac_applied', False)),
							"ransac_inliers": int(ransac_stats.get('inliers', 0)) if ransac_stats.get('ransac_applied', False) else 0,
							"ransac_inline_pct": float(ransac_stats.get('inline_pct', 0.0)) if ransac_stats.get('ransac_applied', False) else None,
							"ransac_outline_pct": float(ransac_stats.get('outline_pct', 0.0)) if ransac_stats.get('ransac_applied', False) else None,
							"ransac_coef": float(ransac_stats.get('coef', 0.0)) if ransac_stats.get('ransac_applied', False) else None,
							"ransac_intercept": float(ransac_stats.get('intercept', 0.0)) if ransac_stats.get('ransac_applied', False) else None,
							"ransac_res_std": float(ransac_stats.get('res_std', 0.0)) if ransac_stats.get('ransac_applied', False) else None,
						})
					else:
						row.update({
							"ransac_applied": False,
							"ransac_inliers": 0,
							"ransac_inline_pct": None,
							"ransac_outline_pct": None,
							"ransac_coef": None,
							"ransac_intercept": None,
							"ransac_res_std": None,
						})

					# añadir features al row (evitar claves conflictivas)
					for k, v in feat.items():
						row[f"feat_{k}"] = v
					csv_rows.append(row)
			else:
				# No se detectaron máscaras: guardar fila "vacía" para que todas las imágenes aparezcan en el CSV
				empty_row = {
					"image_rel": rel_path,
					"image": stem,
					"mask_id": "",
					"num_masks": 0,
					"gt_file_exists": int(gt_file_exists),
					"gt_has_lines": int(gt_has_lines),
					"gt_present": int(gt_has_lines),
					"iou_to_gt": "",
					"label30": "",
					"label50": "",
					"score": None,
					"ml_prob": None,
					"is_kept": 0,
					"ransac_applied": False,
					"ransac_inliers": 0,
					"ransac_inline_pct": None,
					"ransac_outline_pct": None,
					"ransac_coef": None,
					"ransac_intercept": None,
					"ransac_res_std": None,
				}
				csv_rows.append(empty_row)

		# Aplicar filtrado por instancia (reglas) para obtener candidatos
		candidate_indices = set(range(len(pred_mask_list))) if masks is not None else set()
		if args.filter_instances and masks is not None and len(masks) > 0:
			kept_masks = filter_instance_masks(masks, scores, f_args)
			candidate_indices = set()
			for i, m in enumerate(pred_mask_list):
				for km in kept_masks:
					if np.logical_and(m, km).sum() > 0:
						candidate_indices.add(i)
						break

		# Aplicar filtro ML (opcional) sobre candidatos
		final_kept_indices = set()
		ml_probs = {}  # guardar probabilidades para CSV/traza
		if ml_model is not None and feature_cols is not None:
			for i in range(len(pred_mask_list)):
				# solo evaluar ML sobre candidatos; si no queremos filtrar por reglas, candidate_indices contiene todos
				if i not in candidate_indices:
					continue
				m_np = (pred_mask_list[i] > 0).astype(np.uint8)
				try:
					feat = compute_mask_features(m_np, image_np)
					# asegurar keys en orden
					X_mask = np.array([feat.get(c, 0.0) for c in feature_cols], dtype=float).reshape(1, -1)
					if hasattr(ml_model, 'predict_proba'):
						y_proba = float(ml_model.predict_proba(X_mask)[0, 1])
					else:
						y_pred = ml_model.predict(X_mask)[0]
						y_proba = 1.0 if int(y_pred) == 1 else 0.0
				except Exception as e:
					# si falla la extracción, excluir
					print(f"      ⚠️ ML feature/predict fallo para máscara {i}: {e}")
					y_proba = 0.0
				ml_probs[i] = y_proba
				if y_proba >= optimal_threshold:
					final_kept_indices.add(i)
			# DEBUG: resumen de ML por imagen
			try:
				print(f"   [ML FILTER] candidatos evaluados={len(candidate_indices)}, retenidas_por_ML={len(final_kept_indices)}, umbral_opt={optimal_threshold:.3f}")
				if ml_probs:
					top = sorted(ml_probs.items(), key=lambda x: x[1], reverse=True)[:3]
					top_str = ", ".join([f"mask_{k}={v:.3f}" for k, v in top])
					print(f"   [ML FILTER] top_probs: {top_str}")
			except Exception:
				pass
		else:
			# Si no hay ML, mantener los candidatos tal cual
			final_kept_indices = set(candidate_indices)
			if args.apply_ml_filter:
				# se pidió ML pero no se cargó correctamente: aviso por imagen
				print("   [ML FILTER] --apply_ml_filter activado pero no se cargó el clasificador; usando solo filtrado por reglas (si aplica).")

		# Usar final_kept_indices como kept para coloreado y outputs
		kept_indices = final_kept_indices
		# Inicializar estructura de resultados por imagen (evita NameError al añadir ml_prob)
		row_result = {}

		# REUSAR gt_file_exists / gt_has_lines / gt_mask ya calculados más arriba
		gt_present = gt_has_lines  # para compatibilidad con el resto del código

		overlay = image_np.copy()
		counts = {"green":0, "yellow":0, "red":0, "no_pred":0}

		if len(pred_mask_list) == 0:
			counts["no_pred"] = 1
		else:
			for i, pm in enumerate(pred_mask_list):
				mask_bool = pm > 0
				iou = 0.0
				if gt_present:
					iou = compute_iou_with_dilation(pm*255, gt_mask, dilation_radius=args.dilation_iou)
				is_kept = (i in kept_indices)
				# --- RANSAC DEBUG opcional por máscara ---
				if getattr(args, 'debug_ransac', False):
					try:
						ransac_stats = compute_ransac_stats(pm, image_np=image_np, ransac_tol=args.ransac_tol, min_samples=args.ransac_min_samples)
						if ransac_stats.get('ransac_applied'):
							print(f"      [RANSAC] mask_{i}: area={ransac_stats['area']} pts={ransac_stats['num_points']} inliers={ransac_stats['inliers']} inline%={ransac_stats['inline_pct']*100:.2f} outline%={ransac_stats['outline_pct']*100:.2f} coef={ransac_stats['coef']:.3f} int={ransac_stats['intercept']:.2f} res_std={ransac_stats['res_std']:.3f} swap={ransac_stats['swap']}")
							# Dibujar inline% directamente en la overlay (solo en modo debug_ransac)
							'''try:
								inline_pct = ransac_stats.get('inline_pct', 0.0)
								ys_i, xs_i = np.where(pm > 0)
								if len(xs_i) > 0:
									cx = int(xs_i.mean())
									cy = int(ys_i.mean())
									text = f"{inline_pct*100:.1f}%"
									# Outline negro para legibilidad
									cv2.putText(overlay, text, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), thickness=3, lineType=cv2.LINE_AA)
									cv2.putText(overlay, text, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), thickness=1, lineType=cv2.LINE_AA)
							except Exception:
								# no bloquear el bucle por un fallo de dibujo
								pass
							'''
						else:
							if 'error' in ransac_stats:
								print(f"      [RANSAC] mask_{i}: no aplicado (error={ransac_stats['error']})")
							else:
								print(f"      [RANSAC] mask_{i}: no aplicado (pocos puntos o área=0) pts={ransac_stats.get('num_points',0)} area={ransac_stats.get('area',0)}")
					except Exception as e:
						print(f"      ⚠️ [RANSAC] fallo procesamiento mask_{i}: {e}")
				if is_kept:
					if gt_present and iou >= args.iou_match_threshold:
						blend_color_on_mask(overlay, mask_bool, color_rgb=(34,139,34), alpha=args.alpha_good)  # green
						counts["green"] += 1
					else:
						blend_color_on_mask(overlay, mask_bool, color_rgb=(255,200,0), alpha=args.alpha_premium)  # yellow
						counts["yellow"] += 1
				else:
					blend_color_on_mask(overlay, mask_bool, color_rgb=(220,20,60), alpha=args.alpha_bad)  # red
					counts["red"] += 1

				# opcional: añadir probabilidad ML a summary (si existe)
				if 'ml_prob' not in row_result:
					row_result['ml_prob'] = {}
				if i in ml_probs:
					row_result['ml_prob'][f"mask_{i}"] = ml_probs[i]

		if gt_present:
			gt_idx = gt_mask > 0
			blend_color_on_mask(overlay, gt_idx, color_rgb=(0,128,255), alpha=args.alpha_gt)  # blue-ish

		# Dibujar en overlay las líneas ya guardadas en generated_lines (si existe .lines.txt para esta imagen)
		# Color lila (~RGB (147,112,219)) para distinguirlas
		if getattr(args, 'create_txt_line', False):
			try:
				saved_lines_path = os.path.join(lines_out_dir, f"{stem}.lines.txt")
				if os.path.exists(saved_lines_path):
					with open(saved_lines_path, "r") as lf:
						for ln in lf:
							parts = ln.strip().split()
							if len(parts) < 4:
								continue
							try:
								coords = list(map(float, parts))
							except Exception:
								continue
							pts = []
							for ii in range(0, len(coords), 2):
								x = int(round(coords[ii])); y = int(round(coords[ii+1]))
								if x < 0 or y < 0:
									continue
								pts.append((x, y))
							if len(pts) < 2:
								continue
							# Dibujar segmentos de la polyline en lila (antialias)
							for p1, p2 in zip(pts[:-1], pts[1:]):
								cv2.line(overlay, p1, p2, color=(147,112,219), thickness=getattr(args, 'line_draw_thickness', 2), lineType=cv2.LINE_AA)
			except Exception as e:
				print(f"   ⚠️ [create_txt_line] fallo dibujando {stem}.lines.txt en overlay: {e}")

		out_path = os.path.join(overlays_dir, f"{stem}_infer_overlay.png")
		Image.fromarray(overlay).save(out_path)

		# Serializar ml_prob dict para CSV (si existe)
		ml_prob_flat = ""
		if 'ml_prob' in row_result:
			ml_prob_flat = ";".join([f"{k}={v:.3f}" for k,v in row_result['ml_prob'].items()])
		summary_rows.append({
			"image": stem,
			"image_rel": rel_path,
			"num_preds": len(pred_mask_list),
			"num_green": counts["green"],
			"num_yellow": counts["yellow"],
			"num_red": counts["red"],
			"gt_present": int(gt_present),
			"gt_file_exists": int(gt_file_exists),
			"gt_has_lines": int(gt_has_lines),
			"saved_overlay": out_path,
			"ml_probs": ml_prob_flat
		})

		print(f"{stem}: preds={len(pred_mask_list)}, green={counts['green']}, yellow={counts['yellow']}, red={counts['red']}, gt_file_exists={gt_file_exists}, gt_has_lines={gt_has_lines}")

		# Guardado periódico si se solicitó (--save_csv_interval > 0)
		if getattr(args, 'save_csv', False) and getattr(args, 'save_csv_interval', 0) > 0 and save_ts_base is not None:
			interval = int(getattr(args, 'save_csv_interval', 0))
			if interval > 0 and (processed_count % interval == 0):
				print(f"[SAVE_CSV] Guardando CSVs parciales tras procesar {processed_count} imágenes...")
				save_csvs(summary_rows, csv_rows, args.out_dir, save_ts_base)

	# Guardar CSV resumen final (si se pidió)
	if getattr(args, 'save_csv', False):
		# si no hubo timestamp base (p. ej. --save_csv pero interval=0), generar uno ahora
		if save_ts_base is None:
			save_ts_base = datetime.now().strftime("%Y%m%d_%H%M%S")
		print(f"[SAVE_CSV] Guardando CSVs finales...")
		save_csvs(summary_rows, csv_rows, args.out_dir, save_ts_base)

def parse_args():
     parser = argparse.ArgumentParser(description="Infer + visualize SAM3 masks vs GT (autónomo)")
     # image_dir puede usarse si no se pasa --image_list; por compatibilidad lo dejamos opcional
     parser.add_argument("--image_dir", required=False, default=None, help="Carpeta con imágenes y opcionalmente .lines.txt (si no se usa --image_list)")
     parser.add_argument("--image_list", type=str, default=None, help="Fichero .txt con rutas por línea (relativas o absolutas). Si son relativas se resolverán con --image_root o --image_dir")
     parser.add_argument("--image_root", type=str, default=None, help="Directorio root para resolver rutas relativas del --image_list")
     parser.add_argument("--start_index", type=int, default=0, help="Índice de imagen inicial (ej: 1000 para empezar desde la imagen 1000 en la lista)")
     parser.add_argument("--out_dir", required=True, help="Carpeta salida")
     parser.add_argument("--model_id", default="facebook/sam3")
     parser.add_argument("--prompt", default="marker defining driving lane limits")
     parser.add_argument("--score_threshold", type=float, default=0.5)
     parser.add_argument("--mask_threshold", type=float, default=0.5)
     parser.add_argument("--gt_thickness", type=int, default=2)
     parser.add_argument("--min_area", type=int, default=1000)
     parser.add_argument("--min_aspect_ratio", type=float, default=4.0)
     parser.add_argument("--min_skel_length", type=int, default=50)
     parser.add_argument("--min_score", type=float, default=0.3)
     parser.add_argument("--iou_match_threshold", type=float, default=0.3, help="IoU threshold to mark a kept mask as GOOD (green). Default=0.3 (IoU30)")
     parser.add_argument("--dilation_iou", type=int, default=15)
     parser.add_argument("--alpha_good", type=float, default=0.7)
     parser.add_argument("--alpha_premium", type=float, default=0.6)
     parser.add_argument("--alpha_bad", type=float, default=0.6)
     parser.add_argument("--alpha_gt", type=float, default=0.7)
     parser.add_argument("--save_csv", action="store_true")
     parser.add_argument("--filter_instances", action="store_true", help="Aplicar filtrado de instancias antes de marcar kept")
     parser.add_argument("--apply_ml_filter", action="store_true", help="Aplicar filtro ML adicional sobre máscaras")
     parser.add_argument("--classifier_path", default= "/home/its/Hector/SAM3/output_sam3_v2/best_classifier_label30.pkl", help="Ruta a clasificador ML (pickle)")
     parser.add_argument("--optimal_threshold", type=float, help="Umbral óptimo para clasificador ML")
     # RANSAC debug
     parser.add_argument("--debug_ransac", action="store_true", help="Activar debug RANSAC por máscara (imprime stats por máscara)")
     parser.add_argument("--ransac_tol", type=float, default=3.0, help="Residual threshold (px) para RANSAC")
     parser.add_argument("--ransac_min_samples", type=int, default=10, help="Min samples para RANSAC")
     parser.add_argument("--save_mask", action="store_true", help="Guardar máscaras binarias individuales y combinada en out_dir/saved_masks")
     # aliases aceptados por conveniencia (soportar variantes usadas en CLI)
     parser.add_argument("--save_masks", action="store_true", dest="save_mask", help=argparse.SUPPRESS)
     parser.add_argument("--create_txt_line", action="store_true", help="Generar fichero CULane .lines.txt (nombre_imagen.lines.txt) en out_dir/generated_lines a partir de las máscaras")
     parser.add_argument("--line_subsample_k", type=int, default=10, help="Submuestrear skeleton points: tomar 1 cada K puntos (K>=1). Default=1 (mantener todos).")
     parser.add_argument("--line_resample_n", type=int, default=0, help="Si >0, re-muestrear cada línea a N puntos uniformes (sobrescribe subsample_k). Default=0 (desactivado).")
     parser.add_argument("--line_draw_thickness", type=int, default=2, help="Grosor (px) para dibujar la línea guardada (.lines.txt) en el overlay (color lila).")
     parser.add_argument("--save_csv_interval", type=int, default=20, help="Si >0, guarda los CSV cada N imágenes procesadas (0 = solo al final). --save_csv debe activarse.")
     return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_folder(args)