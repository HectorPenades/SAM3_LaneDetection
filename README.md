# SAM3 Inference & CULane Utilities

A tool to run SAM3 instance segmentation on images, optionally filter masks with a ML classifier, convert masks into CULane `.lines.txt`, save mask images and overlays, and export per-mask features to CSV for ML training.

Usage examples (dataset listed with partial paths resolved by --image_root):
- With image list (partial paths in the .txt) and ML filter:
  its@monolith:/...$ python infer_visualize.py --image_root /media/its/Stephany_4T/Hector_data/Datasets/CuLane --out_dir /media/its/Stephany_4T/Hector_data/T_Culane --start_index 0 --apply_ml_filter --classifier_path best_classifier_label30.pkl --debug_ransac --save_masks --create_txt_line --save_csv --image_list /media/its/Stephany_4T/Hector_data/Datasets/CuLane/list/train.txt --image_dir /media/its/Stephany_4T/Hector_data/Datasets/CuLane

  - If the `.txt` contains partial paths (like `/driver_.../00000.jpg`), pass `--image_root` so the script prefixes and resolves them.
  - To disable ML filtering, remove `--apply_ml_filter` and omit `--classifier_path`.

Usage example (all images in a single folder):
- If images are in one folder (no list):
  (lanes_sam3) its@monolith:/...$ python infer_visualize.py --image_dir /media/its/data/dinh/CULane_cropped_left/driver_00_30frame --out_dir /media/its/Stephany_4T/Hector_data/Folder_1 --start_index 3250 --apply_ml_filter --classifier_path best_classifier_label30.pkl --debug_ransac --save_masks --create_txt_line --save_csv

Key flags (short):
- --image_list FILE : text file with one image path per line (relative or absolute). If relative, use --image_root or --image_dir to resolve.
- --image_root DIR  : root directory to prefix relative paths from --image_list.
- --image_dir DIR   : process all images in a folder (legacy / alternative to --image_list).
- --apply_ml_filter : enable ML classifier to filter predicted masks (requires --classifier_path).
- --classifier_path FILE : pickle with trained classifier and feature columns.
- --create_txt_line  : generate CULane `.lines.txt` files from predicted masks.
- --save_mask        : save individual and combined predicted masks to out_dir/saved_masks.
- --save_csv         : save per-mask features and labels to CSV (useful to train ML models).
- --start_index N    : skip images before index N in resolved list.

Notes:
- The generated CSV contains per-mask features and IoU-based labels (e.g. label30/label50) and can be used to train or evaluate an ML filter.
- Keep `--image_root` when your list contains dataset-relative paths; otherwise the script tries the path as-is and then falls back to `--image_dir`.
- README kept minimal and focused — check CLI help (`python infer_visualize.py -h`) for all options.
