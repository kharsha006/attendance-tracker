# =============================================================
#  enroll_employees.py  —  Build employee embedding database
#
#  Source:  employee gallery/  (pre-extracted face crops)
#  Output:  data/employees.db  (SQLite — primary store)
#           enrollment/embeddings.pkl  (pickle — backwards compat)
#
#  Uses InsightFace buffalo_l (ArcFace) — no model training.
#
#  CLI USAGE
#  ─────────────────────────────────────────────────────────────
#  Enroll all employees from gallery (skip already enrolled):
#      python enroll_employees.py
#
#  Force re-enroll everyone (overwrites existing DB records):
#      python enroll_employees.py --force
#
#  Incremental: add / re-enroll a single employee by folder name:
#      python enroll_employees.py --add "kalyan sai"
#
#  List currently enrolled employees:
#      python enroll_employees.py --list
#
#  Remove an employee from the database:
#      python enroll_employees.py --remove kalyan_sai
# =============================================================

import os
import sys
import argparse
import pickle
import logging
import numpy as np
import cv2

from config import GALLERY_DIR, EMPLOYEE_DB_PATH, EMPLOYEE_DB_URL, EMBEDDINGS_FILE
from employee_db import EmployeeDB
from face_engine import FaceEngine

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────
SUPPORTED_EXTS   = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MIN_DET_SCORE    = 0.30   # lowered to allow all enrolled faces to be detected
BANNER           = "=" * 60


# ═══════════════════════════════════════════════════════════════
#  Core enrollment logic
# ═══════════════════════════════════════════════════════════════

def _folder_name_to_id(folder_name: str) -> str:
    """Normalize a folder name to a safe employee_id."""
    return folder_name.strip().lower().replace(" ", "_")


def _embed_folder(engine: FaceEngine, folder_path: str, emp_name: str) -> "tuple[np.ndarray | None, int, int]":
    """
    Process all images in a folder and return the averaged embedding.

    Dual embedding strategy (handles both full photos AND pre-cropped faces):
      1. detect_and_embed()  — runs RetinaFace detection, works on full photos.
      2. embed_face_crop()   — feeds image directly to ArcFace recognition model,
                               bypasses detection.  Used when the image IS already
                               a face crop (employee gallery use case).

    Returns:
        (mean_embedding, used_count, total_count)
        mean_embedding is None if no usable embeddings were found.
    """
    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTS)
    ])
    
    # We process all images in the folder.
    # The augmentations below will massively expand this list.

    total_count = len(image_files)
    if total_count == 0:
        print(f"   [!] No images found in {folder_path}")
        return None, 0, 0

    embeddings  = []
    via_detect  = 0   # images embedded via detection
    via_direct  = 0   # images embedded via direct crop (no detection)
    skipped     = 0   # unreadable / truly empty

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            log.debug("Cannot read: %s", img_file)
            skipped += 1
            continue

        # Resize huge images to prevent CPU hangs
        h, w = frame.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        # Create augmented versions to improve robust detection in different conditions
        augmentations = [frame]
        # 1. Horizontal Flip
        augmentations.append(cv2.flip(frame, 1))
        # 2. Darker (simulates bad lighting)
        augmentations.append(cv2.convertScaleAbs(frame, alpha=0.6, beta=0))
        # 3. Brighter (simulates bright lighting)
        augmentations.append(cv2.convertScaleAbs(frame, alpha=1.2, beta=30))
        # 4. Blurry (simulates being far away from camera)
        augmentations.append(cv2.GaussianBlur(frame, (7, 7), 0))

        for aug_frame in augmentations:
            # Pad the image because it's a tight crop, and the detector needs context
            h, w = aug_frame.shape[:2]
            pad_h, pad_w = h // 2, w // 2
            padded_frame = cv2.copyMakeBorder(aug_frame, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            # --- Strategy 1: full detection pipeline (NOW ON PADDED FRAME) ---
            faces = engine.detect_and_embed(padded_frame)
            if faces:
                best = max(faces, key=lambda f: f["det_score"])
                # Lower threshold for gallery crops (confirmed faces)
                if best["det_score"] >= MIN_DET_SCORE:
                    embeddings.append(best["embedding"])
                    via_detect += 1
            else:
                skipped += 1

    if not embeddings:
        return None, 0, total_count

    # Stack all embeddings to keep them separate (no averaging!)
    # This prevents degrading the vectors and allows robust multi-condition matching
    stacked_emb = np.stack(embeddings, axis=0).astype(np.float32)
    # L2 normalize each embedding independently
    norms = np.linalg.norm(stacked_emb, axis=1, keepdims=True)
    stacked_emb = np.divide(stacked_emb, norms, out=np.zeros_like(stacked_emb), where=norms!=0)

    return stacked_emb, len(embeddings), total_count


def _enroll_one(
    engine: FaceEngine,
    db: EmployeeDB,
    folder_name: str,
    force: bool = False,
) -> bool:
    """
    Enroll a single employee from their gallery sub-folder.

    Returns True on success, False if skipped / failed.
    """
    emp_name = folder_name.strip()
    emp_id   = _folder_name_to_id(emp_name)
    folder_path = os.path.join(GALLERY_DIR, folder_name)

    if not os.path.isdir(folder_path):
        print(f"   [!] Folder not found: {folder_path}")
        return False

    # Skip if already enrolled and not forcing
    if not force and db.employee_exists(emp_id):
        enrolled = db.get_one(emp_id)
        print(
            f"   [SKIP] {emp_name:<20}  already enrolled "
            f"({enrolled['image_count']} imgs)  -- use --force to re-enroll"
        )
        return True

    print(f"   Processing: {emp_name}")

    mean_emb, used, total = _embed_folder(engine, folder_path, emp_name)

    if mean_emb is None:
        print(f"   [!] {emp_name} -- no usable faces found ({total} images scanned). Skipping.")
        return False

    db.upsert(emp_id, emp_name, mean_emb, used)
    status = "Re-enrolled" if db.employee_exists(emp_id) else "Enrolled"
    print(f"   [OK] {emp_name:<20}  {used}/{total} images  ->  stored in DB")
    return True


# --------------------------------------------------------------
#  Pickle export (backwards compatibility)
# --------------------------------------------------------------

def _export_pkl(db: EmployeeDB):
    """
    Export the full employee DB to embeddings.pkl for backward
    compatibility with any code that still calls load_employee_database().
    """
    employee_data = db.get_all()
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(employee_data, f)
    print(f"\n   [PKL] Pickle exported -> {EMBEDDINGS_FILE}  ({len(employee_data)} employee(s))")


# --------------------------------------------------------------
#  CLI commands
# --------------------------------------------------------------

def cmd_list(db: EmployeeDB):
    """Print all enrolled employees."""
    employees = db.list_employees()
    print(BANNER)
    print(f"  ENROLLED EMPLOYEES  ({len(employees)} total)")
    print(BANNER)
    if not employees:
        print("  (none)")
    else:
        print(f"  {'ID':<22} {'Name':<22} {'Images':>6}  Enrolled At")
        print("  " + "-" * 56)
        for e in employees:
            ts = e["enrollment_timestamp"][:19].replace("T", " ")
            print(
                f"  {e['employee_id']:<22} {e['employee_name']:<22} "
                f"{e['image_count']:>6}  {ts}"
            )
    print(BANNER)


def cmd_remove(db: EmployeeDB, employee_id: str):
    """Remove a single employee from the database."""
    if db.delete(employee_id):
        print(f"  [OK] Removed: {employee_id}")
        _export_pkl(db)
    else:
        print(f"  [!] Employee not found: {employee_id}")


def cmd_add(engine: FaceEngine, db: EmployeeDB, folder_name: str):
    """Incrementally enroll a single employee."""
    print(BANNER)
    print(f"  INCREMENTAL ENROLLMENT: {folder_name}")
    print(BANNER)
    ok = _enroll_one(engine, db, folder_name, force=True)
    if ok:
        _export_pkl(db)
        print(f"\n  [OK] Incremental enrollment complete.")
    else:
        print(f"\n  [FAIL] Enrollment failed.")


def cmd_enroll_all(engine: FaceEngine, db: EmployeeDB, force: bool = False):
    """Enroll all employees found in GALLERY_DIR."""
    print(BANNER)
    print("  EMPLOYEE GALLERY ENROLLMENT  (InsightFace ArcFace)")
    print(BANNER)

    if not os.path.isdir(GALLERY_DIR):
        print(f"\n[ERROR] Gallery folder not found: {GALLERY_DIR}")
        print("Expected structure:")
        print("  employee gallery/")
        print("    kalyan sai/    <- sub-folders named after employees")
        print("    krishna/")
        print("    ...")
        sys.exit(1)

    folders = sorted([
        f for f in os.listdir(GALLERY_DIR)
        if os.path.isdir(os.path.join(GALLERY_DIR, f))
    ])

    if not folders:
        print(f"\n[ERROR] No sub-folders found in: {GALLERY_DIR}")
        sys.exit(1)

    print(f"\n  Gallery : {GALLERY_DIR}")
    print(f"  Database: {EMPLOYEE_DB_PATH}")
    print(f"  Found   : {len(folders)} employee folder(s)")
    print(f"  Mode    : {'FORCE re-enroll all' if force else 'Skip already enrolled'}")
    print()

    success, skipped, failed = 0, 0, 0

    for folder_name in folders:
        emp_id = _folder_name_to_id(folder_name)

        if not force and db.employee_exists(emp_id):
            enrolled = db.get_one(emp_id)
            print(
                f"   [SKIP] {folder_name:<20}  already enrolled "
                f"({enrolled['image_count']} imgs)"
            )
            skipped += 1
            continue

        print(f"\n   >> {folder_name}")
        folder_path = os.path.join(GALLERY_DIR, folder_name)
        mean_emb, used, total = _embed_folder(engine, folder_path, folder_name)

        if mean_emb is None:
            print(f"      [!] No usable faces ({total} images scanned). Skipping.")
            failed += 1
            continue

        emp_name = folder_name.strip()
        db.upsert(emp_id, emp_name, mean_emb, used)
        print(f"      [OK] {used}/{total} images embedded & averaged -> stored")
        success += 1

    print()
    print(BANNER)
    print(f"  ENROLLMENT SUMMARY")
    print(BANNER)
    print(f"  Enrolled : {success}")
    print(f"  Skipped  : {skipped}  (already in DB)")
    print(f"  Failed   : {failed}")
    print(f"  Total DB : {db.count()}")
    print(BANNER)

    if success > 0 or (skipped == 0 and failed == 0):
        _export_pkl(db)
    elif skipped > 0 and success == 0:
        # Nothing new enrolled; still export pkl in case it was missing
        _export_pkl(db)


# --------------------------------------------------------------
#  Entry point
# --------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enroll_employees",
        description="Enroll employees from the face gallery into the SQLite embedding database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enroll_employees.py                    # enroll all (skip existing)
  python enroll_employees.py --force            # re-enroll all employees
  python enroll_employees.py --add "kalyan sai" # add / re-enroll one employee
  python enroll_employees.py --list             # show enrolled employees
  python enroll_employees.py --remove kalyan_sai
        """,
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-enroll all employees even if already in the database.",
    )
    parser.add_argument(
        "--add",
        metavar="FOLDER_NAME",
        help="Incrementally enroll a single employee by gallery folder name.",
    )
    parser.add_argument(
        "--remove",
        metavar="EMPLOYEE_ID",
        help="Remove an employee from the database by their ID (e.g. kalyan_sai).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all enrolled employees and exit.",
    )
    return parser


def main():
    parser = build_arg_parser()
    args   = parser.parse_args()

    # -- Initialise DB ----------------------------------------
    db = EmployeeDB(EMPLOYEE_DB_URL)
    db.initialize()

    # -- List-only mode (no engine needed) --------------------
    if args.list:
        cmd_list(db)
        return

    # -- Remove mode (no engine needed) -----------------------
    if args.remove:
        cmd_remove(db, args.remove)
        return

    # -- Load InsightFace -------------------------------------
    print(BANNER)
    print("  Loading InsightFace buffalo_l ...")
    print(BANNER)

    engine = FaceEngine()
    if not engine.available():
        print(" FAILED")
        print("\n[ERROR] InsightFace could not be loaded.")
        print("Install it:  pip install insightface onnxruntime")
        sys.exit(1)

    print("  [OK] InsightFace ready\n")

    # -- Dispatch ---------------------------------------------
    if args.add:
        cmd_add(engine, db, args.add)
    else:
        cmd_enroll_all(engine, db, force=args.force)


if __name__ == "__main__":
    main()
