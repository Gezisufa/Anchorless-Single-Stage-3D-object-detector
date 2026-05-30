# sw_preprocess.py
# SolidWorks assembly preprocessing pipeline via COM (pywin32).
# Opens .sldasm, applies Display State, suppresses components,
# exports to .sldprt and .stl, optionally runs inference.
#
# Requirements:
#   - Windows with SolidWorks installed
#   - pip install pywin32 pandas openpyxl
#
# Usage:
#   python sw_preprocess.py

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

# --- Input / Output paths ---
SLDASM_PATH    = Path(r"C:\Users\andriievskyi\git-repo\voxel-nn\single-stage\in\sample\SG868975R_copy.SLDASM")
WORK_DIR       = Path(r"C:\Users\andriievskyi\git-repo\voxel-nn\single-stage\out\sample")          # copy + exports go here
EXCEL_LOG_PATH = Path(r"C:\Users\andriievskyi\git-repo\voxel-nn\single-stage\out\sample")

# --- Feature flags (stub pattern — flip to enable) ---
MAKE_COPY          = False    # True = save a copy first, work on copy. False = work on original
SET_DISPLAY_STATE  = True     # True = switch to DISPLAY_STATE_NAME before processing
SUPPRESS_HIDDEN    = False     # True = suppress components hidden in Display State
SUPPRESS_BY_LIST   = True    # True = suppress only components in SUPPRESS_LIST
AUTO_RUN_INFER     = False    # True = automatically call infer.py after STL export

# --- Display State ---
DISPLAY_STATE_NAME = "HV only"

# --- Suppress library (used when SUPPRESS_BY_LIST=True) ---
# Components here are KNOWN — they won't appear in STL for inference.
# Their properties go straight into the final Excel table (without Confidence).
#
# Keys = component base names (matches all instances: -1, -2, -3, ...)
# Use full instance name (e.g., "HEX-BOLT_M12x40-1") to match only that one.
#
# Example assembly tree:
#   MyAssembly.sldasm
#   ├── Frame-1                    <- keep (structural)
#   ├── HEX-BOLT_M12x40-1         <- suppress (known fastener)
#   ├── HEX-BOLT_M12x40-2         <- suppress (known fastener)
#   ├── HEX-NUT_M12-1             <- suppress (known fastener)
#   └── Cable_Harness-1            <- suppress (non-structural, no bolts)

SUPPRESS_LIBRARY = {
     "ComponentBaseName": {
         "Quick Descript": "...",
         "bolts": [
             {"Size of bolt": "Mxx", "Torque [N/m]": xx, "Numbers of tightenings": xx},
         ],
     },
}

# Flat list for backward compat (auto-generated from library keys)
SUPPRESS_LIST = list(SUPPRESS_LIBRARY.keys())

# --- Naming ---
COPY_SUFFIX = "_voxel-nn_processing"

# --- Infer config (used when AUTO_RUN_INFER=True) ---
INFER_SCRIPT = Path("infer.py")


# ============================================================
# SolidWorks COM constants
# ============================================================
# Document types
swDocASSEMBLY  = 2
swDocPART      = 1

# Suppress states
swComponentSuppressed      = 0
swComponentResolved         = 2
swComponentFullyResolved    = 3

# Save As types
swSaveAsCurrentVersion = 0

# File types for SaveAs
swFileTypeSLDASM = "SLDASM"
swFileTypeSLDPRT = "SLDPRT"
swFileTypeSTL    = "STL"

# Close options
swDontSaveChanges = 2

# Component visibility
swComponentVisible = 1
swComponentHidden  = 0


# ============================================================
# SolidWorks connection
# ============================================================
def connect_solidworks():
    """
    Connects to running SolidWorks instance via COM.
    SolidWorks must be already open.
    """
    try:
        import win32com.client
        import pythoncom

        sw = win32com.client.Dispatch("SldWorks.Application")
        sw.Visible = True
        print(f"[SW] Connected to SolidWorks {sw.RevisionNumber}")
        return sw
    except ImportError:
        print("[ERROR] pywin32 not installed. Run: pip install pywin32")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Cannot connect to SolidWorks: {e}")
        print("        Make sure SolidWorks is running.")
        sys.exit(1)


# ============================================================
# STAGE 1: Open assembly
# ============================================================
def open_assembly(sw, asm_path: Path):
    """
    Opens .sldasm file in SolidWorks.
    """
    import win32com.client
    import pythoncom

    print(f"\n[STAGE 1] Opening assembly: {asm_path}")

    if not asm_path.exists():
        raise FileNotFoundError(f"Assembly not found: {asm_path}")

    errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

    doc = sw.OpenDoc6(
        str(asm_path),
        swDocASSEMBLY,
        1,      # swOpenDocOptions_Silent
        "",
        errors,
        warnings,
    )

    if doc is None:
        raise RuntimeError(f"Failed to open assembly. Errors={errors}, Warnings={warnings}")

    print(f"  Opened: {doc.GetTitle}")
    return doc


# ============================================================
# STAGE 2: Set Display State
# ============================================================
def set_display_state(doc, state_name: str):
    """
    Activates a specific Display State in the assembly.
    Uses ConfigurationManager path which works with COM late binding.
    """
    print(f"\n[STAGE 2] Setting Display State: '{state_name}'")

    try:
        config_mgr = doc.ConfigurationManager
        config = config_mgr.ActiveConfiguration

        if config is None:
            print("  [WARN] No active configuration found")
            return False

        display_states = config.GetDisplayStates
        if display_states is None:
            print("  [WARN] No display states available")
            return False

        # GetDisplayStates may return a tuple or list
        ds_list = list(display_states) if display_states else []

        if state_name not in ds_list:
            print(f"  [WARN] Display State '{state_name}' not found.")
            print(f"         Available: {ds_list}")
            return False

        config.ApplyDisplayState(state_name)
        print(f"  Display State set to: '{state_name}'")
        return True

    except Exception as e:
        print(f"  [WARN] Could not set Display State: {e}")
        print("         Continuing without Display State change.")
        return False


# ============================================================
# STAGE 3: Save copy of assembly
# ============================================================
def save_assembly_copy(sw, doc, work_dir: Path, suffix: str) -> Path:
    """
    Saves a copy of the assembly to work_dir with suffix appended.
    Closes the original (without saving) and opens the copy.
    Returns: (copy_path, new_doc)
    """
    print(f"\n[STAGE 3] Saving assembly copy")

    work_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(doc.GetTitle).stem
    copy_name = f"{original_name}{suffix}.sldasm"
    copy_path = work_dir / copy_name

    # SaveAs3(FileName, SaveAsVersion, Options)
    # 0 = current version, 2 = swSaveAsOptions_Copy
    doc.SaveAs3(str(copy_path), 0, 2)

    print(f"  Saved copy: {copy_path}")

    # Close original without saving
    sw.CloseDoc(doc.GetTitle)
    print(f"  Closed original (no save)")

    # Open the copy
    copy_doc = open_assembly(sw, copy_path)
    return copy_path, copy_doc


# ============================================================
# STAGE 4: Traverse & suppress components
# ============================================================
def get_all_components(doc) -> list:
    """
    Gets all components in the assembly tree.
    Works with COM late binding (no QueryInterface).
    """
    try:
        # GetComponents(TopLevelOnly): False = flatten entire tree
        components = doc.GetComponents(False)
    except Exception:
        try:
            # Fallback: top-level only
            components = doc.GetComponents(True)
        except Exception as e:
            print(f"  [ERROR] Cannot get components: {e}")
            return []

    if components is None:
        return []
    return list(components)


def resolve_suppress_key(name: str, suppress_list: list) -> str:
    """
    Returns the matched key from SUPPRESS_LIST/SUPPRESS_LIBRARY, or "" if no match.
    Checks exact name first, then base name (without instance suffix).
    """
    if name in suppress_list:
        return name
    if "-" in name:
        base_name = name.rsplit("-", 1)[0]
        if base_name in suppress_list:
            return base_name
    return ""


def should_suppress_component(comp, mode_hidden: bool, mode_list: bool, suppress_list: list) -> Tuple[bool, str, str]:
    """
    Determines whether a component should be suppressed.
    Returns: (should_suppress, reason, matched_library_key)
    """
    name = comp.Name2
    reasons = []
    matched_key = ""

    if mode_list:
        matched_key = resolve_suppress_key(name, suppress_list)
        if matched_key:
            reasons.append("in list")

    if mode_hidden:
        visible = comp.Visible
        if visible == swComponentHidden:
            reasons.append("hidden")

    if reasons:
        return True, "+".join(reasons), matched_key
    return False, "", ""


def suppress_components(doc, log: list):
    """
    Walks the component tree, suppresses based on config flags,
    logs each component status.
    """
    print(f"\n[STAGE 4] Processing component tree")
    print(f"  Suppress hidden:  {SUPPRESS_HIDDEN}")
    print(f"  Suppress by list: {SUPPRESS_BY_LIST}")

    components = get_all_components(doc)
    print(f"  Total components found: {len(components)}")

    suppressed_count = 0
    kept_count = 0

    for comp in components:
        name = comp.Name2
        comp_type = "Assembly" if comp.GetChildrenCount > 0 else "Part"

        do_suppress, reason, lib_key = should_suppress_component(
            comp,
            mode_hidden=SUPPRESS_HIDDEN,
            mode_list=SUPPRESS_BY_LIST,
            suppress_list=SUPPRESS_LIST,
        )

        if do_suppress:
            comp.SetSuppression2(swComponentSuppressed)
            suppressed_count += 1

            # Look up bolt specs from SUPPRESS_LIBRARY
            lib_entry = SUPPRESS_LIBRARY.get(lib_key, None) if lib_key else None

            if lib_entry and lib_entry.get("bolts"):
                # One row per bolt type
                for bolt in lib_entry["bolts"]:
                    log.append({
                        "Component":              name,
                        "Quick Descript":          lib_entry.get("Quick Descript", ""),
                        "Size of bolt":           bolt["Size of bolt"],
                        "Torque [N/m]":           bolt["Torque [N/m]"],
                        "Numbers of tightenings": bolt["Numbers of tightenings"],
                        "Confidence":             "known",
                        "Source":                 "suppressed",
                    })
            else:
                # No bolt specs — just log the component
                log.append({
                    "Component":              name,
                    "Quick Descript":          lib_entry.get("Quick Descript", "") if lib_entry else "",
                    "Size of bolt":           "",
                    "Torque [N/m]":           "",
                    "Numbers of tightenings": "",
                    "Confidence":             "known",
                    "Source":                 "suppressed",
                })

            print(f"    [Suppressed] {comp_type:8s} | {name} ({reason})")
        else:
            kept_count += 1
            print(f"    [Kept      ] {comp_type:8s} | {name}")

    print(f"\n  Summary: {kept_count} kept, {suppressed_count} suppressed")
    return log


# ============================================================
# STAGE 5: Save log to Excel
# ============================================================
def save_component_log(log: List[Dict], excel_path: Path):
    """
    Saves suppressed components to Excel with their known properties.
    Log entries already contain bolt specs from SUPPRESS_LIBRARY.
    If nothing was suppressed, creates an empty Excel with headers.
    """
    print(f"\n[STAGE 5] Saving component log: {excel_path}")

    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

    REPORT_COLUMNS = [
        "Component", "Quick Descript", "Size of bolt",
        "Torque [N/m]", "Numbers of tightenings", "Confidence", "Source",
    ]

    if log:
        df = pd.DataFrame(log, columns=REPORT_COLUMNS)
    else:
        df = pd.DataFrame(columns=REPORT_COLUMNS)

    df.to_excel(excel_path, index=False)

    # Format
    wb = load_workbook(excel_path)
    ws = wb.active
    ws.title = "Components"

    header_font = Font(name="Arial", bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")
        cell.border = thin_border

    for row in range(2, ws.max_row + 1):
        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = Font(name="Arial", size=11)
            cell.alignment = Alignment(horizontal="center")
            cell.border = thin_border

    for col_cells in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = max(max_len + 3, 14)

    ws.freeze_panes = "A2"
    wb.save(excel_path)
    print(f"  Saved: {excel_path} ({len(log)} suppressed components)")


# ============================================================
# STAGE 6: Save as .sldprt
# ============================================================
def save_as_part(sw, doc, work_dir: Path) -> Path:
    """
    Saves current assembly as .sldprt (part file).
    """
    print(f"\n[STAGE 6] Saving as .sldprt")

    name = Path(doc.GetTitle).stem
    prt_path = work_dir / f"{name}.sldprt"

    # SaveAs3(FileName, SaveAsVersion, Options)
    doc.SaveAs3(str(prt_path), 0, 2)

    print(f"  Saved: {prt_path}")
    return prt_path


# ============================================================
# STAGE 7: Save as .stl
# ============================================================
def save_as_stl(sw, doc, work_dir: Path) -> Path:
    """
    Saves current document as .stl.
    """
    print(f"\n[STAGE 7] Exporting as .stl")

    name = Path(doc.GetTitle).stem
    stl_path = work_dir / f"{name}.stl"

    # SaveAs3(FileName, SaveAsVersion, Options)
    doc.SaveAs3(str(stl_path), 0, 2)

    print(f"  Saved: {stl_path}")
    return stl_path


# ============================================================
# STAGE 8: Run inference (STUB)
# ============================================================
def run_inference(stl_path: Path, enabled: bool = False):
    """
    Optionally runs infer.py on the exported STL.

    Args:
        stl_path: path to the exported .stl file
        enabled:  set to True to activate

    When enabled, this will:
      1. Update infer.py config to point to stl_path
      2. Run inference
      3. Run export_excel to generate the report
    """
    if not enabled:
        print(f"\n[STAGE 8] Auto-inference DISABLED")
        print(f"  To run manually:")
        print(f"    1. Set INPUT_MODE='stl' and STL_PATH='{stl_path}' in infer.py")
        print(f"    2. python infer.py")
        return

    print(f"\n[STAGE 8] Running inference on: {stl_path}")

    # Option A: subprocess
    import subprocess
    result = subprocess.run(
        [sys.executable, str(INFER_SCRIPT)],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"  [ERROR] Inference failed:\n{result.stderr}")

    # Option B: direct import (uncomment if preferred)
    # from infer import main as infer_main
    # infer_main()


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    import win32com.client
    import pythoncom

    print("=" * 60)
    print("SolidWorks Assembly Preprocessing Pipeline")
    print("=" * 60)

    # --- Connect ---
    sw = connect_solidworks()

    # --- Stage 1: Open ---
    doc = open_assembly(sw, SLDASM_PATH)

    # --- Stage 2: Display State ---
    if SET_DISPLAY_STATE:
        set_display_state(doc, DISPLAY_STATE_NAME)
    else:
        print("\n[STAGE 2] Display State change DISABLED")

    # --- Stage 3: Copy (optional) ---
    if MAKE_COPY:
        copy_path, doc = save_assembly_copy(sw, doc, WORK_DIR, COPY_SUFFIX)
        print(f"  Working on copy: {copy_path}")
    else:
        print("\n[STAGE 3] Copy DISABLED — working directly on opened file")

    # --- Stage 4: Suppress ---
    component_log = []
    suppress_components(doc, component_log)

    # --- Stage 5: Log to Excel ---
    save_component_log(component_log, EXCEL_LOG_PATH)

    # --- Stage 6: Rebuild + Save as .sldprt ---
    print("\n[STAGE 6] Rebuild + save as .sldprt")
    doc.ForceRebuild3(True)

    prt_path = save_as_part(sw, doc, WORK_DIR)

    # Close assembly, open part
    sw.CloseDoc(doc.GetTitle)

    errors = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warnings = win32com.client.VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    prt_doc = sw.OpenDoc6(str(prt_path), swDocPART, 0, "", errors, warnings)

    # --- Stage 7: Save as .stl ---
    stl_path = save_as_stl(sw, prt_doc, WORK_DIR)

    # --- Stage 8: Inference ---
    run_inference(stl_path, enabled=AUTO_RUN_INFER)

    # --- Done ---
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Part file:      {prt_path}")
    print(f"  STL file:       {stl_path}")
    print(f"  Component log:  {EXCEL_LOG_PATH}")
    if MAKE_COPY:
        print(f"  Assembly copy:  {copy_path}")
    if AUTO_RUN_INFER:
        print(f"  Inference:      DONE")
    else:
        print(f"  Inference:      SKIPPED (set AUTO_RUN_INFER=True to enable)")


if __name__ == "__main__":
    main()