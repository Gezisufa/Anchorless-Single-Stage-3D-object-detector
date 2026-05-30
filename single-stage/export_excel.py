# export_excel.py
# Merges suppressed (known) + detected (inference) into a single Excel report.
#
# Output columns: Component | Class_ID | Size of bolt | Numbers of tightenings | Nm
#
# Usage:
#   python export_excel.py
#
# Dependencies:
#   pip install openpyxl pandas

import pandas as pd
from pathlib import Path
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side

# ============================================================
# CONFIG
# ============================================================
DETECTIONS_XLSX  = Path("out/detections_raw.xlsx")        # from infer.py
SUPPRESSED_XLSX  = Path("out/components_log.xlsx")         # from sw_preprocess.py
OUTPUT_XLSX      = Path("out/detections_report.xlsx")      # final merged report

# ============================================================
# CLASS LIBRARY — for detected (inference) components
# Each class_id maps to bolt specifications.
# One detection can have multiple bolt types (multiple rows).
#
# Assembly: SG883252R
# ============================================================
CLASS_LIBRARY = {
    0: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Two Desks Screw", "Numbers of tightenings": 1, "Nm": 45},
        ],
    },
    1: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Two Desks Screw s Betweener'er", "Numbers of tightenings": 1, "Nm": 45},
        ],
    },
    2: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Client's screw", "Numbers of tightenings": 0, "Nm": 0},
        ],
    },
    3: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Client's screw", "Numbers of tightenings": 0, "Nm": 0},
        ],
    },
    4: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Isolator Double", "Numbers of tightenings": 4, "Nm": 45},
            {"Size of bolt": "M12", "Quick Case": "White support for Isolator Double", "Numbers of tightenings": 2, "Nm": 35},
        ],
    },
    5: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Isolator Double Support", "Numbers of tightenings": 4, "Nm": 45},
        ],
    },
    6: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Isolator Double Support Wider", "Numbers of tightenings": 4, "Nm": 45},
        ],
    },
    7: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Isolator Single with Air", "Numbers of tightenings": 2, "Nm": 45},
            {"Size of bolt": "M12", "Quick Case": "White support for Isolator Double", "Numbers of tightenings": 1, "Nm": 35},
        ],
    },
    8: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Isolator Single Support", "Numbers of tightenings": 2, "Nm": 45},
        ],
    },
    9: {
        "bolts": [
            {"Size of bolt": "M12", "Quick Case": "Isolator Single Support Wider", "Numbers of tightenings": 2, "Nm": 45},
        ],
    },
    # Add more classes:
    # 1: {
    #     "bolts": [
    #         {"Size of bolt": "Mxx", "Quick Case": "...", "Numbers of tightenings": xx, "Nm": xx},
    #     ],
    # },
}

_DEFAULT_ENTRY = {

    "bolts": [{"Size of bolt": "-", "Quick Case": "-", "Numbers of tightenings": "-", "Nm": "-"}],
}

REPORT_COLUMNS = [
    "Component", "Class_ID", "Quick Case", "Size of bolt", "Numbers of tightenings", "Nm",
]


def get_class_entry(class_id: int) -> dict:
    return CLASS_LIBRARY.get(class_id, _DEFAULT_ENTRY)


def load_suppressed(path: Path) -> pd.DataFrame:
    """Loads suppressed components log from sw_preprocess.py (already in report format)."""
    if not path.exists():
        print(f"  [INFO] No suppressed log at: {path}")
        return pd.DataFrame(columns=REPORT_COLUMNS)

    df = pd.read_excel(path)
    print(f"  Loaded {len(df)} suppressed rows from: {path}")
    return df


def expand_detections(path: Path) -> pd.DataFrame:
    """Loads raw detections and expands into report rows."""
    if not path.exists():
        print(f"  [INFO] No detections at: {path}")
        return pd.DataFrame(columns=REPORT_COLUMNS)

    df = pd.read_excel(path)
    print(f"  Loaded {len(df)} detections from: {path}")

    rows = []
    for det_idx, (_, det) in enumerate(df.iterrows(), start=1):
        cls_id = int(det["class_id"])
        entry = get_class_entry(cls_id)

        for bolt_idx, bolt in enumerate(entry["bolts"]):
            rows.append({
                "Component":              f"*NN detection {det_idx}*" if bolt_idx == 0 else "",
                "Class_ID":               cls_id,
                "Quick Case":             bolt.get("Quick Case", ""),
                "Size of bolt":           bolt["Size of bolt"],
                "Numbers of tightenings": bolt["Numbers of tightenings"],
                "Nm":                     bolt["Nm"],
            })

    result = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    print(f"  Expanded to {len(result)} rows")
    return result


def main(
    detections_xlsx: Path = DETECTIONS_XLSX,
    suppressed_xlsx: Path = SUPPRESSED_XLSX,
    output_xlsx: Path = OUTPUT_XLSX,
):
    # ----------------------------------------------------------
    # 1. Load both sources
    # ----------------------------------------------------------
    print("Loading data sources:")
    suppressed_df = load_suppressed(suppressed_xlsx)
    detected_df = expand_detections(detections_xlsx)

    # ----------------------------------------------------------
    # 2. Merge: suppressed first, then detected
    # ----------------------------------------------------------
    merged = pd.concat([suppressed_df, detected_df], ignore_index=True)

    if merged.empty:
        merged = pd.DataFrame(columns=REPORT_COLUMNS)

    print(f"\nMerged: {len(merged)} rows "
          f"({len(suppressed_df)} suppressed + {len(detected_df)} detected)")

    # ----------------------------------------------------------
    # 3. Save base Excel
    # ----------------------------------------------------------
    merged.to_excel(output_xlsx, index=False)

    # ----------------------------------------------------------
    # 4. Format with openpyxl
    # ----------------------------------------------------------
    wb = load_workbook(output_xlsx)
    ws = wb.active
    ws.title = "Report"

    header_font = Font(name="Arial", bold=True, size=11, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="4472C4")
    header_align = Alignment(horizontal="center", vertical="center")

    cell_font = Font(name="Arial", size=11)
    cell_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")

    thin_border = Border(
        left=Side(style="thin"), right=Side(style="thin"),
        top=Side(style="thin"), bottom=Side(style="thin"),
    )

    nn_fill = PatternFill("solid", fgColor="E2EFDA")        # light green for NN detections
    supp_fill = PatternFill("solid", fgColor="D6DCE4")       # light grey for suppressed
    comp_font = Font(name="Arial", bold=True, size=11)       # bold for component names

    # headers
    for col in range(1, ws.max_column + 1):
        cell = ws.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_align
        cell.border = thin_border

    # data rows
    for row in range(2, ws.max_row + 1):
        comp_val = str(ws.cell(row=row, column=1).value or "")
        is_nn = comp_val.startswith("*NN")
        is_comp_header = comp_val and not is_nn

        for col in range(1, ws.max_column + 1):
            cell = ws.cell(row=row, column=col)
            cell.font = cell_font
            cell.alignment = cell_align
            cell.border = thin_border

        # component name column: bold + left aligned
        comp_cell = ws.cell(row=row, column=1)
        if is_comp_header:
            comp_cell.font = comp_font
            comp_cell.alignment = left_align
        elif is_nn:
            comp_cell.font = Font(name="Arial", italic=True, size=11)
            comp_cell.alignment = left_align

        # row background
        if is_nn:
            for col in range(1, ws.max_column + 1):
                ws.cell(row=row, column=col).fill = nn_fill
        elif is_comp_header:
            for col in range(1, ws.max_column + 1):
                ws.cell(row=row, column=col).fill = supp_fill

    # auto column widths
    for col_cells in ws.columns:
        max_len = max(len(str(c.value or "")) for c in col_cells)
        ws.column_dimensions[col_cells[0].column_letter].width = max(max_len + 3, 12)

    ws.freeze_panes = "A2"
    wb.save(output_xlsx)
    print(f"\nSaved: {output_xlsx}")


if __name__ == "__main__":
    main()