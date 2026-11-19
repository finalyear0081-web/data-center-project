"""
Universal File Ingestion Parser
Supports CSV, JSON, XLSX, and XLS formats without hard external dependencies.
Includes built-in XML/ZIP parser for XLSX when openpyxl is not installed.
"""
import os
import io
import json
import zipfile
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, Any, List, Optional
import pandas as pd
import numpy as np

def parse_xlsx_builtin(file_path_or_bytes: Any) -> pd.DataFrame:
    """
    Pure Python parser for Microsoft Excel OpenXML (.xlsx) files.
    Reads sharedStrings.xml and worksheet sheet1.xml without openpyxl.
    """
    if isinstance(file_path_or_bytes, bytes):
        archive = zipfile.ZipFile(io.BytesIO(file_path_or_bytes))
    elif hasattr(file_path_or_bytes, "read"):
        content = file_path_or_bytes.read()
        archive = zipfile.ZipFile(io.BytesIO(content))
    else:
        archive = zipfile.ZipFile(file_path_or_bytes)

    # 1. Load shared strings table if present
    shared_strings: List[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        tree = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        # XML namespace for OpenXML
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for si in tree.findall("ns:si", ns):
            # Text can be in <t> or nested in <r><t>
            text_parts = [t.text or "" for t in si.findall(".//ns:t", ns)]
            shared_strings.append("".join(text_parts))

    # 2. Find first worksheet
    sheet_name = None
    for name in archive.namelist():
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml"):
            sheet_name = name
            break
    if not sheet_name:
        raise ValueError("No valid worksheet found in XLSX archive.")

    sheet_tree = ET.fromstring(archive.read(sheet_name))
    ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    sheet_data = sheet_tree.find("ns:sheetData", ns)
    if sheet_data is None:
        return pd.DataFrame()

    rows_data: List[List[Any]] = []
    for row in sheet_data.findall("ns:row", ns):
        row_vals: List[Any] = []
        for c in row.findall("ns:c", ns):
            cell_type = c.get("t", "")
            val_elem = c.find("ns:v", ns)
            val = val_elem.text if val_elem is not None else ""
            if cell_type == "s" and val.isdigit():
                idx = int(val)
                val = shared_strings[idx] if idx < len(shared_strings) else val
            elif cell_type == "b":
                val = (val == "1")
            row_vals.append(val)
        if any(row_vals):
            rows_data.append(row_vals)

    if not rows_data:
        return pd.DataFrame()

    # First row as header
    header = [str(col).strip() if str(col).strip() else f"col_{i}" for i, col in enumerate(rows_data[0])]
    records = []
    for r in rows_data[1:]:
        # Pad row if ragged
        padded = r + [None] * (len(header) - len(r))
        records.append(padded[:len(header)])

    df = pd.DataFrame(records, columns=header)
    # Infer numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

def parse_dataset_file(file_path: str) -> pd.DataFrame:
    """
    Parses any supported tabular dataset file (CSV, JSON, XLSX, XLS) into a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".csv":
        # Try sniffing delimiter
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                sep = ";" if (";" in first_line and "," not in first_line) else (
                    "\t" if ("\t" in first_line and "," not in first_line) else ","
                )
            return pd.read_csv(file_path, sep=sep)
        except Exception:
            return pd.read_csv(file_path)

    elif ext in [".json"]:
        try:
            return pd.read_json(file_path)
        except Exception:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try finding list inside dict (e.g. data['readings'])
                for k, v in data.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        return pd.DataFrame(v)
                return pd.DataFrame([data])
            raise ValueError("Unsupported JSON layout for tabular time-series telemetry.")

    elif ext in [".xlsx"]:
        try:
            return pd.read_excel(file_path)
        except Exception:
            # Fallback to built-in pure Python OpenXML parser
            return parse_xlsx_builtin(file_path)

    elif ext in [".xls"]:
        try:
            return pd.read_excel(file_path)
        except Exception:
            # If plain text/html/xml masquerading as XLS
            try:
                return pd.read_csv(file_path, sep="\t")
            except Exception:
                return parse_xlsx_builtin(file_path)

    else:
        # Fallback to CSV reader
        return pd.read_csv(file_path)
