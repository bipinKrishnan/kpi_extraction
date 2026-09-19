import os
import re
import time
import json
import hmac
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter


# ============================================================
# Configuration
# ============================================================

load_dotenv()

WORKSPACE_URL = os.getenv("WORKSPACE_URL")
MODAL_KEY = os.getenv("MODAL_KEY")
MODAL_SECRET = os.getenv("MODAL_SECRET")

# App authentication key.
#
# IMPORTANT:
# APP_KEY should be supplied by the Modal secret:
#     kpi-streamlit-secrets
#
# The Modal secret should contain:
#     APP_KEY=<your-secret-app-key>
#
# The value is injected into the container environment and is
# never displayed to the user.
APP_KEY = os.getenv("APP_KEY")

if not WORKSPACE_URL:
    st.error("WORKSPACE_URL is missing from environment variables.")
    st.stop()

if not MODAL_KEY:
    st.error("MODAL_KEY is missing from environment variables.")
    st.stop()

if not MODAL_SECRET:
    st.error("MODAL_SECRET is missing from environment variables.")
    st.stop()

if not APP_KEY:
    st.error(
        "APP_KEY is missing. "
        "Make sure the Modal secret 'kpi-streamlit-secrets' "
        "is attached to the Streamlit deployment."
    )
    st.stop()

ENDPOINT = f"{WORKSPACE_URL.rstrip('/')}/extract/text"

ALLOWED_EXTENSIONS = [
    "pdf",
    "pptx",
    "docx",
]

MAX_FILES = 20
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_WORKERS = 5
REQUEST_START_DELAY_SECONDS = 2

# Financial reporting order
MEASURE_ORDER = [
    "actual",
    "actual_ly",
    "budget",
    "forecast",
]

PERIOD_TYPE_ORDER = {
    "MTH": 1,
    "YTD": 2,
    "LTM": 3,
    "FY": 4,
}


# ============================================================
# Authentication
# ============================================================

def authenticate_app():
    """
    Display the login screen and authenticate the user against
    APP_KEY supplied through the Modal secret.

    Authentication state is stored in Streamlit session state.
    """

    # --------------------------------------------------------
    # Initialize authentication state
    # --------------------------------------------------------

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    # --------------------------------------------------------
    # Already authenticated
    # --------------------------------------------------------

    if st.session_state["authenticated"]:

        # Small logout control in the sidebar
        with st.sidebar:
            st.markdown("### Authentication")

            if st.button(
                "Log out",
                use_container_width=True,
            ):
                st.session_state["authenticated"] = False

                # Clear application results as well so that a
                # logged-out user cannot see previously loaded
                # data after returning to the login screen.
                st.session_state.pop(
                    "result_df",
                    None,
                )

                st.rerun()

        return True

    # --------------------------------------------------------
    # Login page
    # --------------------------------------------------------

    st.title("🔐 VeroPort")

    st.markdown(
        """
        ### Application Login

        Enter your application key to continue.
        """
    )

    st.write("")

    # Use a form so pressing Enter submits the login.
    with st.form(
        "login_form",
        clear_on_submit=False,
    ):

        entered_app_key = st.text_input(
            "App Key",
            type="password",
            placeholder="Enter your app key",
            help="Enter the application key provided to you.",
        )

        submitted = st.form_submit_button(
            "Login",
            type="primary",
            use_container_width=True,
        )

    if submitted:

        entered_app_key = entered_app_key.strip()

        if not entered_app_key:

            st.error(
                "Please enter an app key."
            )

            return False

        # ----------------------------------------------------
        # Constant-time comparison
        # ----------------------------------------------------
        #
        # compare_digest avoids a normal string comparison and
        # is preferable for secret/token comparisons.
        #
        # The actual APP_KEY never gets sent to the browser.
        # ----------------------------------------------------

        if hmac.compare_digest(
            entered_app_key,
            APP_KEY,
        ):

            st.session_state[
                "authenticated"
            ] = True

            st.rerun()

        else:

            st.error(
                "Invalid app key."
            )

    return False


# ============================================================
# Streamlit page configuration
# ============================================================

st.set_page_config(
    page_title="VeroPort",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# Authentication Gate
# ============================================================
#
# Nothing below this point is rendered unless the user has
# successfully authenticated.
# ============================================================

if not authenticate_app():
    st.stop()


# ============================================================
# Generic helpers
# ============================================================

def is_missing(value):
    """Safely determine whether a value should be treated as missing."""

    if value is None:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def clean_value(value):
    """Return None for missing values, otherwise preserve the value."""

    if is_missing(value):
        return None

    if isinstance(value, str):
        value = value.strip()

        if not value:
            return None

    return value


def normalize_scalar(value):
    """
    Convert API fields that may accidentally arrive as a one-item
    list/tuple into a scalar.

    Example:
        ["MTH"] -> "MTH"
        "MTH"   -> "MTH"
    """

    if isinstance(value, (list, tuple)):

        if len(value) == 0:
            return None

        if len(value) == 1:
            return value[0]

        # If multiple values arrive where a scalar is expected,
        # retain them in a readable form.
        return ", ".join(
            str(item)
            for item in value
        )

    return value


# ============================================================
# Dimension helpers
# ============================================================

def normalize_dimensions(dimensions):
    """
    Normalize the API dimensions field into a list of strings.

    Supports:
        - None
        - list
        - tuple
        - dictionary
        - scalar values
    """

    if dimensions is None:
        return []

    if isinstance(dimensions, list):

        return [
            str(item)
            for item in dimensions
            if item is not None
        ]

    if isinstance(dimensions, tuple):

        return [
            str(item)
            for item in dimensions
            if item is not None
        ]

    if isinstance(dimensions, dict):

        return [
            str(value)
            for value in dimensions.values()
            if value is not None
        ]

    return [str(dimensions)]


def format_dimensions(dimensions):
    """
    Keep dimensions represented as a JSON-style list in Excel.

    Example:
        ["Europe", "Consumer"]
    """

    normalized = normalize_dimensions(
        dimensions
    )

    return json.dumps(
        normalized,
        ensure_ascii=False,
    )


# ============================================================
# Page helper
# ============================================================

def extract_page_number(label):
    """
    Extract page information from the API extraction label.

    Examples:
        "Page 12"
        "Page: 12"
        "Page 12 of 80"
        "page_12"
        "p. 12"
        "p12"
        "page=12"
        "page-12"
        "pages 12-13"
    """

    if label is None:
        return None

    label_text = str(label).strip()

    # Page range, e.g. "pages 12-13"
    match = re.search(
        r"\bpages?\s*[:=_\-]?\s*(\d+(?:\s*[-–]\s*\d+)?)",
        label_text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    # p. 12 / p12
    match = re.search(
        r"\bp\.?\s*(\d+(?:\s*[-–]\s*\d+)?)\b",
        label_text,
        flags=re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return label_text


# ============================================================
# Value helper
# ============================================================

def convert_value(value):
    """
    Convert numeric-looking API values to actual Python numbers.

    Examples:
        "12500000"       -> 12500000
        "12,500,000"     -> 12500000
        "12.5%"          -> 0.125
        "EUR 12,500,000" -> remains string

    The function intentionally avoids aggressive parsing of
    ambiguous financial values.
    """

    value = clean_value(value)

    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value

    text = str(value).strip()

    # Remove common thousands separators
    numeric_text = text.replace(
        ",",
        "",
    ).strip()

    # Percentage
    if numeric_text.endswith("%"):

        try:

            number = float(
                numeric_text[:-1].strip()
            )

            return number / 100

        except ValueError:

            return value

    # Plain integer / decimal
    try:

        if re.fullmatch(
            r"[-+]?\d+",
            numeric_text,
        ):

            return int(
                numeric_text
            )

        if re.fullmatch(
            r"[-+]?(?:\d+\.\d*|\.\d+)",
            numeric_text,
        ):

            return float(
                numeric_text
            )

    except ValueError:

        pass

    return value


# ============================================================
# Period helpers
# ============================================================

def normalize_period_type(period_type):
    """Normalize period type to uppercase."""

    period_type = normalize_scalar(
        period_type
    )

    if period_type is None:
        return None

    return str(
        period_type
    ).strip().upper()


def normalize_measure(measure):
    """Normalize measure to lowercase."""

    measure = normalize_scalar(
        measure
    )

    if measure is None:
        return None

    return str(
        measure
    ).strip().lower()


def normalize_fx_basis(fx_basis):
    """
    Normalize FX basis into a readable scalar value.

    This supports the same possible API shapes as other scalar
    observation fields, including a one-item list/tuple.
    """

    fx_basis = normalize_scalar(
        fx_basis
    )

    if fx_basis is None:
        return None

    return str(
        fx_basis
    ).strip()


def period_sort_key(period):
    """
    Create a useful chronological sort key.

    Handles common formats such as:
        2026-08
        2026-08-31
        2026
        FY2026
        Q1 2026
    """

    if period is None:
        return (
            9999,
            99,
            99,
            "",
        )

    text = str(
        period
    ).strip().upper()

    # YYYY-MM
    match = re.search(
        r"\b(20\d{2})[-/](\d{1,2})\b",
        text,
    )

    if match:

        year = int(
            match.group(1)
        )

        month = int(
            match.group(2)
        )

        return (
            year,
            month,
            0,
            text,
        )

    # Q1 2026 / 2026 Q1
    match = re.search(
        r"\bQ([1-4])\s*[-/]?\s*(20\d{2})\b",
        text,
    )

    if match:

        quarter = int(
            match.group(1)
        )

        year = int(
            match.group(2)
        )

        return (
            year,
            quarter * 3,
            1,
            text,
        )

    match = re.search(
        r"\b(20\d{2})\s*[-/]?\s*Q([1-4])\b",
        text,
    )

    if match:

        year = int(
            match.group(1)
        )

        quarter = int(
            match.group(2)
        )

        return (
            year,
            quarter * 3,
            1,
            text,
        )

    # FY2026 / 2026
    match = re.search(
        r"\bFY\s*(20\d{2})\b",
        text,
    )

    if match:

        year = int(
            match.group(1)
        )

        return (
            year,
            12,
            2,
            text,
        )

    match = re.fullmatch(
        r"(20\d{2})",
        text,
    )

    if match:

        return (
            int(match.group(1)),
            12,
            2,
            text,
        )

    # Fallback
    return (
        9999,
        99,
        99,
        text,
    )


def column_sort_key(column):
    """
    Sort a financial matrix column by:
        1. Period
        2. Period Type
        3. Measure
    """

    period = column["period"]
    period_type = column["period_type"]
    measure = column["measure"]

    return (
        period_sort_key(
            period
        ),
        PERIOD_TYPE_ORDER.get(
            period_type,
            99,
        ),
        (
            MEASURE_ORDER.index(
                measure
            )
            if measure in MEASURE_ORDER
            else 99
        ),
        str(
            measure or ""
        ),
    )


# ============================================================
# Flatten API response
# ============================================================

def get_extractions(response_data):
    """
    Support both:

    1. Current app response:
        {
            "extractions": [...]
        }

    2. Direct backend KPI response:
        {
            "kpis": [...]
        }

    Returns a normalized list of extraction-like objects.
    """

    if not isinstance(
        response_data,
        dict,
    ):

        return []

    # Current wrapper structure
    if isinstance(
        response_data.get(
            "extractions"
        ),
        list,
    ):

        return (
            response_data.get(
                "extractions"
            )
            or []
        )

    # Direct KPI structure
    if isinstance(
        response_data.get(
            "kpis"
        ),
        list,
    ):

        return [
            {
                "source": None,
                "label": None,
                "data": {
                    "kpis": (
                        response_data.get(
                            "kpis"
                        )
                        or []
                    ),
                },
            }
        ]

    return []


def flatten_extractions(
    response_data,
    default_source,
):
    """
    Convert backend response into one row per KPI observation.

    fx_basis is retained as an observation-level field.
    """

    rows = []

    if not isinstance(
        response_data,
        dict,
    ):

        return pd.DataFrame()

    extractions = get_extractions(
        response_data
    )

    for extraction in extractions:

        if not isinstance(
            extraction,
            dict,
        ):

            continue

        extraction_source = (
            extraction.get(
                "source"
            )
            or default_source
        )

        extraction_label = (
            extraction.get(
                "label"
            )
        )

        page_number = extract_page_number(
            extraction_label
        )

        data = (
            extraction.get(
                "data"
            )
            or {}
        )

        if not isinstance(
            data,
            dict,
        ):

            continue

        # ----------------------------------------------------
        # Document metadata
        # ----------------------------------------------------

        document = (
            data.get(
                "document"
            )
            or {}
        )

        if not isinstance(
            document,
            dict,
        ):

            document = {}

        company = document.get(
            "company"
        )

        holding = document.get(
            "holding"
        )

        reporting_period = document.get(
            "reporting_period"
        )

        document_source = document.get(
            "source"
        )

        source = (
            document_source
            or extraction_source
            or default_source
        )

        source_document = source

        # ----------------------------------------------------
        # KPI list
        # ----------------------------------------------------

        kpis = (
            data.get(
                "kpis"
            )
            or []
        )

        if not isinstance(
            kpis,
            list,
        ):

            continue

        # ----------------------------------------------------
        # KPIs
        # ----------------------------------------------------

        for kpi in kpis:

            if not isinstance(
                kpi,
                dict,
            ):

                continue

            kpi_name = kpi.get(
                "name"
            )

            observations = (
                kpi.get(
                    "observations"
                )
                or []
            )

            if not isinstance(
                observations,
                list,
            ):

                continue

            # ------------------------------------------------
            # Observations
            # ------------------------------------------------

            for observation in observations:

                if not isinstance(
                    observation,
                    dict,
                ):

                    continue

                dimensions = (
                    normalize_dimensions(
                        observation.get(
                            "dimensions"
                        )
                    )
                )

                period = normalize_scalar(
                    observation.get(
                        "period"
                    )
                )

                period_type = (
                    normalize_period_type(
                        observation.get(
                            "period_type"
                        )
                    )
                )

                measure = normalize_measure(
                    observation.get(
                        "measure"
                    )
                )

                value = convert_value(
                    observation.get(
                        "value"
                    )
                )

                unit = normalize_scalar(
                    observation.get(
                        "unit"
                    )
                )

                currency = normalize_scalar(
                    observation.get(
                        "currency"
                    )
                )

                fx_basis = normalize_fx_basis(
                    observation.get(
                        "fx_basis"
                    )
                )

                rows.append(
                    {
                        "source": source,
                        "source_document": source_document,
                        "page_number": page_number,
                        "extraction_source": extraction_source,
                        "extraction_label": extraction_label,
                        "company": company,
                        "holding": holding,
                        "reporting_period": reporting_period,
                        "kpi": clean_value(
                            kpi_name
                        ),
                        "measure": measure,
                        "period": period,
                        "period_type": period_type,
                        "value": value,
                        "currency": currency,
                        "fx_basis": fx_basis,
                        "unit": unit,
                        "dimensions": dimensions,
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        rows
    )


# ============================================================
# Extract a single file
# ============================================================

def extract_single_file(
    uploaded_file,
    headers,
):
    """
    Send exactly one file in exactly one HTTP request.
    """

    filename = uploaded_file.name

    try:

        file_bytes = (
            uploaded_file.getvalue()
        )

        files = {
            "files": (
                filename,
                file_bytes,
                uploaded_file.type
                or "application/octet-stream",
            )
        }

        response = requests.post(
            ENDPOINT,
            headers=headers,
            files=files,
            timeout=3600,
        )

        response.raise_for_status()

        return {
            "filename": filename,
            "success": True,
            "response_data": response.json(),
            "error": None,
        }

    except requests.exceptions.Timeout:

        return {
            "filename": filename,
            "success": False,
            "response_data": None,
            "error": (
                "Request timed out after 1 hour."
            ),
        }

    except requests.exceptions.HTTPError as e:

        status_code = (
            e.response.status_code
            if e.response is not None
            else "unknown"
        )

        response_text = (
            e.response.text
            if e.response is not None
            else ""
        )

        return {
            "filename": filename,
            "success": False,
            "response_data": None,
            "error": (
                f"HTTP {status_code}"
                + (
                    f": {response_text}"
                    if response_text
                    else ""
                )
            ),
        }

    except requests.exceptions.RequestException as e:

        return {
            "filename": filename,
            "success": False,
            "response_data": None,
            "error": (
                f"Request failed: {e}"
            ),
        }

    except ValueError:

        return {
            "filename": filename,
            "success": False,
            "response_data": None,
            "error": (
                "The API returned invalid JSON."
            ),
        }

    except Exception as e:

        return {
            "filename": filename,
            "success": False,
            "response_data": None,
            "error": (
                f"Unexpected error: {e}"
            ),
        }


# ============================================================
# Financial matrix preparation
# ============================================================

def build_financial_matrix(df):

    if df.empty:
        return pd.DataFrame(), []

    working = df.copy()

    for column in [
        "kpi",
        "dimensions",
        "fx_basis",
        "unit",
        "currency",
        "period",
        "period_type",
        "measure",
        "value",
    ]:

        if column not in working.columns:
            working[column] = None

    working["kpi"] = working[
        "kpi"
    ].apply(
        lambda x: (
            str(x).strip()
            if not is_missing(x)
            else "Unnamed KPI"
        )
    )

    working["dimensions"] = working[
        "dimensions"
    ].apply(
        normalize_dimensions
    )

    working["dimensions_display"] = (
        working[
            "dimensions"
        ].apply(
            format_dimensions
        )
    )

    working["fx_basis"] = working[
        "fx_basis"
    ].apply(
        lambda x: (
            str(x).strip()
            if not is_missing(x)
            else ""
        )
    )

    working["unit"] = working[
        "unit"
    ].apply(
        lambda x: (
            str(x).strip()
            if not is_missing(x)
            else ""
        )
    )

    working["period"] = working[
        "period"
    ].apply(
        normalize_scalar
    )

    working["period_type"] = working[
        "period_type"
    ].apply(
        normalize_period_type
    )

    working["measure"] = working[
        "measure"
    ].apply(
        normalize_measure
    )

    working = working[
        working["period"].notna()
        & working["period_type"].notna()
        & working["measure"].notna()
    ].copy()

    if working.empty:
        return pd.DataFrame(), []

    column_records = (
        working[
            [
                "period",
                "period_type",
                "measure",
            ]
        ]
        .drop_duplicates()
        .to_dict("records")
    )

    column_records.sort(
        key=column_sort_key
    )

    working["row_key"] = list(
        zip(
            working["kpi"],
            working[
                "dimensions_display"
            ],
            working["fx_basis"],
            working["unit"],
        )
    )

    grouped_values = (
        working.groupby(
            [
                "row_key",
                "period",
                "period_type",
                "measure",
            ],
            sort=False,
            dropna=False,
        )["value"]
        .apply(
            lambda values: (
                values.iloc[0]
                if len(values) == 1
                else " | ".join(
                    str(v)
                    for v in values
                )
            )
        )
        .reset_index()
    )

    matrix_rows = []

    unique_rows = (
        working[
            [
                "row_key",
                "kpi",
                "dimensions_display",
                "fx_basis",
                "unit",
            ]
        ]
        .drop_duplicates(
            subset=["row_key"]
        )
    )

    for _, base_row in unique_rows.iterrows():

        row = {
            "Metric": base_row[
                "kpi"
            ],
            "Dimensions": base_row[
                "dimensions_display"
            ],
            "FX Basis": base_row[
                "fx_basis"
            ],
            "UoM": base_row[
                "unit"
            ],
        }

        row_key = base_row[
            "row_key"
        ]

        matching_values = (
            grouped_values[
                grouped_values[
                    "row_key"
                ]
                == row_key
            ]
        )

        for column in column_records:

            period = column[
                "period"
            ]

            period_type = column[
                "period_type"
            ]

            measure = column[
                "measure"
            ]

            match = matching_values[
                (
                    matching_values[
                        "period"
                    ]
                    == period
                )
                & (
                    matching_values[
                        "period_type"
                    ]
                    == period_type
                )
                & (
                    matching_values[
                        "measure"
                    ]
                    == measure
                )
            ]

            if match.empty:
                value = None
            else:
                value = match.iloc[0][
                    "value"
                ]

            row[
                (
                    period,
                    period_type,
                    measure,
                )
            ] = value

        matrix_rows.append(
            row
        )

    matrix_df = pd.DataFrame(
        matrix_rows
    )

    if not matrix_df.empty:

        matrix_df = matrix_df.sort_values(
            by=[
                "Metric",
                "Dimensions",
                "FX Basis",
                "UoM",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        )

    return (
        matrix_df,
        column_records,
    )


# ============================================================
# Excel formatting helpers
# ============================================================

def format_measure_label(measure):

    labels = {
        "actual": "Actual",
        "actual_ly": "Actual LY",
        "budget": "Budget",
        "forecast": "Forecast",
    }

    return labels.get(
        measure,
        str(measure).replace(
            "_",
            " ",
        ).title(),
    )


def format_period_header(
    period,
    period_type,
):

    return (
        f"{period} – {period_type}"
    )


def apply_value_number_format(
    cell,
    unit=None,
    value=None,
):

    if value is None:
        return

    unit_text = (
        str(unit).strip().lower()
        if unit is not None
        else ""
    )

    if "%" in unit_text or "percent" in unit_text:

        cell.number_format = "0.0%"

    elif any(
        token in unit_text
        for token in [
            "eur",
            "usd",
            "gbp",
            "chf",
            "jpy",
        ]
    ):

        cell.number_format = "#,##0.00"

    else:

        cell.number_format = "#,##0.00"


# ============================================================
# Create Financial KPIs worksheet
# ============================================================

def create_financial_kpis_sheet(
    workbook,
    result_df,
):

    worksheet = workbook.create_sheet(
        "Financial KPIs"
    )

    matrix_df, column_records = (
        build_financial_matrix(
            result_df
        )
    )

    title_font = Font(
        bold=True,
        size=16,
    )

    metadata_label_font = Font(
        bold=True,
        size=10,
    )

    metadata_value_font = Font(
        size=10,
    )

    header_font = Font(
        bold=True,
        size=10,
    )

    metric_font = Font(
        bold=True,
        size=10,
    )

    thin_gray = Side(
        style="thin",
        color="D9D9D9",
    )

    medium_gray = Side(
        style="medium",
        color="A6A6A6",
    )

    border = Border(
        bottom=thin_gray,
    )

    header_border = Border(
        top=medium_gray,
        bottom=medium_gray,
    )

    period_fill = PatternFill(
        fill_type="solid",
        fgColor="D9EAF7",
    )

    measure_fill = PatternFill(
        fill_type="solid",
        fgColor="EAF2F8",
    )

    metadata_row = (
        result_df.iloc[0]
        if not result_df.empty
        else {}
    )

    company = (
        metadata_row.get(
            "company"
        )
        if hasattr(
            metadata_row,
            "get",
        )
        else None
    )

    holding = (
        metadata_row.get(
            "holding"
        )
        if hasattr(
            metadata_row,
            "get",
        )
        else None
    )

    reporting_period = (
        metadata_row.get(
            "reporting_period"
        )
        if hasattr(
            metadata_row,
            "get",
        )
        else None
    )

    worksheet.merge_cells(
        start_row=1,
        start_column=1,
        end_row=1,
        end_column=max(
            4 + len(column_records),
            4,
        ),
    )

    title_cell = worksheet.cell(
        row=1,
        column=1,
        value="Financial KPIs",
    )

    title_cell.font = title_font

    title_cell.alignment = Alignment(
        horizontal="left",
        vertical="center",
    )

    worksheet.row_dimensions[
        1
    ].height = 24

    metadata_items = []

    if (
        not is_missing(company)
        and str(company).strip()
    ):

        metadata_items.append(
            (
                "Company",
                company,
            )
        )

    if (
        not is_missing(holding)
        and str(holding).strip()
    ):

        metadata_items.append(
            (
                "Holding",
                holding,
            )
        )

    if (
        not is_missing(
            reporting_period
        )
        and str(
            reporting_period
        ).strip()
    ):

        metadata_items.append(
            (
                "Reporting Period",
                reporting_period,
            )
        )

    metadata_row_number = 3

    for label, value in metadata_items:

        worksheet.cell(
            row=metadata_row_number,
            column=1,
            value=label,
        ).font = (
            metadata_label_font
        )

        worksheet.cell(
            row=metadata_row_number,
            column=2,
            value=value,
        ).font = (
            metadata_value_font
        )

        metadata_row_number += 1

    header_start_row = (
        metadata_row_number + 1
    )

    static_headers = [
        "Metric",
        "Dimensions",
        "FX Basis",
        "UoM",
    ]

    for (
        column_index,
        header_name,
    ) in enumerate(
        static_headers,
        start=1,
    ):

        worksheet.cell(
            row=header_start_row,
            column=column_index,
            value=header_name,
        )

        worksheet.merge_cells(
            start_row=header_start_row,
            start_column=column_index,
            end_row=header_start_row + 1,
            end_column=column_index,
        )

        cell = worksheet.cell(
            row=header_start_row,
            column=column_index,
        )

        cell.font = header_font

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        cell.fill = period_fill
        cell.border = header_border

    period_groups = []

    group_start = 5
    previous_group = None

    for index, column in enumerate(
        column_records
    ):

        group = (
            column["period"],
            column["period_type"],
        )

        excel_column = (
            5 + index
        )

        if previous_group is None:

            previous_group = group
            group_start = excel_column

        elif group != previous_group:

            period_groups.append(
                (
                    previous_group,
                    group_start,
                    excel_column - 1,
                )
            )

            previous_group = group
            group_start = excel_column

    if previous_group is not None:

        period_groups.append(
            (
                previous_group,
                group_start,
                4 + len(column_records),
            )
        )

    for (
        group,
        start_column,
        end_column,
    ) in period_groups:

        period, period_type = group

        worksheet.merge_cells(
            start_row=header_start_row,
            start_column=start_column,
            end_row=header_start_row,
            end_column=end_column,
        )

        cell = worksheet.cell(
            row=header_start_row,
            column=start_column,
            value=format_period_header(
                period,
                period_type,
            ),
        )

        cell.font = header_font

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

        cell.fill = period_fill
        cell.border = header_border

        for column in range(
            start_column,
            end_column + 1,
        ):

            worksheet.cell(
                row=header_start_row,
                column=column,
            ).fill = period_fill

            worksheet.cell(
                row=header_start_row,
                column=column,
            ).border = header_border

    for index, column in enumerate(
        column_records
    ):

        excel_column = (
            5 + index
        )

        cell = worksheet.cell(
            row=header_start_row + 1,
            column=excel_column,
            value=format_measure_label(
                column["measure"]
            ),
        )

        cell.font = header_font

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

        cell.fill = measure_fill
        cell.border = header_border

    worksheet.row_dimensions[
        header_start_row
    ].height = 22

    worksheet.row_dimensions[
        header_start_row + 1
    ].height = 22

    data_start_row = (
        header_start_row + 2
    )

    for (
        row_index,
        (_, row),
    ) in enumerate(
        matrix_df.iterrows()
    ):

        excel_row = (
            data_start_row + row_index
        )

        metric_cell = worksheet.cell(
            row=excel_row,
            column=1,
            value=row["Metric"],
        )

        metric_cell.font = metric_font

        metric_cell.alignment = Alignment(
            vertical="center",
        )

        dimensions_cell = worksheet.cell(
            row=excel_row,
            column=2,
            value=row["Dimensions"],
        )

        dimensions_cell.alignment = Alignment(
            vertical="center",
            wrap_text=True,
        )

        fx_basis_cell = worksheet.cell(
            row=excel_row,
            column=3,
            value=row["FX Basis"],
        )

        fx_basis_cell.alignment = Alignment(
            vertical="center",
            wrap_text=True,
        )

        uom_cell = worksheet.cell(
            row=excel_row,
            column=4,
            value=row["UoM"],
        )

        uom_cell.alignment = Alignment(
            vertical="center",
        )

        for index, column in enumerate(
            column_records
        ):

            excel_column = (
                5 + index
            )

            key = (
                column["period"],
                column["period_type"],
                column["measure"],
            )

            value = row.get(
                key
            )

            cell = worksheet.cell(
                row=excel_row,
                column=excel_column,
                value=value,
            )

            cell.alignment = Alignment(
                horizontal="right",
                vertical="center",
            )

            apply_value_number_format(
                cell,
                unit=row["UoM"],
                value=value,
            )

        for column in range(
            1,
            5 + len(column_records),
        ):

            worksheet.cell(
                row=excel_row,
                column=column,
            ).border = border

    worksheet.freeze_panes = (
        f"E{data_start_row}"
    )

    if not matrix_df.empty:

        worksheet.auto_filter.ref = (
            f"A{header_start_row}:"
            f"{get_column_letter(4 + len(column_records))}"
            f"{data_start_row + len(matrix_df) - 1}"
        )

    worksheet.column_dimensions[
        "A"
    ].width = 28

    worksheet.column_dimensions[
        "B"
    ].width = 32

    worksheet.column_dimensions[
        "C"
    ].width = 18

    worksheet.column_dimensions[
        "D"
    ].width = 14

    for column in range(
        5,
        5 + len(column_records),
    ):

        worksheet.column_dimensions[
            get_column_letter(column)
        ].width = 15

    worksheet.sheet_view.showGridLines = False

    worksheet.page_setup.orientation = (
        "landscape"
    )

    worksheet.page_setup.fitToWidth = 1
    worksheet.page_setup.fitToHeight = 0

    worksheet.sheet_properties.pageSetUpPr.fitToPage = True

    worksheet.print_title_rows = (
        f"{header_start_row}:"
        f"{header_start_row + 1}"
    )

    return worksheet


# ============================================================
# Create KPI Detail worksheet
# ============================================================

def create_kpi_detail_sheet(
    workbook,
    result_df,
):

    worksheet = workbook.create_sheet(
        "KPI Detail"
    )

    columns = [
        "Source Document",
        "Page",
        "Company",
        "Holding",
        "Reporting Period",
        "Metric",
        "Dimensions",
        "Period",
        "Period Type",
        "Measure",
        "Value",
        "UoM",
        "Currency",
        "FX Basis",
    ]

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="D9EAF7",
    )

    header_font = Font(
        bold=True,
    )

    thin = Side(
        style="thin",
        color="D9D9D9",
    )

    for (
        column_index,
        column_name,
    ) in enumerate(
        columns,
        start=1,
    ):

        cell = worksheet.cell(
            row=1,
            column=column_index,
            value=column_name,
        )

        cell.font = header_font
        cell.fill = header_fill

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )

        cell.border = Border(
            bottom=thin,
        )

    if result_df.empty:
        return worksheet

    for (
        row_index,
        (_, row),
    ) in enumerate(
        result_df.iterrows(),
        start=2,
    ):

        values = [
            row.get(
                "source_document"
            ),
            row.get(
                "page_number"
            ),
            row.get(
                "company"
            ),
            row.get(
                "holding"
            ),
            row.get(
                "reporting_period"
            ),
            row.get(
                "kpi"
            ),
            format_dimensions(
                row.get(
                    "dimensions"
                )
            ),
            row.get(
                "period"
            ),
            row.get(
                "period_type"
            ),
            format_measure_label(
                row.get(
                    "measure"
                )
            ),
            row.get(
                "value"
            ),
            row.get(
                "unit"
            ),
            row.get(
                "currency"
            ),
            row.get(
                "fx_basis"
            ),
        ]

        for (
            column_index,
            value,
        ) in enumerate(
            values,
            start=1,
        ):

            if is_missing(value):
                value = None

            cell = worksheet.cell(
                row=row_index,
                column=column_index,
                value=value,
            )

            cell.alignment = Alignment(
                vertical="center",
                wrap_text=True,
            )

            if column_index == 11:

                apply_value_number_format(
                    cell,
                    unit=row.get(
                        "unit"
                    ),
                    value=value,
                )

    worksheet.freeze_panes = "A2"

    worksheet.auto_filter.ref = (
        f"A1:{get_column_letter(len(columns))}"
        f"{len(result_df) + 1}"
    )

    widths = {
        "A": 32,
        "B": 10,
        "C": 25,
        "D": 25,
        "E": 20,
        "F": 30,
        "G": 35,
        "H": 15,
        "I": 15,
        "J": 15,
        "K": 18,
        "L": 14,
        "M": 14,
        "N": 18,
    }

    for column, width in widths.items():

        worksheet.column_dimensions[
            column
        ].width = width

    worksheet.sheet_view.showGridLines = False

    return worksheet


# ============================================================
# Create Source Notes worksheet
# ============================================================

def create_source_notes_sheet(
    workbook,
    result_df,
):

    worksheet = workbook.create_sheet(
        "Source Notes"
    )

    columns = [
        "Source Document",
        "Page",
        "Metric",
        "Dimensions",
        "Period",
        "Period Type",
        "Measure",
        "Value",
        "UoM",
        "Currency",
        "FX Basis",
    ]

    header_fill = PatternFill(
        fill_type="solid",
        fgColor="D9EAF7",
    )

    header_font = Font(
        bold=True,
    )

    for (
        column_index,
        column_name,
    ) in enumerate(
        columns,
        start=1,
    ):

        cell = worksheet.cell(
            row=1,
            column=column_index,
            value=column_name,
        )

        cell.font = header_font
        cell.fill = header_fill

        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
        )

    if result_df.empty:
        return worksheet

    for (
        row_index,
        (_, row),
    ) in enumerate(
        result_df.iterrows(),
        start=2,
    ):

        values = [
            row.get(
                "source_document"
            ),
            row.get(
                "page_number"
            ),
            row.get(
                "kpi"
            ),
            format_dimensions(
                row.get(
                    "dimensions"
                )
            ),
            row.get(
                "period"
            ),
            row.get(
                "period_type"
            ),
            format_measure_label(
                row.get(
                    "measure"
                )
            ),
            row.get(
                "value"
            ),
            row.get(
                "unit"
            ),
            row.get(
                "currency"
            ),
            row.get(
                "fx_basis"
            ),
        ]

        for (
            column_index,
            value,
        ) in enumerate(
            values,
            start=1,
        ):

            if is_missing(value):
                value = None

            cell = worksheet.cell(
                row=row_index,
                column=column_index,
                value=value,
            )

            cell.alignment = Alignment(
                vertical="center",
                wrap_text=True,
            )

            if column_index == 8:

                apply_value_number_format(
                    cell,
                    unit=row.get(
                        "unit"
                    ),
                    value=value,
                )

    worksheet.freeze_panes = "A2"

    worksheet.auto_filter.ref = (
        f"A1:{get_column_letter(len(columns))}"
        f"{len(result_df) + 1}"
    )

    widths = {
        "A": 32,
        "B": 10,
        "C": 30,
        "D": 35,
        "E": 15,
        "F": 15,
        "G": 15,
        "H": 18,
        "I": 14,
        "J": 14,
        "K": 18,
    }

    for column, width in widths.items():

        worksheet.column_dimensions[
            column
        ].width = width

    worksheet.sheet_view.showGridLines = False

    return worksheet


# ============================================================
# Build Excel workbook
# ============================================================

def create_excel_workbook(
    result_df
):

    workbook = Workbook()

    default_sheet = workbook.active

    workbook.remove(
        default_sheet
    )

    create_financial_kpis_sheet(
        workbook,
        result_df,
    )

    create_kpi_detail_sheet(
        workbook,
        result_df,
    )

    create_source_notes_sheet(
        workbook,
        result_df,
    )

    workbook.active = (
        workbook.sheetnames.index(
            "Financial KPIs"
        )
    )

    output = BytesIO()

    workbook.save(
        output
    )

    output.seek(0)

    return output.getvalue()


# ============================================================
# Main VeroPort application
# ============================================================

st.title(
    "📊 VeroPort Extract"
)

st.write(
    "Upload financial documents and extract KPI observations. "
    f"You can upload up to {MAX_FILES} files, with each file "
    f"limited to {MAX_FILE_SIZE_MB} MB. Each file is sent as a "
    "separate request. Requests are started 2 seconds apart "
    "and run concurrently."
)


# ============================================================
# File uploader
# ============================================================

uploaded_files = st.file_uploader(
    f"Choose files (maximum {MAX_FILES} files, "
    f"{MAX_FILE_SIZE_MB} MB per file)",
    type=ALLOWED_EXTENSIONS,
    accept_multiple_files=True,
)


# ============================================================
# Validate uploaded files
# ============================================================

if uploaded_files:

    # --------------------------------------------------------
    # Validate number of files
    # --------------------------------------------------------

    if len(uploaded_files) > MAX_FILES:

        st.error(
            f"You selected {len(uploaded_files)} files. "
            f"The maximum allowed is {MAX_FILES} files."
        )

        st.stop()

    # --------------------------------------------------------
    # Validate individual file sizes
    # --------------------------------------------------------

    oversized_files = [
        uploaded_file
        for uploaded_file in uploaded_files
        if uploaded_file.size
        > MAX_FILE_SIZE_BYTES
    ]

    if oversized_files:

        st.error(
            f"{len(oversized_files)} file(s) exceed the "
            f"{MAX_FILE_SIZE_MB} MB per-file limit."
        )

        for uploaded_file in oversized_files:

            size_mb = (
                uploaded_file.size
                / (1024 * 1024)
            )

            st.error(
                f"📄 **{uploaded_file.name}** — "
                f"{size_mb:.2f} MB "
                f"(maximum {MAX_FILE_SIZE_MB} MB)"
            )

        st.stop()

    # --------------------------------------------------------
    # Display selected files
    # --------------------------------------------------------

    st.subheader(
        f"Selected files "
        f"({len(uploaded_files)}/{MAX_FILES})"
    )

    for uploaded_file in uploaded_files:

        size_mb = (
            uploaded_file.size
            / (1024 * 1024)
        )

        st.write(
            f"📄 **{uploaded_file.name}** "
            f"({size_mb:.2f} MB)"
        )

    # ========================================================
    # Process files
    # ========================================================

    if st.button(
        "🚀 Extract KPIs",
        type="primary",
        use_container_width=True,
    ):

        headers = {
            "Modal-Key": MODAL_KEY,
            "Modal-Secret": MODAL_SECRET,
        }

        total_files = len(
            uploaded_files
        )

        progress = st.progress(
            0
        )

        status_text = st.empty()

        results = []

        # ====================================================
        # Submit requests with delay
        # ====================================================

        with st.spinner(
            f"Starting {total_files} "
            f"extraction requests..."
        ):

            with ThreadPoolExecutor(
                max_workers=min(
                    MAX_WORKERS,
                    total_files,
                )
            ) as executor:

                futures = []

                for (
                    index,
                    uploaded_file,
                ) in enumerate(
                    uploaded_files
                ):

                    if index > 0:

                        time.sleep(
                            REQUEST_START_DELAY_SECONDS
                        )

                    future = executor.submit(
                        extract_single_file,
                        uploaded_file,
                        headers,
                    )

                    futures.append(
                        (
                            future,
                            uploaded_file.name,
                        )
                    )

                    status_text.info(
                        f"Started "
                        f"{index + 1}/{total_files}: "
                        f"{uploaded_file.name}"
                    )

                # =============================================
                # Collect results
                # =============================================

                completed = 0

                for (
                    future,
                    filename,
                ) in futures:

                    try:

                        result = future.result()

                    except Exception as e:

                        result = {
                            "filename": filename,
                            "success": False,
                            "response_data": None,
                            "error": (
                                "Unexpected worker error: "
                                f"{e}"
                            ),
                        }

                    results.append(
                        result
                    )

                    completed += 1

                    progress.progress(
                        completed
                        / total_files
                    )

                    if result["success"]:

                        status_text.success(
                            f"Completed "
                            f"{completed}/{total_files}: "
                            f"{filename}"
                        )

                    else:

                        status_text.error(
                            f"Failed "
                            f"{completed}/{total_files}: "
                            f"{filename}"
                        )

        # ====================================================
        # Aggregate results
        # ====================================================

        all_rows = []

        all_skipped = []

        all_failed_extractions = []

        request_failures = []

        successful_files = 0

        total_extractions = 0

        for result in results:

            filename = result[
                "filename"
            ]

            # ------------------------------------------------
            # Request failed
            # ------------------------------------------------

            if not result[
                "success"
            ]:

                request_failures.append(
                    {
                        "file": filename,
                        "error": result[
                            "error"
                        ],
                    }
                )

                continue

            successful_files += 1

            response_data = (
                result[
                    "response_data"
                ]
                or {}
            )

            # ------------------------------------------------
            # Validate response
            # ------------------------------------------------

            if not isinstance(
                response_data,
                dict,
            ):

                all_failed_extractions.append(
                    {
                        "source": filename,
                        "error": (
                            "API response was not a JSON object."
                        ),
                    }
                )

                continue

            # ------------------------------------------------
            # Count extractions
            # ------------------------------------------------

            extractions = (
                response_data.get(
                    "extractions"
                )
                or []
            )

            if extractions:

                total_extractions += len(
                    extractions
                )

            elif isinstance(
                response_data.get(
                    "kpis"
                ),
                list,
            ):

                total_extractions += 1

            # ------------------------------------------------
            # KPI observations
            # ------------------------------------------------

            file_df = flatten_extractions(
                response_data,
                default_source=filename,
            )

            if not file_df.empty:

                all_rows.append(
                    file_df
                )

            # ------------------------------------------------
            # Skipped
            # ------------------------------------------------

            skipped = response_data.get(
                "skipped",
                [],
            )

            if skipped:

                if isinstance(
                    skipped,
                    list,
                ):

                    all_skipped.extend(
                        skipped
                    )

                else:

                    all_skipped.append(
                        skipped
                    )

            # ------------------------------------------------
            # Failed extractions
            # ------------------------------------------------

            failed = response_data.get(
                "failed_extractions",
                [],
            )

            if failed:

                if isinstance(
                    failed,
                    list,
                ):

                    all_failed_extractions.extend(
                        failed
                    )

                else:

                    all_failed_extractions.append(
                        failed
                    )

        # ====================================================
        # Combine KPI observations
        # ====================================================

        if all_rows:

            result_df = pd.concat(
                all_rows,
                ignore_index=True,
                sort=False,
            )

        else:

            result_df = pd.DataFrame()

        # ====================================================
        # Store results
        # ====================================================

        if not result_df.empty:

            st.session_state[
                "result_df"
            ] = result_df

            st.success(
                f"Completed "
                f"{successful_files}/{total_files} files — "
                f"{len(result_df)} observations"
            )

        else:

            st.session_state.pop(
                "result_df",
                None,
            )

            if successful_files:

                st.warning(
                    f"Completed "
                    f"{successful_files}/{total_files} files, "
                    "but no KPI observations were found."
                )

                if total_extractions == 0:

                    st.info(
                        "The API returned no extraction items."
                    )

                else:

                    st.info(
                        f"The API returned "
                        f"{total_extractions} extraction(s), "
                        "but they contained no KPI observations."
                    )

            else:

                st.error(
                    "All extraction requests failed."
                )

        # ====================================================
        # Request failures
        # ====================================================

        if request_failures:

            with st.expander(
                f"❌ Failed requests "
                f"({len(request_failures)})",
                expanded=True,
            ):

                for failure in request_failures:

                    st.error(
                        f"**{failure['file']}** — "
                        f"{failure['error']}"
                    )

        # ====================================================
        # Skipped items
        # ====================================================

        if all_skipped:

            with st.expander(
                f"⚠️ Skipped items "
                f"({len(all_skipped)})"
            ):

                st.json(
                    all_skipped
                )

        # ====================================================
        # Failed extractions
        # ====================================================

        if all_failed_extractions:

            with st.expander(
                f"❌ Failed extractions "
                f"({len(all_failed_extractions)})"
            ):

                st.json(
                    all_failed_extractions
                )


# ============================================================
# Display results
# ============================================================

if "result_df" in st.session_state:

    result_df = st.session_state[
        "result_df"
    ]

    st.divider()

    # ========================================================
    # Summary
    # ========================================================

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.metric(
            "Sources",
            result_df[
                "source"
            ].nunique(),
        )

    with col2:

        st.metric(
            "KPIs",
            result_df[
                "kpi"
            ].nunique(),
        )

    with col3:

        st.metric(
            "Observations",
            len(result_df),
        )

    # ========================================================
    # Financial Preview
    # ========================================================

    st.divider()

    st.subheader(
        "Financial KPI Preview"
    )

    st.caption(
        "KPIs are presented as rows, with dimensions, FX Basis "
        "and UoM as row identifiers. Period and period type are "
        "grouped above Actual, Actual LY, Budget and Forecast."
    )

    matrix_df, column_records = (
        build_financial_matrix(
            result_df
        )
    )

    if not matrix_df.empty:

        preview_df = matrix_df.copy()

        renamed_columns = {}

        for column in preview_df.columns:

            if isinstance(
                column,
                tuple,
            ):

                period = column[0]

                period_type = column[1]

                measure = column[2]

                renamed_columns[
                    column
                ] = (
                    f"{period} – "
                    f"{period_type} – "
                    f"{format_measure_label(measure)}"
                )

        preview_df = preview_df.rename(
            columns=renamed_columns
        )

        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.warning(
            "No valid financial observations found."
        )

    # ========================================================
    # Excel Export
    # ========================================================

    st.divider()

    st.subheader(
        "Export"
    )

    st.write(
        "The Excel workbook contains a financial-report-style "
        "matrix including FX Basis, plus the underlying KPI "
        "detail and source traceability."
    )

    try:

        excel_data = create_excel_workbook(
            result_df
        )

        st.download_button(
            label=(
                "⬇️ Download Financial KPI Excel"
            ),
            data=excel_data,
            file_name=(
                "financial_kpis.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            type="primary",
            use_container_width=True,
        )

    except Exception as e:

        st.error(
            f"Could not create Excel export: {e}"
        )

    # ========================================================
    # Detailed observations
    # ========================================================

    st.divider()

    st.subheader(
        "KPI Detail"
    )

    detail_preview = result_df.copy()

    if "dimensions" in detail_preview.columns:

        detail_preview[
            "dimensions"
        ] = detail_preview[
            "dimensions"
        ].apply(
            format_dimensions
        )

    display_columns = [
        "source_document",
        "page_number",
        "kpi",
        "dimensions",
        "fx_basis",
        "period",
        "period_type",
        "measure",
        "value",
        "unit",
        "currency",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in detail_preview.columns
    ]

    detail_preview = detail_preview[
        available_columns
    ].copy()

    detail_preview = detail_preview.rename(
        columns={
            "source_document": "Source Document",
            "page_number": "Page",
            "kpi": "Metric",
            "dimensions": "Dimensions",
            "fx_basis": "FX Basis",
            "period": "Period",
            "period_type": "Period Type",
            "measure": "Measure",
            "value": "Value",
            "unit": "UoM",
            "currency": "Currency",
        }
    )

    st.dataframe(
        detail_preview,
        use_container_width=True,
        hide_index=True,
    )