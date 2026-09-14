import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================

load_dotenv()

WORKSPACE_URL = os.getenv("WORKSPACE_URL")
MODAL_KEY = os.getenv("MODAL_KEY")
MODAL_SECRET = os.getenv("MODAL_SECRET")

if not WORKSPACE_URL:
    st.error("WORKSPACE_URL is missing from .env")
    st.stop()

if not MODAL_KEY:
    st.error("MODAL_KEY is missing from .env")
    st.stop()

if not MODAL_SECRET:
    st.error("MODAL_SECRET is missing from .env")
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


# ============================================================
# Tag helpers
# ============================================================

def normalize_tags(dimensions):
    """
    Normalize the API dimensions field into a flat list
    of strings.

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


def format_tags(tags):
    """
    Convert a list of tags into a display-friendly string.
    """

    normalized = normalize_tags(tags)

    if not normalized:
        return "—"

    return ", ".join(normalized)


# ============================================================
# Page helper
# ============================================================

def extract_page_number(label):
    """
    Extract page information from the API extraction label.

    The API response places page information in:
        extractions[*]["label"]

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

    The original label is retained separately as
    extraction_label for traceability.
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
# Flatten API response
# ============================================================

def flatten_extractions(response_data, default_source):
    """
    Convert the backend response structure into one row
    per KPI observation.

    Expected structure:

        {
            "units": [...],
            "skipped": [],
            "extractions": [
                {
                    "source": "...",
                    "label": "...",
                    "data": {
                        "document": {
                            "company": "...",
                            "holding": "...",
                            "reporting_period": "...",
                            "source": "..."
                        },
                        "kpis": [
                            {
                                "name": "...",
                                "observations": [
                                    {
                                        "measure": "...",
                                        "period": "...",
                                        "period_type": "...",
                                        "value": "...",
                                        "currency": "...",
                                        "unit": "...",
                                        "dimensions": [...]
                                    }
                                ]
                            }
                        ]
                    }
                }
            ],
            "failed_extractions": []
        }

    Returns one dataframe row per observation.

    Each row contains:
        - source
        - source_document
        - page_number
        - extraction_source
        - extraction_label
        - company
        - holding
        - reporting_period
        - kpi
        - measure
        - period
        - period_type
        - value
        - currency
        - unit
        - tags
    """

    rows = []

    if not isinstance(response_data, dict):
        return pd.DataFrame()

    extractions = response_data.get("extractions") or []

    # --------------------------------------------------------
    # Process every extraction
    # --------------------------------------------------------

    for extraction in extractions:

        if not isinstance(extraction, dict):
            continue

        extraction_source = (
            extraction.get("source")
            or default_source
        )

        extraction_label = extraction.get(
            "label"
        )

        # Page information comes from extraction label
        page_number = extract_page_number(
            extraction_label
        )

        data = extraction.get("data") or {}

        if not isinstance(data, dict):
            continue

        # ----------------------------------------------------
        # Document metadata
        # ----------------------------------------------------

        document = data.get("document") or {}

        if not isinstance(document, dict):
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

        # Prefer document source when available.
        source = (
            document_source
            or extraction_source
            or default_source
        )

        source_document = source

        # ----------------------------------------------------
        # KPIs are nested inside data
        # ----------------------------------------------------

        kpis = data.get("kpis") or []

        if not isinstance(kpis, list):
            continue

        # ----------------------------------------------------
        # Process KPIs
        # ----------------------------------------------------

        for kpi in kpis:

            if not isinstance(kpi, dict):
                continue

            kpi_name = kpi.get(
                "name"
            )

            observations = kpi.get(
                "observations"
            ) or []

            if not isinstance(observations, list):
                continue

            # ------------------------------------------------
            # Process observations
            # ------------------------------------------------

            for observation in observations:

                if not isinstance(observation, dict):
                    continue

                dimensions = observation.get(
                    "dimensions"
                )

                tags = normalize_tags(
                    dimensions
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

                        "kpi": kpi_name,

                        "measure": observation.get(
                            "measure"
                        ),

                        "period": observation.get(
                            "period"
                        ),

                        "period_type": observation.get(
                            "period_type"
                        ),

                        "value": observation.get(
                            "value"
                        ),

                        "currency": observation.get(
                            "currency"
                        ),

                        "unit": observation.get(
                            "unit"
                        ),

                        "tags": tags,
                    }
                )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


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

        file_bytes = uploaded_file.getvalue()

        files = {
            "files": (
                filename,
                file_bytes,
                uploaded_file.type
                or "application/octet-stream",
            )
        }

        # ====================================================
        # PRODUCTION API REQUEST
        # ====================================================
        
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
# Build observation display dataframe
# ============================================================

def build_observation_display_df(measure_df):
    """
    Build the Streamlit dataframe for observations.
    """

    display_rows = []

    for _, row in measure_df.iterrows():

        display_rows.append(
            {
                "Source Document": (
                    row.get("source_document")
                    if pd.notna(
                        row.get("source_document")
                    )
                    else "—"
                ),

                "Page": (
                    row.get("page_number")
                    if pd.notna(
                        row.get("page_number")
                    )
                    and row.get("page_number") is not None
                    else "—"
                ),

                "Period": row.get(
                    "period"
                ),

                "Period Type": row.get(
                    "period_type"
                ),

                "Value": row.get(
                    "value"
                ),

                "Currency": (
                    row.get("currency")
                    if pd.notna(
                        row.get("currency")
                    )
                    else "—"
                ),

                "Unit": (
                    row.get("unit")
                    if pd.notna(
                        row.get("unit")
                    )
                    else "—"
                ),

                "Tags": format_tags(
                    row.get("tags")
                ),
            }
        )

    return pd.DataFrame(
        display_rows
    )


# ============================================================
# Display KPI-centric grouped view
# ============================================================

def display_kpi_view(df):
    """
    Display KPI observations grouped strictly by:

        Source document
            -> Page
                -> KPI
                    -> Measure
                        -> Observations

    This prevents KPIs from different pages from being mixed.
    """

    if df.empty:

        st.warning(
            "No KPI observations found."
        )

        return

    # ========================================================
    # Source document
    # ========================================================

    source_groups = df.groupby(
        "source",
        sort=False,
        dropna=False,
    )

    for source, source_df in source_groups:

        source_name = (
            str(source)
            if pd.notna(source)
            else "Unknown source"
        )

        with st.expander(
            f"📄 {source_name} "
            f"({len(source_df)} observations)",
            expanded=True,
        ):

            # ------------------------------------------------
            # Document metadata
            # ------------------------------------------------

            metadata_row = source_df.iloc[0]

            company = metadata_row.get(
                "company"
            )

            holding = metadata_row.get(
                "holding"
            )

            reporting_period = metadata_row.get(
                "reporting_period"
            )

            metadata_parts = []

            if pd.notna(company) and company:

                metadata_parts.append(
                    f"**Company:** {company}"
                )

            if pd.notna(holding) and holding:

                metadata_parts.append(
                    f"**Holding:** {holding}"
                )

            if (
                pd.notna(reporting_period)
                and reporting_period
            ):

                metadata_parts.append(
                    f"**Reporting period:** "
                    f"{reporting_period}"
                )

            if metadata_parts:

                st.markdown(
                    "  \n".join(
                        metadata_parts
                    )
                )

            st.divider()

            # =================================================
            # PAGE
            # =================================================

            page_groups = source_df.groupby(
                "page_number",
                sort=False,
                dropna=False,
            )

            for page_number, page_df in page_groups:

                if (
                    pd.isna(page_number)
                    or page_number is None
                ):

                    page_name = "Page Unknown"

                else:

                    page_name = (
                        f"Page {page_number}"
                    )

                st.subheader(
                    f"📑 {page_name}"
                )

                # =================================================
                # KPI
                # =================================================

                kpi_groups = page_df.groupby(
                    "kpi",
                    sort=False,
                    dropna=False,
                )

                for kpi, kpi_df in kpi_groups:

                    kpi_name = (
                        str(kpi)
                        if pd.notna(kpi)
                        else "Unnamed KPI"
                    )

                    observation_count = len(
                        kpi_df
                    )

                    with st.expander(
                        f"**{kpi_name}** "
                        f"({observation_count} observations)",
                        expanded=True,
                    ):

                        # =========================================
                        # Measure
                        # =========================================

                        measure_groups = kpi_df.groupby(
                            "measure",
                            sort=False,
                            dropna=False,
                        )

                        for measure, measure_df in measure_groups:

                            measure_name = (
                                str(measure)
                                if pd.notna(measure)
                                else "Unknown"
                            )

                            st.markdown(
                                f"**Measure: "
                                f"`{measure_name}`**"
                            )

                            observation_df = (
                                build_observation_display_df(
                                    measure_df
                                )
                            )

                            st.dataframe(
                                observation_df,
                                use_container_width=True,
                                hide_index=True,
                            )

                st.divider()


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="VeroPort",
    page_icon="📊",
    layout="wide",
)

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
        if uploaded_file.size > MAX_FILE_SIZE_BYTES
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

                for index, uploaded_file in enumerate(
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

                for future, filename in futures:

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
                        completed / total_files
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
            # Validate response structure
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

            extractions = response_data.get(
                "extractions"
            ) or []

            total_extractions += len(
                extractions
            )

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

                # --------------------------------------------
                # Show useful diagnostic information
                # --------------------------------------------

                if total_extractions == 0:

                    st.info(
                        "The API returned no items in "
                        "`extractions`."
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
    # Downloads
    # ========================================================

    st.divider()

    st.subheader(
        "Export"
    )

    # ========================================================
    # Excel
    # ========================================================

    excel_path = (
        "kpi_observations.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        from openpyxl.styles import (
            Alignment,
            Font,
            PatternFill,
        )

        from openpyxl.utils import (
            get_column_letter,
        )

        workbook = writer.book

        # ----------------------------------------------------
        # Create worksheet
        # ----------------------------------------------------

        worksheet = workbook.create_sheet(
            "KPI Observations"
        )

        current_row = 1

        # ====================================================
        # Source groups
        # ====================================================

        source_groups = result_df.groupby(
            "source",
            sort=False,
            dropna=False,
        )

        for source, source_df in source_groups:

            source_name = (
                str(source)
                if pd.notna(source)
                else "Unknown source"
            )

            # ------------------------------------------------
            # Source heading
            # ------------------------------------------------

            worksheet.cell(
                current_row,
                1,
                f"📄 {source_name} "
                f"({len(source_df)} observations)",
            )

            worksheet.cell(
                current_row,
                1,
            ).font = Font(
                bold=True,
                size=14,
            )

            current_row += 1

            # ------------------------------------------------
            # Document metadata
            # ------------------------------------------------

            metadata_row = source_df.iloc[0]

            metadata = [
                (
                    "Company",
                    metadata_row.get(
                        "company"
                    ),
                ),
                (
                    "Holding",
                    metadata_row.get(
                        "holding"
                    ),
                ),
                (
                    "Reporting Period",
                    metadata_row.get(
                        "reporting_period"
                    ),
                ),
            ]

            for label, value in metadata:

                if (
                    value is not None
                    and pd.notna(value)
                    and str(value).strip()
                ):

                    worksheet.cell(
                        current_row,
                        1,
                        label,
                    ).font = Font(
                        bold=True
                    )

                    worksheet.cell(
                        current_row,
                        2,
                        value,
                    )

                    current_row += 1

            current_row += 1

            # =================================================
            # PAGE GROUPS
            # =================================================

            page_groups = source_df.groupby(
                "page_number",
                sort=False,
                dropna=False,
            )

            for page_number, page_df in page_groups:

                if (
                    pd.isna(page_number)
                    or page_number is None
                ):

                    page_name = "Page Unknown"

                else:

                    page_name = (
                        f"Page {page_number}"
                    )

                # ------------------------------------------------
                # Page heading
                # ------------------------------------------------

                worksheet.cell(
                    current_row,
                    1,
                    f"📑 {page_name}",
                )

                worksheet.cell(
                    current_row,
                    1,
                ).font = Font(
                    bold=True,
                    size=12,
                )

                current_row += 1

                # =================================================
                # KPI groups
                # =================================================

                kpi_groups = page_df.groupby(
                    "kpi",
                    sort=False,
                    dropna=False,
                )

                for kpi, kpi_df in kpi_groups:

                    kpi_name = (
                        str(kpi)
                        if pd.notna(kpi)
                        else "Unnamed KPI"
                    )

                    observation_count = len(
                        kpi_df
                    )

                    # ------------------------------------------------
                    # KPI heading
                    # ------------------------------------------------

                    worksheet.cell(
                        current_row,
                        1,
                        f"{kpi_name} "
                        f"({observation_count} observations)",
                    )

                    worksheet.cell(
                        current_row,
                        1,
                    ).font = Font(
                        bold=True,
                        size=11,
                    )

                    current_row += 1

                    # =================================================
                    # Measure groups
                    # =================================================

                    measure_groups = kpi_df.groupby(
                        "measure",
                        sort=False,
                        dropna=False,
                    )

                    for measure, measure_df in measure_groups:

                        measure_name = (
                            str(measure)
                            if pd.notna(measure)
                            else "Unknown"
                        )

                        # ------------------------------------------------
                        # Measure heading
                        # ------------------------------------------------

                        worksheet.cell(
                            current_row,
                            1,
                            f"Measure: {measure_name}",
                        )

                        worksheet.cell(
                            current_row,
                            1,
                        ).font = Font(
                            bold=True,
                            italic=True,
                        )

                        current_row += 1

                        headers = [
                            "Source Document",
                            "Page",
                            "Period",
                            "Period Type",
                            "Value",
                            "Currency",
                            "Unit",
                            "Tags",
                        ]

                        # -------------------------------------------------
                        # Write headers
                        # -------------------------------------------------

                        for col, header in enumerate(
                            headers,
                            start=1,
                        ):

                            cell = worksheet.cell(
                                current_row,
                                col,
                                header,
                            )

                            cell.font = Font(
                                bold=True
                            )

                            cell.fill = PatternFill(
                                fill_type="solid",
                                fgColor="D9EAF7",
                            )

                            cell.alignment = Alignment(
                                horizontal="center"
                            )

                        current_row += 1

                        # =================================================
                        # Write observations
                        # =================================================

                        for _, row in measure_df.iterrows():

                            tags_value = format_tags(
                                row.get("tags")
                            )

                            currency = row.get(
                                "currency"
                            )

                            unit = row.get(
                                "unit"
                            )

                            source_document = row.get(
                                "source_document"
                            )

                            page_number = row.get(
                                "page_number"
                            )

                            values = [
                                (
                                    source_document
                                    if pd.notna(
                                        source_document
                                    )
                                    and source_document is not None
                                    else "—"
                                ),

                                (
                                    page_number
                                    if pd.notna(
                                        page_number
                                    )
                                    and page_number is not None
                                    else "—"
                                ),

                                row.get(
                                    "period"
                                ),

                                row.get(
                                    "period_type"
                                ),

                                row.get(
                                    "value"
                                ),

                                (
                                    currency
                                    if pd.notna(
                                        currency
                                    )
                                    else "—"
                                ),

                                (
                                    unit
                                    if pd.notna(
                                        unit
                                    )
                                    else "—"
                                ),

                                tags_value,
                            ]

                            for col, value in enumerate(
                                values,
                                start=1,
                            ):

                                if value is not None:

                                    try:

                                        if pd.isna(
                                            value
                                        ):

                                            value = None

                                    except (
                                        TypeError,
                                        ValueError,
                                    ):

                                        pass

                                worksheet.cell(
                                    current_row,
                                    col,
                                    value,
                                )

                            current_row += 1

                        current_row += 1

                    current_row += 1

                current_row += 1

            current_row += 1

        # ========================================================
        # Excel formatting
        # ========================================================

        for column_cells in worksheet.iter_cols():

            max_length = 0

            for cell in column_cells:

                if cell.value is not None:

                    value_length = len(
                        str(
                            cell.value
                        )
                    )

                    max_length = max(
                        max_length,
                        value_length,
                    )

            column_index = (
                column_cells[0].column
            )

            width = max(
                12,
                min(
                    60,
                    max_length + 2,
                ),
            )

            worksheet.column_dimensions[
                get_column_letter(
                    column_index
                )
            ].width = width

        worksheet.column_dimensions[
            "H"
        ].width = 45

        worksheet.freeze_panes = "A1"

    # ========================================================
    # Download Excel
    # ========================================================

    with open(
        excel_path,
        "rb",
    ) as f:

        excel_data = f.read()

    st.download_button(
        label=(
            "⬇️ Download observations Excel"
        ),
        data=excel_data,
        file_name=(
            "kpi_observations.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
    )

    # ========================================================
    # KPI-centric grouped view
    # ========================================================

    st.header(
        "KPI Observations"
    )

    st.caption(
        "Observations are grouped by source document, "
        "then page, KPI, and measure. Each observation "
        "includes its source document and page number."
    )

    display_kpi_view(
        result_df
    )