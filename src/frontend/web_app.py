import os
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

ALLOWED_EXTENSIONS = ["pdf", "pptx", "docx"]

# Maximum number of files allowed in one batch.
MAX_FILES = 20

# Maximum size of each individual file.
MAX_FILE_SIZE_MB = 50
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# Maximum number of requests running concurrently.
MAX_WORKERS = 5

# Delay between STARTING requests.
REQUEST_START_DELAY_SECONDS = 2


# ============================================================
# Dimension helpers
# ============================================================

def normalize_dimensions(dimensions):
    """
    Normalize an observation's dimensions into a dictionary.

    Supported examples:

        None
            -> {}

        {}
            -> {}

        {
            "region": "Europe",
            "segment": "Retail"
        }
            -> same dictionary

        [
            {"region": "Europe"},
            {"segment": "Retail"}
        ]
            -> {
                "region": "Europe",
                "segment": "Retail"
            }

    If dimensions cannot be interpreted as key-value pairs,
    an empty dictionary is returned.
    """

    if dimensions is None:
        return {}

    # Normal case:
    # dimensions is already a dictionary.
    if isinstance(dimensions, dict):
        return dimensions

    # Defensive support for a list of dictionaries.
    if isinstance(dimensions, list):

        normalized = {}

        for item in dimensions:

            if isinstance(item, dict):
                normalized.update(item)

        return normalized

    return {}


def get_dimension_keys_for_observations(observations):
    """
    Get the dimension keys ONLY for one observations list.

    This is intentionally NOT calculated across the entire
    dataframe.

    Example:

        observations = [
            {
                "dimensions": {
                    "region": "Europe",
                    "segment": "Retail"
                }
            },
            {
                "dimensions": {
                    "region": "Asia",
                    "segment": "Wholesale"
                }
            }
        ]

    returns:

        ["region", "segment"]

    If another KPI has:

        {
            "dimensions": {
                "scope": "Group"
            }
        }

    "scope" will NOT be added to the first observations list.
    """

    dimension_keys = []

    seen = set()

    for observation in observations:

        dimensions = normalize_dimensions(
            observation.get("dimensions")
        )

        for key in dimensions.keys():

            key = str(key)

            if key not in seen:

                seen.add(key)
                dimension_keys.append(key)

    return dimension_keys


def get_dimension_keys_for_dataframe(measure_df):
    """
    Get dimension keys for one measure dataframe only.

    The dataframe is expected to contain a 'dimensions'
    column where each cell contains a dictionary or None.

    This is used by the Excel export.

    IMPORTANT:
    This function is called separately for every measure_df,
    so dimensions remain independent between observation lists.
    """

    dimension_keys = []

    seen = set()

    if "dimensions" not in measure_df.columns:
        return dimension_keys

    for dimensions in measure_df["dimensions"]:

        dimensions = normalize_dimensions(
            dimensions
        )

        for key in dimensions.keys():

            key = str(key)

            if key not in seen:

                seen.add(key)
                dimension_keys.append(key)

    return dimension_keys


# ============================================================
# Flatten API response
# ============================================================

def flatten_extractions(response_data):
    """
    Convert the nested extraction response into one row per
    KPI observation.

    IMPORTANT:

    Dimensions are deliberately NOT flattened here.

    They are kept as dictionaries so that they can later be
    expanded independently for each observations/measure list.

    Example:

        dimensions:
        {
            "region": "Europe",
            "segment": "Retail"
        }

    stays as:

        {
            "region": "Europe",
            "segment": "Retail"
        }

    inside the dataframe.
    """

    rows = []

    for extraction in response_data.get(
        "extractions",
        [],
    ):

        source = extraction.get("source")
        label = extraction.get("label")

        data = extraction.get("data") or {}

        document = data.get("document") or {}

        kpis = data.get("kpis") or []

        company = document.get("company")
        holding = document.get("holding")

        for kpi in kpis:

            kpi_name = kpi.get("name")

            observations = kpi.get(
                "observations"
            ) or []

            for observation in observations:

                dimensions = normalize_dimensions(
                    observation.get("dimensions")
                )

                rows.append(
                    {
                        "source": source,
                        "company": company,
                        "holding": holding,
                        "label": label,
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
                        "unit": observation.get("unit"),

                        # Keep dimensions as a dictionary.
                        #
                        # Do NOT convert this to JSON text.
                        "dimensions": dimensions,
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
# Build display dataframe for ONE observations list
# ============================================================

def build_observation_display_df(measure_df):
    """
    Build the Streamlit dataframe for ONE measure/observations
    list.

    The dimension columns are calculated ONLY from this
    measure_df.

    Therefore dimensions from another KPI/measure will not
    appear here.
    """

    display_rows = []

    # --------------------------------------------------------
    # Find dimension keys ONLY within this observations list
    # --------------------------------------------------------

    dimension_keys = get_dimension_keys_for_dataframe(
        measure_df
    )

    # --------------------------------------------------------
    # Build rows
    # --------------------------------------------------------

    for _, row in measure_df.iterrows():

        display_row = {
            "Period": row.get("period"),
            "Period Type": row.get("period_type"),
            "Value": row.get("value"),
            "Currency": (
                row.get("currency")
                if pd.notna(row.get("currency"))
                else "—"
            ),
            "Unit": (
                row.get("unit")
                if pd.notna(row.get("unit"))
                else "—"
            ),
        }

        # ----------------------------------------------------
        # Add only dimensions belonging to THIS observations
        # list.
        # ----------------------------------------------------

        dimensions = normalize_dimensions(
            row.get("dimensions")
        )

        for dimension_key in dimension_keys:

            value = dimensions.get(
                dimension_key
            )

            if value is None:

                display_row[dimension_key] = "—"

            else:

                display_row[dimension_key] = value

        display_rows.append(
            display_row
        )

    return pd.DataFrame(display_rows)


# ============================================================
# Display KPI-centric grouped view
# ============================================================

def display_kpi_view(df):
    """
    Display KPI observations using:

        Source document
            -> Page / label
                -> KPI
                    -> Measure
                        -> Observations

    Each measure/observations list gets its OWN independent
    set of dimension columns.
    """

    if df.empty:

        st.warning(
            "No KPI observations found."
        )

        return

    # ========================================================
    # Source
    # ========================================================

    for source, source_df in df.groupby(
        "source",
        sort=False,
        dropna=False,
    ):

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

            # =================================================
            # Company / holding / page
            # =================================================

            page_groups = source_df.groupby(
                [
                    "company",
                    "holding",
                    "label",
                ],
                sort=False,
                dropna=False,
            )

            for (
                company,
                holding,
                label,
            ), page_df in page_groups:

                company_name = (
                    str(company)
                    if pd.notna(company)
                    else "Unknown company"
                )

                page_name = (
                    str(label)
                    if pd.notna(label)
                    else "Unknown"
                )

                if (
                    pd.notna(holding)
                    and holding
                ):

                    page_title = (
                        f"📄 Page {page_name} · "
                        f"{company_name} · "
                        f"Holding: {holding} "
                        f"({len(page_df)} observations)"
                    )

                else:

                    page_title = (
                        f"📄 Page {page_name} · "
                        f"{company_name} "
                        f"({len(page_df)} observations)"
                    )

                with st.expander(
                    page_title,
                    expanded=True,
                ):

                    # =========================================
                    # KPI
                    # =========================================

                    for kpi, kpi_df in page_df.groupby(
                        "kpi",
                        sort=False,
                        dropna=False,
                    ):

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

                            # =================================
                            # Measure
                            # =================================

                            for measure, measure_df in kpi_df.groupby(
                                "measure",
                                sort=False,
                                dropna=False,
                            ):

                                measure_name = (
                                    str(measure)
                                    if pd.notna(measure)
                                    else "Unknown"
                                )

                                st.markdown(
                                    f"**Measure: "
                                    f"`{measure_name}`**"
                                )

                                # ---------------------------------
                                # IMPORTANT:
                                #
                                # This dataframe contains ONLY
                                # this particular observations list.
                                #
                                # Therefore the dimensions are
                                # determined independently.
                                # ---------------------------------

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


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="VeroPort",
    page_icon="📊",
    layout="wide",
)

st.title("📊 VeroPort Extract")

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

        total_files = len(uploaded_files)

        progress = st.progress(0)

        status_text = st.empty()

        results = []

        # ====================================================
        # Submit requests with 2-second delay
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

                    # Wait before every request except
                    # the first.
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

                    results.append(result)

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

        for result in results:

            filename = result["filename"]

            # ------------------------------------------------
            # Request failed
            # ------------------------------------------------

            if not result["success"]:

                request_failures.append(
                    {
                        "file": filename,
                        "error": result["error"],
                    }
                )

                continue

            successful_files += 1

            response_data = (
                result["response_data"]
                or {}
            )

            # ------------------------------------------------
            # KPI observations
            # ------------------------------------------------

            file_df = flatten_extractions(
                response_data
            )

            if not file_df.empty:

                all_rows.append(file_df)

            # ------------------------------------------------
            # Skipped
            # ------------------------------------------------

            skipped = response_data.get(
                "skipped",
                [],
            )

            if skipped:

                all_skipped.extend(
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

                all_failed_extractions.extend(
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

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Sources",
            result_df["source"].nunique(),
        )

    with col2:

        st.metric(
            "Companies",
            result_df["company"].nunique(),
        )

    with col3:

        st.metric(
            "KPIs",
            result_df["kpi"].nunique(),
        )

    with col4:

        st.metric(
            "Observations",
            len(result_df),
        )

    # ========================================================
    # Downloads
    # ========================================================

    st.divider()

    st.subheader("Export")

    # ========================================================
    # Excel
    # ========================================================

    excel_path = "kpi_observations.xlsx"

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

        # ----------------------------------------------------
        # Source groups
        # ----------------------------------------------------

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

            # =================================================
            # Page groups
            # =================================================

            page_groups = source_df.groupby(
                [
                    "company",
                    "holding",
                    "label",
                ],
                sort=False,
                dropna=False,
            )

            for (
                company,
                holding,
                label,
            ), page_df in page_groups:

                company_name = (
                    str(company)
                    if pd.notna(company)
                    else "Unknown company"
                )

                page_name = (
                    str(label)
                    if pd.notna(label)
                    else "Unknown"
                )

                if (
                    pd.notna(holding)
                    and holding
                ):

                    page_title = (
                        f"📄 Page {page_name} · "
                        f"{company_name} · "
                        f"Holding: {holding} "
                        f"({len(page_df)} observations)"
                    )

                else:

                    page_title = (
                        f"📄 Page {page_name} · "
                        f"{company_name} "
                        f"({len(page_df)} observations)"
                    )

                worksheet.cell(
                    current_row,
                    1,
                    page_title,
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

                for kpi, kpi_df in page_df.groupby(
                    "kpi",
                    sort=False,
                    dropna=False,
                ):

                    kpi_name = (
                        str(kpi)
                        if pd.notna(kpi)
                        else "Unnamed KPI"
                    )

                    observation_count = len(
                        kpi_df
                    )

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

                    for measure, measure_df in kpi_df.groupby(
                        "measure",
                        sort=False,
                        dropna=False,
                    ):

                        measure_name = (
                            str(measure)
                            if pd.notna(measure)
                            else "Unknown"
                        )

                        worksheet.cell(
                            current_row,
                            1,
                            f"Measure: "
                            f"{measure_name}",
                        )

                        worksheet.cell(
                            current_row,
                            1,
                        ).font = Font(
                            bold=True,
                            italic=True,
                        )

                        current_row += 1

                        # =================================================
                        # IMPORTANT:
                        #
                        # Determine dimensions ONLY for this
                        # particular measure/observations list.
                        # =================================================

                        dimension_columns = (
                            get_dimension_keys_for_dataframe(
                                measure_df
                            )
                        )

                        # -------------------------------------------------
                        # Base headers
                        # -------------------------------------------------

                        headers = [
                            "Period",
                            "Period Type",
                            "Value",
                            "Currency",
                            "Unit",
                        ]

                        # -------------------------------------------------
                        # Add ONLY this observation list's
                        # dimension columns.
                        # -------------------------------------------------

                        headers.extend(
                            dimension_columns
                        )

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

                            values = [
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
                                    row.get(
                                        "currency"
                                    )
                                    if pd.notna(
                                        row.get(
                                            "currency"
                                        )
                                    )
                                    else "—"
                                ),
                                (
                                    row.get(
                                        "unit"
                                    )
                                    if pd.notna(
                                        row.get(
                                            "unit"
                                        )
                                    )
                                    else "—"
                                ),
                            ]

                            # ---------------------------------------------
                            # Get dimensions for THIS observation.
                            # ---------------------------------------------

                            dimensions = normalize_dimensions(
                                row.get(
                                    "dimensions"
                                )
                            )

                            # ---------------------------------------------
                            # Add dimensions in the exact same order
                            # as the headers.
                            # ---------------------------------------------

                            for dimension in dimension_columns:

                                dimension_value = (
                                    dimensions.get(
                                        dimension
                                    )
                                )

                                if dimension_value is None:

                                    dimension_value = "—"

                                values.append(
                                    dimension_value
                                )

                            # ---------------------------------------------
                            # Write row
                            # ---------------------------------------------

                            for col, value in enumerate(
                                values,
                                start=1,
                            ):

                                if pd.isna(value):

                                    value = None

                                worksheet.cell(
                                    current_row,
                                    col,
                                    value,
                                )

                            current_row += 1

                        # Blank row after observations.
                        current_row += 1

                    # Blank row after KPI.
                    current_row += 1

                # Blank row after page.
                current_row += 1

            # Blank row after source.
            current_row += 1

        # ========================================================
        # Excel formatting
        # ========================================================

        # Base widths.
        widths = {
            1: 18,   # Period
            2: 18,   # Period Type
            3: 18,   # Value
            4: 16,   # Currency
            5: 16,   # Unit
        }

        # --------------------------------------------------------
        # Since different sections can have different dimension
        # columns, scan the worksheet headers and calculate a
        # sensible width for every column.
        # --------------------------------------------------------

        for column_cells in worksheet.iter_cols():

            max_length = 0

            for cell in column_cells:

                if cell.value is not None:

                    value_length = len(
                        str(cell.value)
                    )

                    max_length = max(
                        max_length,
                        value_length,
                    )

            column_index = (
                column_cells[0].column
            )

            # Keep widths reasonable.
            width = max(
                12,
                min(
                    45,
                    max_length + 2,
                ),
            )

            worksheet.column_dimensions[
                get_column_letter(
                    column_index
                )
            ].width = width

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
        label="⬇️ Download observations Excel",
        data=excel_data,
        file_name="kpi_observations.xlsx",
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
    )

    # ========================================================
    # KPI-centric grouped view
    # ========================================================

    st.header("KPI Observations")

    display_kpi_view(
        result_df
    )
