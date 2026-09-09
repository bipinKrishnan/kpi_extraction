import os
from pathlib import Path

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

ALLOWED_EXTENSIONS = ["pdf", "pptx", "xlsx", "docx"]


# ============================================================
# Flatten API response
# ============================================================

def flatten_extractions(response_data):
    """
    Convert the nested extraction response into one row per
    KPI observation.

    Output columns:

        source
        company
        holding
        label
        kpi
        measure
        period
        period_type
        value
        currency
        unit
    """

    rows = []

    for extraction in response_data.get("extractions", []):
        source = extraction.get("source")
        label = extraction.get("label")

        data = extraction.get("data") or {}
        document = data.get("document") or {}
        kpis = data.get("kpis") or []

        company = document.get("company")
        holding = document.get("holding")

        for kpi in kpis:
            kpi_name = kpi.get("name")
            unit = kpi.get("unit")

            for observation in kpi.get("observations") or []:
                rows.append(
                    {
                        "source": source,
                        "company": company,
                        "holding": holding,
                        "label": label,
                        "kpi": kpi_name,
                        "measure": observation.get("measure"),
                        "period": observation.get("period"),
                        "period_type": observation.get("period_type"),
                        "value": observation.get("value"),
                        "currency": observation.get("currency"),
                        "unit": unit,
                    }
                )

    return pd.DataFrame(rows)


# ============================================================
# Display KPI-centric grouped view
# ============================================================

def display_kpi_view(df):
    """
    Display KPI observations using a collapsible hierarchy:

        Source document
            -> Page / label
                -> KPI
                    -> Measure
                        -> Observations
    """

    if df.empty:
        st.warning("No KPI observations found.")
        return

    # ========================================================
    # Group by source document
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

        # ----------------------------------------------------
        # SOURCE FOLD
        # ----------------------------------------------------

        with st.expander(
            f"📄 {source_name} "
            f"({len(source_df)} observations)",
            expanded=True,
        ):

            # =================================================
            # Group by page / label
            # =================================================

            page_groups = source_df.groupby(
                ["company", "holding", "label"],
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

                # --------------------------------------------
                # PAGE FOLD
                # --------------------------------------------

                if pd.notna(holding) and holding:
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
                    # Group by KPI
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

                        observation_count = len(kpi_df)

                        # -------------------------------------
                        # KPI FOLD
                        # -------------------------------------

                        with st.expander(
                            f"**{kpi_name}** "
                            f"({observation_count} observations)",
                            expanded=True,
                        ):

                            # ================================
                            # Group by measure
                            # ================================

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
                                    f"**Measure: `{measure_name}`**"
                                )

                                display_rows = []

                                # --------------------------------
                                # Build observation table
                                # --------------------------------

                                for _, row in measure_df.iterrows():

                                    display_rows.append(
                                        {
                                            "Period": row["period"],
                                            "Period Type": row[
                                                "period_type"
                                            ],
                                            "Value": row["value"],
                                            "Currency": (
                                                row["currency"]
                                                if pd.notna(
                                                    row["currency"]
                                                )
                                                else "—"
                                            ),
                                            "Unit": (
                                                row["unit"]
                                                if pd.notna(
                                                    row["unit"]
                                                )
                                                else "—"
                                            ),
                                        }
                                    )

                                observation_df = pd.DataFrame(
                                    display_rows
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
    page_title="KPI Extractor",
    page_icon="📊",
    layout="wide",
)

st.title("📊 KPI Extractor")

st.write(
    "Upload financial documents and extract KPI observations. "
    "Results are grouped by source document, company, page, "
    "KPI, and measure."
)


# ============================================================
# File uploader
# ============================================================

uploaded_files = st.file_uploader(
    "Choose files",
    type=ALLOWED_EXTENSIONS,
    accept_multiple_files=True,
)


# ============================================================
# Process files
# ============================================================

if uploaded_files:

    st.subheader("Selected files")

    for uploaded_file in uploaded_files:
        size_kb = uploaded_file.size / 1024

        st.write(
            f"📄 **{uploaded_file.name}** "
            f"({size_kb:.1f} KB)"
        )

    if st.button(
        "🚀 Extract KPIs",
        type="primary",
        use_container_width=True,
    ):

        all_rows = []

        progress = st.progress(0)

        for file_index, uploaded_file in enumerate(
            uploaded_files
        ):

            st.write(
                f"Processing **{uploaded_file.name}**..."
            )

            try:

                # --------------------------------------------
                # Prepare multipart request
                # --------------------------------------------

                file_bytes = uploaded_file.getvalue()

                files = {
                    "files": (
                        uploaded_file.name,
                        file_bytes,
                        uploaded_file.type
                        or "application/octet-stream",
                    )
                }

                headers = {
                    "Modal-Key": MODAL_KEY,
                    "Modal-Secret": MODAL_SECRET,
                }

                # --------------------------------------------
                # Call extraction API
                # --------------------------------------------

                with st.spinner(
                    f"Extracting {uploaded_file.name}..."
                ):

                    response = requests.post(
                        ENDPOINT,
                        headers=headers,
                        files=files,
                        timeout=3600,  # 1 hour
                    )

                response.raise_for_status()

                response_data = response.json()

                # --------------------------------------------
                # Flatten response
                # --------------------------------------------

                df = flatten_extractions(
                    response_data
                )

                if not df.empty:
                    all_rows.append(df)

                    st.success(
                        f"Completed: "
                        f"{uploaded_file.name} — "
                        f"{len(df)} observations"
                    )
                else:
                    st.warning(
                        f"No KPI observations found in "
                        f"{uploaded_file.name}"
                    )

                # --------------------------------------------
                # Show skipped / failed extractions
                # --------------------------------------------

                skipped = response_data.get(
                    "skipped",
                    [],
                )

                failed = response_data.get(
                    "failed_extractions",
                    [],
                )

                if skipped:
                    with st.expander(
                        f"⚠️ Skipped items "
                        f"({len(skipped)})"
                    ):
                        st.json(skipped)

                if failed:
                    with st.expander(
                        f"❌ Failed extractions "
                        f"({len(failed)})"
                    ):
                        st.json(failed)

            # ------------------------------------------------
            # Error handling
            # ------------------------------------------------

            except requests.exceptions.Timeout:

                st.error(
                    f"Timed out after 1 hour: "
                    f"{uploaded_file.name}"
                )

            except requests.exceptions.HTTPError:

                st.error(
                    f"HTTP error for "
                    f"{uploaded_file.name}: "
                    # f"{response.status_code}"
                )

                # if response.text:
                #     st.code(response.text)

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Request failed for "
                    f"{uploaded_file.name}: {e}"
                )

            except ValueError:

                st.error(
                    f"The API returned invalid JSON for "
                    f"{uploaded_file.name}."
                )

                if response.text:
                    st.code(response.text)

            except Exception as e:

                st.error(
                    f"Unexpected error processing "
                    f"{uploaded_file.name}: {e}"
                )

            finally:

                progress.progress(
                    (file_index + 1)
                    / len(uploaded_files)
                )

        # ====================================================
        # Combined results
        # ====================================================

        if all_rows:

            result_df = pd.concat(
                all_rows,
                ignore_index=True,
            )

            # Store dataframe in session state so it remains
            # available after Streamlit reruns.
            st.session_state["result_df"] = result_df


# ============================================================
# Display results
# ============================================================

if "result_df" in st.session_state:

    result_df = st.session_state["result_df"]

    st.divider()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # KPI-centric grouped view
    # --------------------------------------------------------

    st.header("KPI Observations")

    display_kpi_view(result_df)

    # ========================================================
    # Downloads
    # ========================================================

    st.divider()

    st.subheader("Export")

    # Normalized CSV
    csv_data = result_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download observations CSV",
        data=csv_data,
        file_name="kpi_observations.csv",
        mime="text/csv",
    )

    # Excel
    excel_path = "kpi_observations.xlsx"

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:

        result_df.to_excel(
            writer,
            index=False,
            sheet_name="Observations",
        )

    with open(excel_path, "rb") as f:
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