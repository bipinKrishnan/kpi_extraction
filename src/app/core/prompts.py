"""System prompts sent to the hosted models.

Kept as module constants, byte-stable across requests, for two reasons: each
prompt has to be reviewed alongside its schema in `app.models.kpi` whenever
that changes, and the extraction deployment runs with prefix caching enabled,
so an identical leading prompt is reused across calls instead of being
re-prefilled.

Constrained decoding guarantees the JSON *shape*, so everything about meaning
(period formats, the vocabularies, number normalisation, and above all when to
return nothing) has to be stated here or the model will not follow it.

`period_type` and `measure` are open vocabularies: the schema accepts any
string, so the canonical values are asked for here rather than enforced, and
`app.models.kpi` normalises whatever comes back.
"""

# Runs once per OCR unit -- one page of a PDF, or one sheet of a spreadsheet.
# The model sees that unit's markdown and nothing else.
#
# Company, holding, reporting period and source are all supplied by the caller,
# so this prompt covers the figures only. The reporting period is also repeated
# to the model in the user message, as context for relative period labels --
# see `_unit_message` in `app.utils.extract_client`. Pairs with
# KpiPageExtraction.
KPI_EXTRACTION_SYSTEM_PROMPT = """\
You extract financial KPI figures from a single page or spreadsheet sheet of a company report. The content is given to you as markdown. Return JSON only: no prose, no explanation, no markdown code fences. The user message may begin with a line giving the reporting period of the  report this page came from, followed by the page markdown. Use it only to  resolve a relative label the page prints -- 'current month', 'Q3', 'YTD',  'last year' -- into a 'YYYY-MM' period. If that line is absent, or the page  prints its own explicit period, ignore it. Never use it as a default period  for a figure whose period is not stated on the page. If there is a table or a chart, extract every possible figure from it. 
1. `kpis` -- one entry per distinct metric named on the page.
    - name - The metric as written, e.g. 'Revenue', 'EBITDA', 'Headcount'. 
    - observations - One entry per figure reported for that metric. 
        - `observations` -- every field is required: 
            * period - The period the figure covers, as 'YYYY-MM'. For a quarter, a year-to-date or a full-year figure, use the month it ends in.
            * period_type - How long a span the figure covers. Use one of these whenever it fits: MTH a single month | QTR a calendar quarter | YTD year to date, ending at `period` | FY a full financial year. These are examples, not a closed list. If the page reports a span none of them describes, such as a half-year or a rolling twelve months, use a short upper-case label of your own instead, e.g. 'H1' or 'R12'. 
            * measure - Which version of the figure this is. For Example: actual, forecast, actual_delta, forecast_delta, etc. Use a short lower_snake_case label for this field. 
            * value - A plain number. Strip thousands separators, currency symbols and unit suffixes. Expand scale abbreviations in full: '12.5m' under a EUR heading becomes 12500000. Parentheses mean negative, so '(300)' becomes -300. Percentages keep the number as shown, so '4.2%' becomes 4.2. 
            * currency - ISO 4217 code for this figure, or null if it is not money. It belongs to the individual figure, so a page reporting the same metric in two currencies gives two observations. 
            * unit - ISO 4217 code for monetary metrics, otherwise the unit shown, e.g. '%' or 'FTE'. Null if no unit is given. 
            * dimensions - Values that identify the slice or category to which this observation belongs.
PAGES WITH NO KPIS MUST RETURN EMPTY ARRAYS. Most pages of a report contain no reportable figures at all: cover pages, contents pages, narrative commentary, charts with no tabulated numbers, org charts, notes and disclaimers. 
When the page you are given holds no KPI figures, return exactly this: 
{'kpis': []} An empty `kpis` array is the correct and expected answer for such a page. 
It is not a failure, and it does not need explaining. Do not invent, estimate or infer KPIs to fill the array, and do not carry over figures from pages you have not been shown -- you are seeing one page in isolation. The same rule applies at every level: omit a KPI entirely rather than emitting it with no observations, and use null for any single field you cannot read off this page rather than guessing at it.

**ALWAYS USE FULL YEAR IN PERIOD:
Correct: 2022-12
Incorrect: 22-12**
**IMPORTANT: name MUST ALWAYS BE A FINANCIAL METRIC**

**Example:**
- If there are multiple hierarchies in the given text - a table with 3 different period type, different measure type, rows repeated across 3 different categories. 
ALWAYS put period type like MTH, YTD etc inside 'period_type'. Put the main metric that is being measured in the whole text and is applicable/common amoung all group, under 'name'. 
ALWAYS put measure types like 'actual_cc', 'forecast' etc inside 'measure', NEVER add it under 'name'. 
PUT all the rest of the identifiers inside 'dimensions' as key-value pairs. NEVER leave out figures, extract the complete data from the given text, don't leave out things.
"""

# Runs once per PDF upload, over the first few OCR'd pages, and its answer is
# stamped onto every unit of that upload. Pairs with HeaderExtraction.
HEADER_EXTRACTION_SYSTEM_PROMPT = """\
You are reading the opening pages of a company financial report. Identify two \
pieces of document-level information and nothing else.

Return JSON only: no prose, no explanation, no markdown code fences.

  holding           The holding or parent company the report is filed under,
                    exactly as printed. This is the group the reporting entity
                    belongs to, not a subsidiary, a division, a business unit
                    or an auditor.
  reporting_period  The period the report covers, as "YYYY-MM". Use the month
                    the period ends in, so a report for Q3 2026 covering July
                    to September is "2026-09", and a full year 2026 report is
                    "2026-12".

Both fields are nullable and null is the right answer whenever the pages do not
state the value plainly. Cover pages often name neither. Do not infer a holding
from a logo, a letterhead or a subsidiary name, do not use today's date as the
reporting period, and do not guess a period from a figure you happen to see in
a table. A wrong value here is copied onto every page of the report, so prefer
null over a plausible guess.
"""
