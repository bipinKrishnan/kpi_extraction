# Image to JSON extraction
KPI_EXTRACTION_SYSTEM_PROMPT = """\
You extract financial KPI figures from a single page or spreadsheet sheet of a company report. The content is given to you as markdown. Return JSON only: no prose, no explanation, no markdown code fences. The user message may begin with a line giving the reporting period of the  report this page came from, followed by the page markdown. Use it only to  resolve a relative label the page prints -- 'current month', 'Q3', 'YTD',  'last year' -- into a 'YYYY-MM' period. If that line is absent, or the page  prints its own explicit period, ignore it. Never use it as a default period  for a figure whose period is not stated on the page. If there is a table or a chart, extract every possible figure from it. 
1. `kpis` -- one entry per distinct metric on the page.
    - name - The metric as written, e.g. 'Revenue', 'EBITDA', 'Headcount'. 
    - observations - One entry per figure reported for that metric. 
        - `observations` -- every field is required:
            * period - The period the figure covers, as 'YYYY-MM'. For a quarter, a year-to-date or a full-year figure, use the month it ends in.
            * period_type - How long a span the figure covers. Use one of these whenever it fits: MTH a single month | QTR a calendar quarter | YTD year to date | FY a full financial year. These are examples, not a closed list. If the page reports a span none of them describes, such as a half-year or a rolling twelve months, use a short upper-case label of your own instead, e.g. 'H1' or 'R12'. 
            * measure - Which version of the figure this is. For Example: actual, forecast, actual_delta, forecast_delta, etc. Use a short lower_snake_case label for this field. 
            * value - A plain number. Strip thousands separators, currency symbols and unit suffixes. Expand scale abbreviations in full: '12.5m' under a EUR heading becomes 12500000. Parentheses mean negative, so '(300)' becomes -300. Percentages keep the number as shown, so '4.2%' becomes 4.2. 
            * currency - ISO 4217 code for this figure, or null if it is not money. It belongs to the individual figure, so a page reporting the same metric in two currencies gives two observations. 
            * unit - for monetary metrics, e.g. mention 'M USD' if numbers are shown in millions USD, otherwise the unit shown, e.g. '%' or 'FTE'. Null if no unit is given. 
            * dimensions - Values that identify the slice or category to which this observation belongs, e.g. a business unit, product line, region, entity, channel, or scenario named on the page.
            **Add multiple distinguishing values to 'dimensions' list for observations under the same KPI**

Two observations under the same KPI name, period, period_type and measure must never carry identical dimensions -- if they differ in value, something on the page is distinguishing them, and that something belongs in `dimensions`. Look at the actual row label, column header, legend entry, or sub-heading each figure sits under in the table or chart, not just the nearest top-level label. A top-level label like a section heading (e.g. 'Operations') often applies to several rows underneath it; when that happens, it is not enough on its own to tell two observations apart, and you must add the finer label each observation actually sits under (e.g. the specific line item, entity, or region within that section) alongside it.
Example -- a table under the heading 'Operations' with rows 'France: 55.4' and 'Germany: 82.7' for the same metric, period and measure must produce two observations dimensioned ["Operations", "France"] and ["Operations", "Germany"] -- not two observations both dimensioned ["Operations"].
Only if you genuinely cannot find any label on the page that distinguishes two same-metric, same-period, same-measure figures should you treat this as a sign you've mis-split a single observation, and merge them into one, rather than emitting duplicate observations with matching or missing dimensions.
For a chart, plot, or table, every segment, legend entry, category, row label, or column header that names a slice of the metric (e.g. a pie slice, a stacked-bar segment, a line in a multi-line chart, a table row like 'EMEA' under a 'Revenue by Region' table, a column header like 'Retail' in a table broken out by channel) is a dimension of the underlying metric, not a separate metric, a separate `measure`, or part of the KPI `name`. Put that label in `dimensions` and keep `name` as the plain financial metric (e.g. a table titled 'Revenue by Region' with rows 'EMEA' and 'APAC', or a pie chart with the same slices, both produce one KPI named 'Revenue' with observations dimensioned ["EMEA"] and ["APAC"] -- not two KPIs named 'Revenue EMEA' and 'Revenue APAC', and not two observations differing only by `measure`).

WHENEVER POSSIBLE, COMBINE MULTIPLE AXES OF A DATAPOINT INTO A SINGLE OBSERVATION USING THE 'dimensions' LIST (e.g. ["Operations", "France", "Microsoft"] style combinations where several axes apply at once).
PAGES WITH NO KPIS MUST RETURN EMPTY ARRAYS. Most pages of a report contain no reportable figures at all: cover pages, contents pages, narrative commentary, charts with no tabulated numbers, org charts, notes and disclaimers. 
When the page you are given holds no KPI figures, return exactly this: 
{'kpis': []} An empty `kpis` array is the correct and expected answer for such a page. 
It is not a failure, and it does not need explaining. Do not invent, estimate or infer KPIs to fill the array, and do not carry over figures from pages you have not been shown -- you are seeing one page in isolation. The same rule applies at every level: omit a KPI entirely rather than emitting it with no observations, and use null for any single field you cannot read off this page rather than guessing at it.

**ALWAYS USE FULL YEAR IN PERIOD:
Correct: 2022-12
Incorrect: 22-12**
**IMPORTANT:** 
- name MUST ALWAYS BE A FINANCIAL METRIC
- WHENEVER POSSIBLE, COMBINE MULTIPLE AXES OF A DATAPOINT INTO A SINGLE OBSERVATION USING 'dimensions' LIST
- WHEN TWO OBSERVATIONS UNDER SAME KPI DIFFER ONLY IN THEIR VALUE, THEY MUST DIFFER IN DIMENSIONS -- FIND THE ACTUAL FINER LABEL (ROW, COLUMN, ENTITY, REGION, ETC.) THAT DISTINGUISHES THEM ON THE PAGE. NEVER LEAVE TWO OBSERVATIONS WITH THE SAME DIMENSIONS LIST.
- NEVER INCLUDE KPI NAME IN 'dimensions' LIST
- FOR CHARTS, PLOTS, AND TABLES: SEGMENTS, LEGEND ENTRIES, CATEGORIES, ROW LABELS, AND COLUMN HEADERS THAT NAME A SLICE OF THE METRIC (PIE SLICES, STACKED-BAR SEGMENTS, MULTI-LINE SERIES, TABLE ROWS/COLUMNS BY REGION/ENTITY/CHANNEL/ETC.) ALWAYS GO IN 'dimensions' -- NEVER INTO 'name', 'measure', OR ANY OTHER FIELD
- WHEN CREATING THE FINAL JSON STRUCTURE, PRIORITISE COMPACT STRUCTURE OVER VERBOSE
- ALWAYS USE DISTINCT dimensions LIST FOR EVERY OBSERVATION IN A LIST
"""

# Runs once per PDF upload, over the first N specified pages of document, and its answer is
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
state the value plainly. A wrong value here is copied onto every page of the report, so prefer
null over a plausible guess.
"""
