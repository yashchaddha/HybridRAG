"""Deterministic test fixtures: a small SQLite database and three PDFs.

Domain: a fictional HVAC manufacturer ("AeroFlow") selling to fleet customers.
The data is engineered so that realistic questions genuinely require BOTH
sources, e.g.:

  "Acme Corp reported compressor failures on their AeroFlow X200 units.
   How many X200 units have they purchased since 2025, and are these
   failures covered under warranty?"

  -> unit counts / order dates  = SQLite
  -> coverage terms / bulletins = PDFs
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

from .config import settings

# ---------------------------------------------------------------------------
# 1. SQLite seed (structured source)
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    segment TEXT NOT NULL,
    region TEXT NOT NULL
);
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL,
    warranty_years INTEGER NOT NULL
);
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date TEXT NOT NULL,          -- ISO date
    status TEXT NOT NULL
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL
);
CREATE TABLE support_tickets (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    opened_date TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL
);
"""

CUSTOMERS = [
    (1, "Acme Corp", "Industrial", "North"),
    (2, "Borealis Labs", "Pharma", "North"),
    (3, "Cascade Foods", "Food & Beverage", "West"),
    (4, "Delta Logistics", "Warehousing", "South"),
    (5, "Evergreen Clinics", "Healthcare", "East"),
]

PRODUCTS = [
    (1, "AF-X200", "AeroFlow X200", "Industrial Dehumidifier", 1899.0, 2),
    (2, "AF-X100", "AeroFlow X100", "Industrial Dehumidifier", 1249.0, 2),
    (3, "HM-P50", "HydroMax P50", "Condensate Pump", 749.0, 1),
    (4, "CC-T8", "ClimaCore T8", "Smart Thermostat", 329.0, 2),
]

# (id, customer_id, order_date, status)
ORDERS = [
    # --- Acme Corp -----------------------------------------------------
    (1001, 1, "2024-07-19", "delivered"),   # pre-2025 X200s (excluded by "since 2025")
    (1002, 1, "2025-01-15", "delivered"),
    (1003, 1, "2025-04-02", "delivered"),
    (1004, 1, "2025-04-02", "delivered"),   # same-day thermostat order
    (1005, 1, "2026-02-10", "delivered"),
    # --- Borealis Labs --------------------------------------------------
    (1006, 2, "2025-03-11", "delivered"),
    (1007, 2, "2025-09-30", "delivered"),
    # --- Cascade Foods ---------------------------------------------------
    (1008, 3, "2024-11-05", "delivered"),
    (1009, 3, "2025-06-21", "delivered"),
    # --- Delta Logistics --------------------------------------------------
    (1010, 4, "2025-02-14", "delivered"),
    (1011, 4, "2026-01-08", "shipped"),
    # --- Evergreen Clinics --------------------------------------------------
    (1012, 5, "2025-08-17", "delivered"),
]

# (id, order_id, product_id, quantity, unit_price)
ORDER_ITEMS = [
    (1, 1001, 1, 5, 1899.0),    # Acme  X200 x5  (2024)
    (2, 1002, 1, 8, 1899.0),    # Acme  X200 x8  (2025)
    (3, 1003, 1, 6, 1899.0),    # Acme  X200 x6  (2025)
    (4, 1004, 4, 10, 329.0),    # Acme  T8  x10  (2025)
    (5, 1005, 1, 4, 1899.0),    # Acme  X200 x4  (2026)   -> since-2025 X200 total = 18
    (6, 1006, 2, 12, 1249.0),   # Borealis X100
    (7, 1007, 1, 3, 1899.0),    # Borealis X200 x3 (2025)
    (8, 1008, 3, 20, 749.0),    # Cascade pumps (2024)
    (9, 1009, 1, 2, 1899.0),    # Cascade X200 x2 (2025)
    (10, 1010, 4, 30, 329.0),   # Delta thermostats
    (11, 1011, 2, 6, 1249.0),   # Delta X100 (2026)
    (12, 1012, 3, 8, 749.0),    # Evergreen pumps
]

# (id, customer_id, product_id, opened_date, issue_type, status, summary)
TICKETS = [
    (1, 1, 1, "2025-06-03", "compressor failure", "open",
     "Two X200 units shut down intermittently; compressor stops under load."),
    (2, 1, 1, "2025-09-22", "compressor failure", "open",
     "Third X200 unit showing identical compressor shutdown pattern."),
    (3, 1, 1, "2026-01-12", "compressor failure", "investigating",
     "Recurring compressor relay clicks then trips; unit serial X2-25A-0142."),
    (4, 1, 4, "2025-05-08", "display fault", "resolved",
     "ClimaCore T8 panel flickers after firmware 2.3 update."),
    (5, 3, 1, "2025-08-19", "condensate leak", "resolved",
     "Drain pan overflow on one X200 unit; cleared blocked line."),
    (6, 2, 2, "2025-10-02", "noise complaint", "resolved",
     "X100 fan bearing noise; bearing replaced under warranty."),
]


def build_database(db_path: Path | None = None) -> Path:
    db_path = db_path or settings.db_path
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany("INSERT INTO customers VALUES (?,?,?,?)", CUSTOMERS)
        conn.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", PRODUCTS)
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", ORDER_ITEMS)
        conn.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?)", TICKETS)
        conn.commit()
    finally:
        conn.close()
    return db_path


# ---------------------------------------------------------------------------
# 2. PDF corpus (unstructured source)
# ---------------------------------------------------------------------------
# Each document is a list of pages; each page is a list of (kind, text) where
# kind is "h" (heading) or "p" (paragraph). Page boundaries are explicit so
# page-level citations in the demo are exact by construction.

PDF_DOCS: dict[str, list[list[tuple[str, str]]]] = {
    "AeroFlow_X200_Specification.pdf": [
        [  # page 1
            ("h", "AeroFlow X200 Product Specification"),
            ("p", "The AeroFlow X200 is an industrial dehumidifier engineered for "
                  "continuous-duty operation in warehouses, production floors, and "
                  "cold-chain staging areas. It removes up to 200 litres of moisture "
                  "per day at 30 degrees C and 80 percent relative humidity."),
            ("h", "Key Specifications"),
            ("p", "Rated extraction: 200 L/day. Airflow: 2,000 m3/h. Power draw: "
                  "2.4 kW at 230 V. Refrigerant: R-32. Compressor: sealed rotary "
                  "compressor, model RC-9 series. Net weight: 86 kg. Noise level: "
                  "58 dB(A) at 3 m."),
        ],
        [  # page 2
            ("h", "Operating Conditions"),
            ("p", "The X200 is rated for ambient temperatures between 5 degrees C "
                  "and 40 degrees C. Operation below 5 degrees C can cause coil "
                  "icing and compressor stress and is outside the supported "
                  "envelope. Supply voltage must remain within 200-240 V; sustained "
                  "operation outside this range may damage the compressor relay."),
            ("h", "Compressor Subsystem"),
            ("p", "The sealed rotary compressor is the primary wear component of "
                  "the X200. It is protected by a thermal cut-out and a start relay "
                  "(part SR-1142). The compressor assembly is field-replaceable by "
                  "certified technicians using service kit CK-200."),
        ],
        [  # page 3
            ("h", "Maintenance Schedule"),
            ("p", "Filters should be cleaned every 500 operating hours and replaced "
                  "every 2,000 hours. The condensate line should be flushed "
                  "quarterly. An annual inspection of the compressor start relay is "
                  "recommended for units in continuous duty."),
        ],
    ],
    "AeroFlow_Warranty_Policy.pdf": [
        [  # page 1
            ("h", "AeroFlow Limited Warranty Policy"),
            ("p", "This policy applies to all AeroFlow-branded equipment sold on or "
                  "after 1 January 2024, including the AeroFlow X100 and AeroFlow "
                  "X200 dehumidifier lines."),
            ("h", "Standard Coverage"),
            ("p", "AeroFlow warrants each unit to be free from defects in materials "
                  "and workmanship for a period of two (2) years from the date of "
                  "delivery. During this period AeroFlow will repair or replace "
                  "defective parts, including labour."),
        ],
        [  # page 2
            ("h", "Compressor Coverage"),
            ("p", "The sealed compressor assembly carries extended coverage of five "
                  "(5) years from the date of delivery. Compressor coverage "
                  "includes parts and labour during the first two (2) years and "
                  "parts-only for years three (3) through five (5). Compressor "
                  "failures attributable to manufacturing defects, including start "
                  "relay defects, are covered for the full five-year term."),
            ("h", "Exclusions"),
            ("p", "This warranty does not cover damage caused by operation below 5 "
                  "degrees C ambient, supply voltage outside 200-240 V, unauthorised "
                  "modification or service, or failure to follow the published "
                  "maintenance schedule."),
        ],
        [  # page 3
            ("h", "Claims Procedure"),
            ("p", "Warranty claims must be submitted through the AeroFlow service "
                  "portal within thirty (30) days of fault detection and must "
                  "include the unit serial number and proof of purchase. Fleet "
                  "customers purchasing ten (10) or more units in a rolling twelve "
                  "month period qualify for advance replacement: a replacement unit "
                  "ships before the failed unit is returned."),
        ],
    ],
    "Field_Service_Bulletin_FSB-2025-03.pdf": [
        [  # page 1
            ("h", "Field Service Bulletin FSB-2025-03"),
            ("p", "Subject: intermittent compressor shutdown on AeroFlow X200 "
                  "units. Severity: high. Issued: 12 May 2025."),
            ("h", "Affected Units"),
            ("p", "This bulletin applies to AeroFlow X200 units manufactured "
                  "between January 2025 and March 2025, serial number prefix "
                  "X2-25A. Root cause analysis traced the fault to start relay "
                  "component lot L-1142 (part SR-1142), which can stick under "
                  "thermal load and trip the compressor."),
        ],
        [  # page 2
            ("h", "Corrective Action"),
            ("p", "Certified technicians must install Relay Retrofit Kit RK-114 "
                  "and update controller firmware to version 2.4.1 on all affected "
                  "units. The retrofit takes approximately 40 minutes per unit."),
            ("h", "Warranty Treatment"),
            ("p", "Repairs performed under this bulletin are covered at no charge "
                  "for all affected serial numbers, regardless of the standard "
                  "warranty term, when completed before 31 December 2026."),
        ],
    ],
}


def build_pdfs(pdf_dir: Path | None = None) -> list[Path]:
    pdf_dir = pdf_dir or settings.pdf_dir
    pdf_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    written: list[Path] = []
    for filename, pages in PDF_DOCS.items():
        path = pdf_dir / filename
        doc = SimpleDocTemplate(str(path), pagesize=letter,
                                leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                                topMargin=0.9 * inch, bottomMargin=0.9 * inch)
        story = []
        for i, page in enumerate(pages):
            for kind, text in page:
                style = styles["Heading1"] if kind == "h" else styles["BodyText"]
                story.append(Paragraph(text, style))
                story.append(Spacer(1, 10))
            if i < len(pages) - 1:
                story.append(PageBreak())
        doc.build(story)
        written.append(path)
    return written


if __name__ == "__main__":
    print("db ->", build_database())
    for p in build_pdfs():
        print("pdf ->", p)
