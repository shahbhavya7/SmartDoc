"""Generate a deep, realistic evaluation corpus of company PDFs.

The original test PDFs totalled ~21 chunks, at which scale ``top_k=4`` retrieves
~19% of the corpus on every query. That makes retrieval metrics statistically
meaningless and makes the hard query classes (multi-hop, comparison, exhaustive
extraction, synthesis) impossible to reproduce, let alone measure. This builds a
corpus with enough depth and structure that the right answer is genuinely hard to
find.

Deliberately planted retrieval challenges
-----------------------------------------
1. **Multi-hop chains** -- the bridging fact lives in a different document from
   the answer ("Northwind Logistics is Tier 3" in the vendor register; "Tier 3
   requires SOC 2 every 12 months" in the vendor policy).
2. **Cross-document comparisons** -- the same concept defined with different
   values in the employee vs contractor handbooks.
3. **Exhaustive lists** -- a 9-row fault-code table plus codes documented only in
   prose, so extraction must find both.
4. **Near-duplicate distractors** -- the contractor handbook mirrors
   employee-policy wording with different numbers, punishing retrieval that
   matches on phrasing rather than entity.
5. **Cross-page sections** -- sections long enough to span a page break.
6. **Running headers/footers** -- repeated on every page, so stripping is
   exercised.
7. **Line-wrap hyphenation** -- long words split across breaks.

Output is deterministic (fixed seed), so the eval gold set stays valid.

Usage:
    .venv/bin/python -m scripts.generate_corpus [--out data] [--pad 8]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

import fitz

SEED = 20260731

PAGE_W, PAGE_H = 595, 842  # A4 in points
MARGIN_X = 62
TOP_Y = 96
BOTTOM_Y = 760
BODY_SIZE = 9.5
BODY_LEAD = 13.0
HEAD_SIZE = 12.0
TITLE_SIZE = 17.0

FOOTER_TEXT = "Acme Corporation - Internal Use Only"

# Multiplier applied to every filler request. This is the knob that sets corpus
# depth: the load-bearing facts are fixed, so raising PAD raises the volume of
# plausible-but-irrelevant surrounding text that retrieval must discriminate
# against. At PAD=8 a top-k of 6 touches ~1-2% of the corpus, which is the regime
# real deployments operate in.
PAD = 8


@dataclass
class Table:
    columns: list[str]
    rows: list[list[str]]


@dataclass
class Section:
    heading: str
    paragraphs: list[str] = field(default_factory=list)
    table: Table | None = None


@dataclass
class Document:
    filename: str
    title: str
    subtitle: str
    sections: list[Section]


FILLER = [
    "This section should be read together with the remainder of this document and "
    "any superseding guidance issued by the People Operations team.",
    "Managers are responsible for applying these provisions consistently across "
    "their reporting lines and for escalating ambiguous cases rather than "
    "improvising local interpretations.",
    "Where a provision conflicts with a statutory entitlement in the employee's "
    "jurisdiction, the statutory entitlement prevails and the People Operations "
    "team must be notified so this document can be amended.",
    "Requests submitted outside the documented process may be delayed, since the "
    "reviewing team will not receive the automated notification that ordinarily "
    "accompanies a correctly filed request.",
    "Records created under this section are retained in accordance with the "
    "corporate retention schedule and are available to the employee on request.",
    "Exceptions require written approval from a department head and are reviewed "
    "annually to confirm that the original justification still applies.",
    "Nothing in this section limits the company's obligation to comply with "
    "applicable health, safety, and employment legislation.",
    "Questions about interpretation should be directed to the owning team listed "
    "in the document header rather than to an individual contributor.",
    "The company reviews this guidance at least once per calendar year and after "
    "any material change to the underlying systems or regulations.",
    "Employees who believe a provision has been applied incorrectly may raise a "
    "grievance through the standard channel without fear of retaliation.",
    "Where automation is available, the relevant workflow tool is the system of "
    "record and manual spreadsheets must not be used to track approvals.",
    "Training materials referenced in this section are maintained in the learning "
    "management system and are updated when this document changes.",
    "The owning team publishes a summary of material changes alongside each "
    "revision so readers need not compare full versions manually.",
    "Historical versions of this document remain available in the document "
    "management system for audit purposes and must not be treated as current.",
    "Local teams may issue supplementary guidance provided it does not reduce any "
    "entitlement or weaken any control described here.",
    "Approvals recorded outside the designated system are not considered valid "
    "even where the approver confirms them verbally at a later date.",
    "Where a deadline falls on a weekend or public holiday, the deadline moves to "
    "the next working day in the employee's registered location.",
    "The company may audit compliance with this section at any time and will give "
    "reasonable notice except where advance notice would defeat the audit.",
    "Data generated in the course of applying this section is processed under the "
    "employee privacy notice and is not used for unrelated purposes.",
    "Team leads should raise capacity concerns before they affect delivery rather "
    "than absorbing them silently within the team.",
    "Any template referenced in this section is authoritative only in its current "
    "published form; locally cached copies must not be relied upon.",
    "Failure to follow the documented process does not invalidate an underlying "
    "entitlement, but may delay the point at which it takes effect.",
    "The reviewing team aims to respond within five working days and will notify "
    "the requester if a longer period is required.",
    "Where this document refers to a role rather than a named individual, the "
    "responsibility sits with whoever currently holds that role.",
    "Automated reminders are issued before each deadline; the absence of a "
    "reminder does not extend the deadline.",
    "Aggregate reporting on this section is provided to the leadership team "
    "quarterly and excludes individually identifying detail.",
    "Employees on long-term leave are excluded from routine reminders and are "
    "re-enrolled on their documented return date.",
    "The company retains discretion to apply a more generous outcome than this "
    "document requires where circumstances plainly warrant it.",
    "Terminology used here follows the definitions in the corporate glossary "
    "unless this document states otherwise explicitly.",
    "Requests are processed in the order received, except where a documented "
    "urgency criterion applies and has been approved.",
    "The relevant workflow retains a full audit trail, including the identity of "
    "each approver and the timestamp of each decision.",
    "Where a third party is involved, the company remains accountable for the "
    "outcome regardless of which party performs the work.",
    "Nothing in this section creates a contractual entitlement beyond those set "
    "out in the individual's written terms.",
    "Managers should document the rationale for any discretionary decision so it "
    "can be explained consistently if questioned later.",
    "The company will not penalise an employee for raising a concern in good "
    "faith, even where the concern is ultimately not substantiated.",
    "Periodic sampling is used to confirm that recorded outcomes match the "
    "supporting evidence retained alongside them.",
]


def filler(rng: random.Random, count: int) -> list[str]:
    """Return ``count * PAD`` multi-sentence filler paragraphs.

    Real policy documents are mostly connective tissue around a few load-bearing
    facts. Composing each paragraph from 2-4 sampled sentences gives
    combinatorial variety, so retrieval must discriminate on meaning rather than
    latching onto the only text on the page.
    """
    paragraphs: list[str] = []
    for _ in range(count * PAD):
        sentences = rng.sample(FILLER, rng.randint(2, 4))
        paragraphs.append(" ".join(sentences))
    return paragraphs


def build_documents() -> list[Document]:
    """Build the corpus definition, with all planted facts in place."""
    rng = random.Random(SEED)
    docs: list[Document] = []

    docs.append(
        Document(
            "employee_handbook.pdf",
            "Employee Handbook",
            "Terms, Entitlements, and Working Arrangements",
            [
                Section(
                    "1. Scope and Application",
                    [
                        "This handbook applies to all directly employed staff of Acme "
                        "Corporation, including full-time and part-time employees on "
                        "permanent and fixed-term contracts. It does not apply to "
                        "independent contractors or agency workers, whose terms are set "
                        "out in the Contractor Handbook.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "2. Compensation Bands",
                    [
                        "Employees are assigned to one of three compensation bands: "
                        "Standard, Senior, or Executive. Band assignment is determined at "
                        "hire and reviewed during the annual compensation cycle. Band "
                        "determines annual leave entitlement, notice period, and "
                        "eligibility for the deferred bonus scheme.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "3. Annual Leave Entitlement",
                    [
                        "Employees in the Standard band accrue twenty (20) days of paid "
                        "annual leave per calendar year. Employees in the Senior band "
                        "accrue twenty-four (24) days. Employees in the Executive band "
                        "accrue twenty-eight (28) days of paid annual leave per calendar "
                        "year.",
                        "Annual leave accrues monthly in arrears and is credited on the "
                        "last working day of each month. Part-time employees accrue leave "
                        "on a pro-rata basis according to their contracted hours.",
                        "A maximum of five (5) unused annual leave days may be carried "
                        "into the following calendar year. Carried days expire on 31 "
                        "March and cannot be exchanged for payment except on termination.",
                        *filler(rng, 6),
                    ],
                ),
                Section(
                    "4. Sick Leave",
                    [
                        "All employees are entitled to ten (10) paid sick days per "
                        "calendar year regardless of compensation band. A medical "
                        "certificate is required for any absence exceeding three (3) "
                        "consecutive working days.",
                        "Sick leave does not accrue and unused days do not carry over.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "5. Parental Leave",
                    [
                        "Primary caregivers are entitled to sixteen (16) weeks of paid "
                        "parental leave. Secondary caregivers are entitled to four (4) "
                        "weeks of paid parental leave. Both entitlements require six (6) "
                        "months of continuous service at the expected date of leave.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "6. Notice Periods",
                    [
                        "Standard band employees must give four (4) weeks written notice. "
                        "Senior band employees must give eight (8) weeks. Executive band "
                        "employees must give twelve (12) weeks written notice.",
                        *filler(rng, 4),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "contractor_handbook.pdf",
            "Contractor Handbook",
            "Engagement Terms for Independent Contractors",
            [
                Section(
                    "1. Scope and Application",
                    [
                        "This handbook applies to independent contractors and agency "
                        "workers engaged by Acme Corporation. Contractors are not "
                        "employees and the Employee Handbook does not apply to them.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "2. Leave and Time Off",
                    [
                        "Contractors do not accrue paid annual leave. Contractors may take "
                        "unpaid time off with fourteen (14) days written notice to the "
                        "engaging manager. There is no carry-over provision because no "
                        "entitlement accrues.",
                        "Contractors are not entitled to paid sick days. A contractor who "
                        "is unable to work must notify the engaging manager on the first "
                        "day of absence.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "3. Remote Working",
                    [
                        "Contractors may work remotely for up to five (5) days per week "
                        "where the statement of work does not require on-site presence. "
                        "This differs from the employee arrangement described in the "
                        "Remote Work Policy.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "4. Notice and Termination",
                    [
                        "Either party may terminate a contractor engagement with two (2) "
                        "weeks written notice, unless the statement of work specifies a "
                        "longer period.",
                        *filler(rng, 4),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "remote_work_policy.pdf",
            "Remote Work Policy",
            "Hybrid and Fully Remote Working Arrangements",
            [
                Section(
                    "1. Eligibility",
                    [
                        "Employees whose role does not require regular on-site presence "
                        "may request a hybrid working arrangement. Eligibility is reviewed "
                        "annually and may change if role responsibilities change.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "2. Permitted Remote Days",
                    [
                        "Employees may work remotely for up to three (3) days per week. "
                        "Fully remote arrangements are available only where the role has "
                        "been designated remote-eligible by the department head and the "
                        "People Operations team.",
                        "Core collaboration hours are 10:00 to 15:00 in the employee's "
                        "registered time zone, during which employees are expected to be "
                        "reachable regardless of location.",
                        *filler(rng, 6),
                    ],
                ),
                Section(
                    "3. Equipment and Expenses",
                    [
                        "The company provides a laptop and one external monitor. A "
                        "one-time home-office allowance of three hundred (300) currency "
                        "units is available in the first year of a hybrid arrangement.",
                        *filler(rng, 4),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "data_classification_standard.pdf",
            "Data Classification Standard",
            "Classification Tiers and Assignment Rules",
            [
                Section(
                    "1. Purpose",
                    [
                        "This standard defines the classification tiers applied to all "
                        "company information assets and assigns specific record types to "
                        "those tiers. Handling requirements for each tier are defined in "
                        "the Information Security Handbook.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "2. Classification Tiers",
                    [
                        "Four tiers are defined: Public, Internal, Confidential, and "
                        "Restricted. Restricted is the most sensitive tier and carries the "
                        "strictest handling requirements.",
                        *filler(rng, 4),
                    ],
                    table=Table(
                        ["Record type", "Classification"],
                        [
                            ["Published marketing material", "Public"],
                            ["Internal process documentation", "Internal"],
                            ["Commercial contract terms", "Confidential"],
                            ["Payroll records", "Restricted"],
                            ["Employee medical records", "Restricted"],
                            ["Customer payment card data", "Restricted"],
                            ["Aggregated usage analytics", "Internal"],
                        ],
                    ),
                ),
                Section(
                    "3. Reclassification",
                    [
                        "A record's classification may be lowered only with written "
                        "approval from the data owner and the Information Security team.",
                        *filler(rng, 4),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "information_security_handbook.pdf",
            "Information Security Handbook",
            "Controls, Handling Requirements, and Incident Response",
            [
                Section(
                    "1. Scope",
                    [
                        "This handbook defines the technical and procedural controls that "
                        "apply to information assets according to the tier assigned in the "
                        "Data Classification Standard.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "2. Handling Requirements by Tier",
                    [
                        "Restricted data must be encrypted at rest using AES-256 and in "
                        "transit using TLS 1.3 or later. Confidential data must be "
                        "encrypted at rest using AES-128 or stronger. Internal data must "
                        "not be stored on personal devices. Public data has no encryption "
                        "requirement.",
                        "Access to Restricted data additionally requires hardware-token "
                        "multi-factor authentication and is logged for a minimum of "
                        "twenty-four (24) months.",
                        *filler(rng, 6),
                    ],
                ),
                Section(
                    "3. Password and Authentication Policy",
                    [
                        "Passwords must be at least twelve (12) characters and are not "
                        "subject to mandatory rotation on a fixed schedule. Accounts lock "
                        "after five (5) consecutive failed authentication attempts and "
                        "unlock automatically after thirty (30) minutes.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "4. Backup and Retention",
                    [
                        "Backup archives are retained for ninety (90) days before secure "
                        "destruction. Restoration from backup requires approval from the "
                        "Information Security team and is tested quarterly.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "5. Incident Response",
                    [
                        "Suspected incidents must be reported to the Information Security "
                        "team within one (1) hour of discovery. A Priority 1 incident is "
                        "one that affects Restricted data or halts a customer-facing "
                        "service.",
                        *filler(rng, 5),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "vendor_management_policy.pdf",
            "Vendor Management Policy",
            "Tiering, Due Diligence, and Ongoing Assurance",
            [
                Section(
                    "1. Vendor Tiers",
                    [
                        "Vendors are assigned to Tier 1, Tier 2, or Tier 3 based on the "
                        "sensitivity of the data they process and the criticality of the "
                        "service they provide. Tier 3 denotes the highest risk.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "2. Assurance Requirements by Tier",
                    [
                        "Tier 1 vendors must complete a self-assessment questionnaire "
                        "every twenty-four (24) months. Tier 2 vendors must complete an "
                        "independent penetration test every eighteen (18) months. Tier 3 "
                        "vendors must complete a SOC 2 Type II review every twelve (12) "
                        "months and provide the report to the Information Security team.",
                        "A vendor that fails to provide current assurance evidence is "
                        "suspended from new work until the evidence is supplied.",
                        *filler(rng, 6),
                    ],
                ),
                Section(
                    "3. Onboarding Due Diligence",
                    [
                        "New vendors must complete due diligence before any contract is "
                        "signed. Due diligence for Tier 3 vendors additionally requires a "
                        "site visit or an equivalent virtual assessment.",
                        *filler(rng, 5),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "vendor_register.pdf",
            "Approved Vendor Register",
            "Current Tier Assignments",
            [
                Section(
                    "1. Register Maintenance",
                    [
                        "This register records the current tier assignment for every "
                        "approved vendor. Tier assignments are reviewed when a contract is "
                        "renewed or when the scope of processed data changes.",
                        *filler(rng, 4),
                    ],
                    table=Table(
                        ["Vendor", "Service", "Tier"],
                        [
                            ["Northwind Logistics", "Freight and fulfilment", "Tier 3"],
                            ["Bluepeak Analytics", "Usage reporting", "Tier 2"],
                            ["Carrolton Print", "Marketing collateral", "Tier 1"],
                            ["Halden Payroll Services", "Payroll processing", "Tier 3"],
                            ["Orimoto Cloud", "Application hosting", "Tier 3"],
                            ["Vestry Facilities", "Office cleaning", "Tier 1"],
                            ["Ledbury Legal", "Contract review", "Tier 2"],
                        ],
                    ),
                ),
                Section(
                    "2. Suspended Vendors",
                    [
                        "No vendors are currently suspended. A suspended vendor may not be "
                        "issued new work orders until reinstated by the Vendor Management "
                        "team.",
                        *filler(rng, 3),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "widgetx_operations_manual.pdf",
            "WidgetX Technical Operations Manual",
            "Installation, Calibration, Fault Codes, and Maintenance",
            [
                Section(
                    "1. Overview",
                    [
                        "This manual covers industrial installation of the WidgetX unit, "
                        "including electrical requirements, calibration intervals, and "
                        "diagnostic fault codes.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "2. Electrical Requirements",
                    [
                        "Connect the unit to a grounded twenty (20) amp circuit using the "
                        "supplied power cable. Do not use an extension cord rated below "
                        "twenty amps, as undervoltage at startup can trigger a false "
                        "thermal fault.",
                        "The unit has a rated operating weight limit of forty-two (42) "
                        "kilograms and must be mounted on a level surface.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "3. Diagnostic Fault Codes",
                    [
                        "The unit reports faults on the front panel display. The following "
                        "table lists the codes reported at startup and during operation.",
                    ],
                    table=Table(
                        ["Code", "Meaning", "Required action"],
                        [
                            ["E-01", "Grounding fault", "Electrician must verify circuit"],
                            ["E-02", "Thermal overload", "Allow 20 minutes to cool"],
                            ["E-03", "Pressure sensor drift", "Recalibrate per Section 4"],
                            ["E-04", "Firmware checksum mismatch", "Reflash firmware"],
                            ["E-05", "Network link lost", "Check switch port and cable"],
                            ["E-06", "Calibration interval exceeded", "Perform calibration"],
                            ["E-07", "Door interlock open", "Close and latch access door"],
                            ["E-08", "Supply voltage out of range", "Verify 20 amp circuit"],
                            ["E-09", "Internal fan failure", "Replace fan assembly"],
                        ],
                    ),
                ),
                Section(
                    "4. Additional Fault Conditions",
                    [
                        "Two further conditions are reported without a numbered code. A "
                        "flashing amber indicator denotes a pending firmware update that "
                        "has not yet been applied. A steady red indicator with no display "
                        "text denotes a failed power supply module, which must be replaced "
                        "by a qualified technician.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "5. Calibration",
                    [
                        "Calibrate the unit every one hundred and eighty (180) days of "
                        "operation, or after any transport of the unit. Calibration drift "
                        "is the most common root cause of intermittent pressure readings.",
                        *filler(rng, 5),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "expense_reimbursement_sop.pdf",
            "Expense Reimbursement SOP",
            "Submission, Approval, and Payment",
            [
                Section(
                    "1. Scope",
                    [
                        "This procedure applies to all employees submitting business "
                        "expenses for reimbursement. Contractors invoice expenses through "
                        "the accounts payable process instead.",
                        *filler(rng, 4),
                    ],
                ),
                Section(
                    "2. Submission Procedure",
                    [
                        "Step 1: Collect itemised receipts for every expense. A card "
                        "statement alone is not an acceptable receipt.",
                        "Step 2: Enter each expense in the expense portal within thirty "
                        "(30) days of the transaction date. Expenses submitted after "
                        "sixty (60) days are rejected automatically.",
                        "Step 3: Attach a scanned or photographed copy of each receipt to "
                        "the corresponding line item.",
                        "Step 4: Select the correct cost centre. Incorrect cost centres "
                        "are the most common cause of approval delay.",
                        "Step 5: Submit the report for manager approval. The manager has "
                        "five (5) working days to approve or reject.",
                        "Step 6: Approved reports are paid in the next payment run, which "
                        "executes every second Friday.",
                        *filler(rng, 6),
                    ],
                ),
                Section(
                    "3. Approved Categories",
                    [
                        "Reimbursable categories include airfare and rail in economy class, "
                        "ground transportation, accommodation up to two hundred (200) "
                        "currency units per night, and client meals with a documented "
                        "business purpose.",
                        "Alcohol is reimbursable only as part of a client meal and must not "
                        "exceed thirty (30) percent of the total bill.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "4. Non-Reimbursable Items",
                    [
                        "Personal entertainment, traffic fines, airline seat upgrades, and "
                        "in-room minibar charges are never reimbursable.",
                        *filler(rng, 4),
                    ],
                ),
            ],
        )
    )

    docs.append(
        Document(
            "onboarding_guide.pdf",
            "New Hire Onboarding Guide",
            "First Thirty Days",
            [
                Section(
                    "1. Week One",
                    [
                        "On the first day, collect your laptop and access badge from the "
                        "facilities desk. Enrol in hardware-token multi-factor "
                        "authentication before accessing any internal system.",
                        "Complete the Information Security Awareness training within the "
                        "first five (5) business days. Complete the Code of Conduct "
                        "training within the first ten (10) business days.",
                        *filler(rng, 5),
                    ],
                ),
                Section(
                    "2. Required Trainings",
                    [
                        "All new hires must complete the following trainings. Trainings are "
                        "assigned automatically in the learning management system on the "
                        "first day of employment.",
                    ],
                    table=Table(
                        ["Training", "Deadline"],
                        [
                            ["Information Security Awareness", "5 business days"],
                            ["Code of Conduct", "10 business days"],
                            ["Data Classification Basics", "15 business days"],
                            ["Anti-Bribery and Corruption", "20 business days"],
                            ["Health and Safety Induction", "10 business days"],
                            ["Accessibility Fundamentals", "30 business days"],
                            ["Incident Reporting", "15 business days"],
                        ],
                    ),
                ),
                Section(
                    "3. Thirty Day Review",
                    [
                        "Your manager will hold a thirty-day review to confirm that "
                        "onboarding objectives have been met and that all mandatory "
                        "trainings are complete.",
                        *filler(rng, 4),
                    ],
                ),
            ],
        )
    )

    return docs


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------


def _clip(text: str, max_width: float) -> str:
    """Trim ``text`` so it fits ``max_width`` at table font size."""
    if fitz.get_text_length(text, fontsize=8.5) <= max_width:
        return text
    out = text
    while out and fitz.get_text_length(out + "...", fontsize=8.5) > max_width:
        out = out[:-1]
    return out + "..."


def _hyphenate(text: str) -> list[str]:
    """Split long words with a soft hyphen to exercise de-hyphenation."""
    out: list[str] = []
    for word in text.split():
        if len(word) >= 13 and "-" not in word:
            cut = len(word) // 2
            out.append(word[:cut] + "-")
            out.append(word[cut:])
        else:
            out.append(word)
    return out


class PdfBuilder:
    """Paginating PDF writer with running headers and footers."""

    def __init__(self, doc: Document) -> None:
        self.spec = doc
        self.pdf = fitz.open()
        self.page: fitz.Page | None = None
        self.y = 0.0
        self.page_no = 0
        self._new_page(first=True)

    def _new_page(self, first: bool = False) -> None:
        self.page = self.pdf.new_page(width=PAGE_W, height=PAGE_H)
        self.page_no += 1
        self.page.insert_text(
            (MARGIN_X, 52), self.spec.title, fontsize=8, color=(0.42, 0.42, 0.42)
        )
        self.page.draw_line(
            fitz.Point(MARGIN_X, 60), fitz.Point(PAGE_W - MARGIN_X, 60), width=0.4
        )
        self.page.insert_text(
            (MARGIN_X, BOTTOM_Y + 30), FOOTER_TEXT, fontsize=7.5, color=(0.5, 0.5, 0.5)
        )
        self.page.insert_text((PAGE_W / 2 - 6, BOTTOM_Y + 44), str(self.page_no), fontsize=8)
        self.y = TOP_Y if first else 84.0

    def _ensure(self, needed: float) -> None:
        if self.y + needed > BOTTOM_Y:
            self._new_page()

    def title_block(self) -> None:
        self.page.insert_text((MARGIN_X, self.y), self.spec.title, fontsize=TITLE_SIZE)
        self.y += 24
        self.page.insert_text(
            (MARGIN_X, self.y), self.spec.subtitle, fontsize=10, color=(0.3, 0.3, 0.3)
        )
        self.y += 26

    def heading(self, text: str) -> None:
        self._ensure(34)
        self.y += 8
        self.page.insert_text((MARGIN_X, self.y), text, fontsize=HEAD_SIZE)
        self.y += 17

    def _emit_line(self, text: str) -> None:
        self._ensure(BODY_LEAD)
        self.page.insert_text((MARGIN_X, self.y), text, fontsize=BODY_SIZE)
        self.y += BODY_LEAD

    def paragraph(self, text: str) -> None:
        """Insert wrapped body text."""
        width = PAGE_W - 2 * MARGIN_X
        line: list[str] = []
        for word in _hyphenate(text):
            trial = " ".join(line + [word])
            if fitz.get_text_length(trial, fontsize=BODY_SIZE) > width:
                self._emit_line(" ".join(line))
                line = [word]
            else:
                line.append(word)
        if line:
            self._emit_line(" ".join(line))
        self.y += 6

    def table(self, table: Table) -> None:
        """Render a simple bordered table."""
        col_w = (PAGE_W - 2 * MARGIN_X) / len(table.columns)
        row_h = 17.0
        self._ensure(row_h * 2)
        self.y += 6

        def row(cells: list[str], bold: bool) -> None:
            self._ensure(row_h + 2)
            top = self.y - 11
            for i, cell in enumerate(cells):
                x = MARGIN_X + i * col_w
                self.page.draw_rect(fitz.Rect(x, top, x + col_w, top + row_h), width=0.4)
                self.page.insert_text(
                    (x + 4, self.y),
                    _clip(cell, col_w - 8),
                    fontsize=9 if bold else 8.5,
                )
            self.y += row_h

        row(table.columns, bold=True)
        for r in table.rows:
            row(r, bold=False)
        self.y += 10

    def save(self, out_dir: Path) -> tuple[Path, int]:
        for i, section in enumerate(self.spec.sections):
            if i == 0:
                self.title_block()
            self.heading(section.heading)
            for para in section.paragraphs:
                self.paragraph(para)
            if section.table:
                self.table(section.table)
        path = out_dir / self.spec.filename
        self.pdf.save(path)
        pages = self.pdf.page_count
        self.pdf.close()
        return path, pages


def main() -> None:
    global PAD

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", help="Output directory.")
    parser.add_argument(
        "--pad",
        type=int,
        default=PAD,
        help="Filler multiplier controlling corpus depth (default: %(default)s).",
    )
    args = parser.parse_args()
    PAD = args.pad

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = build_documents()
    total_pages = 0
    for spec in specs:
        path, pages = PdfBuilder(spec).save(out_dir)
        total_pages += pages
        print(f"  {path.name:44} {pages:3} pages")

    print(f"\n{len(specs)} documents, {total_pages} pages total -> {out_dir}")


if __name__ == "__main__":
    main()
