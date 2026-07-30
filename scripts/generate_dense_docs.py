"""One-off generator: writes two long, dense company PDFs into data/.

The 5 original test PDFs are ~2 pages / ~100-170 words per page, so at
CHUNK_SIZE 500/800/1200 tokens the splitter never splits within a page and
every config yields identical chunks -- the chunk-size eval would measure
nothing. This script adds two denser, multi-page documents (several hundred
tokens per page, multiple sections) with a couple of facts deliberately
placed at the *tail* of a long section, far from the section heading, so
that a small chunk size can split the heading/topic words away from the
answer sentence while a larger chunk size keeps them together. That is the
"near a section boundary" condition the chunk-size eval needs to be
defensible.

Run once:
    .venv/bin/python -m scripts.generate_dense_docs
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PAGE_RECT = fitz.Rect(56, 56, 540, 760)  # generous margins, letter-ish page
FONT_SIZE = 10.5
FONT = "helv"


def _write_pdf(filename: str, pages: list[str]) -> None:
    """Write ``pages`` (one string per page) into a new PDF at data/filename."""
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=612, height=792)  # US Letter
        page.insert_textbox(
            PAGE_RECT,
            text,
            fontsize=FONT_SIZE,
            fontname=FONT,
            align=fitz.TEXT_ALIGN_LEFT,
        )
    out_path = DATA_DIR / filename
    doc.save(out_path)
    doc.close()
    print(f"Wrote {out_path} ({len(pages)} pages)")


# ---------------------------------------------------------------------------
# Document A: Data Security Policy Handbook
# ---------------------------------------------------------------------------

SECURITY_PAGES: list[str] = [
    # Page 1
    "Data Security Policy Handbook\n"
    "Acme Corporation - Information Security Program\n\n"
    "1. Purpose and Scope\n"
    "This handbook establishes the mandatory data security controls that apply to "
    "all Acme Corporation employees, contractors, and third-party vendors who "
    "access, process, store, or transmit company information on any device or "
    "network connected to Acme systems. It covers access control, data "
    "classification, encryption, backup and retention, incident response, and "
    "vendor risk management. Compliance with this handbook is a condition of "
    "employment and of any vendor services agreement, and violations are subject "
    "to disciplinary action up to and including termination of employment or "
    "contract.\n\n"
    "2. Data Classification\n"
    "All company information must be classified into one of four tiers: Public, "
    "Internal, Confidential, or Restricted. Public data may be shared outside the "
    "company without restriction. Internal data is intended for employees only "
    "and should not be shared externally without approval. Confidential data "
    "includes customer records, financial statements, and unreleased product "
    "plans, and requires manager approval before external sharing. Restricted "
    "data includes authentication credentials, encryption keys, and personally "
    "identifiable information subject to regulatory protection, and may only be "
    "accessed by employees with an explicit, documented business need and only "
    "over an encrypted channel.",
    # Page 2
    "3. Access Control\n"
    "Access to company systems is granted on a least-privilege basis: employees "
    "receive only the permissions required for their current role, and access is "
    "reviewed quarterly by system owners. All accounts must use multi-factor "
    "authentication for any system holding Confidential or Restricted data. "
    "Shared accounts are prohibited except for designated service accounts that "
    "are inventoried and rotated on a documented schedule. When an employee "
    "changes roles or leaves the company, their access must be revoked or "
    "adjusted within one business day by the IT Service Desk, and this "
    "deprovisioning is logged for the annual access audit.\n\n"
    "3.1 Password and Lockout Policy\n"
    "Passwords must be at least twelve characters and are not required to be "
    "rotated on a fixed schedule, consistent with current guidance that "
    "favors length over forced rotation. Accounts are automatically locked "
    "after five consecutive failed login attempts within a fifteen-minute "
    "window, and the lockout is released automatically after thirty minutes or "
    "sooner by an IT Service Desk agent after identity verification.",
    # Page 3 -- long "Backup and Data Retention" section; the retention
    # figure sits at the very tail of a long paragraph, far from the
    # section heading and topic vocabulary that a question would use.
    "4. Backup and Data Retention\n"
    "4.1 Backup Architecture\n"
    "Acme operates a tiered backup architecture spanning production databases, "
    "file servers, and virtual machine images across two geographically separate "
    "data centers. Primary backups are taken as nightly incremental snapshots "
    "orchestrated by the central backup scheduler, capturing only the blocks that "
    "changed since the previous snapshot to minimize storage growth and network "
    "load on the production storage array. In addition to the nightly "
    "incrementals, a full baseline backup of every production volume is taken "
    "once per week during the Saturday maintenance window, when application "
    "traffic is lowest and the backup job can saturate the available network "
    "bandwidth without degrading customer-facing performance.\n"
    "Every backup, whether incremental or full, is encrypted at rest using "
    "AES-256 before it ever leaves the primary data center, and the encryption "
    "keys used for backup archives are managed separately from the keys used for "
    "live production data, stored in a dedicated hardware security module that "
    "only the backup service account and two named security engineers can "
    "access. Backups are then replicated asynchronously to an off-site cold "
    "storage facility operated by a certified third-party provider under a "
    "signed data processing agreement, and the replication job verifies a "
    "checksum of every archive after transfer to guard against silent "
    "corruption in transit. Backup integrity is additionally verified on a "
    "monthly basis through a full restoration drill performed by the "
    "infrastructure team, in which a randomly selected production volume is "
    "restored into an isolated recovery environment and validated against a "
    "checksum manifest, with results logged in the disaster-recovery runbook "
    "for audit purposes. Any restoration drill that fails validation triggers an "
    "immediate incident ticket and a follow-up backup within twenty-four hours.\n"
    "The backup scheduler itself runs on a dedicated, hardened host that is "
    "patched on the same monthly cycle as the rest of the production fleet, and "
    "access to the scheduler's administrative console is restricted to the "
    "infrastructure team through a bastion host with session recording enabled, "
    "so that every configuration change to the backup jobs is attributable to a "
    "named engineer and reviewable after the fact. Backup job definitions, "
    "including source volume, schedule, retention target, and destination "
    "storage tier, are stored as version-controlled configuration rather than "
    "edited directly in the scheduler's user interface, so that any change to "
    "how a given system is backed up goes through the same peer-review process "
    "as a production code change, and can be rolled back if a misconfiguration "
    "is discovered. Capacity on the cold storage tier is monitored continuously, "
    "and the infrastructure team receives an automated alert when utilization "
    "crosses eighty percent of the provisioned quota, well before the tier is "
    "at risk of rejecting new archives, giving enough lead time to either "
    "expand the quota or investigate unexpectedly rapid growth in backup "
    "volume, which is itself sometimes an early indicator of a runaway job or "
    "an unexpected spike in production data.\n"
    "After all of this scheduling, encryption, replication, and verification "
    "activity, the retention period for full system backups held in cold "
    "storage is 90 days, after which backup archives are automatically purged "
    "by the cold storage provider's lifecycle policy and cannot be recovered.",
    # Page 4
    "5. Encryption in Transit and at Rest\n"
    "All Confidential and Restricted data must be encrypted using TLS 1.2 or "
    "higher whenever it is transmitted across any network, including internal "
    "network segments, and encrypted at rest using AES-256 on any storage volume, "
    "database, or removable media. Laptops issued to employees are encrypted at "
    "the disk level by default through the corporate device management platform, "
    "and encryption cannot be disabled by the end user. Any exception to this "
    "policy, for example for a legacy system that cannot support current TLS "
    "versions, must be documented, time-boxed, and approved in writing by the "
    "Chief Information Security Officer before the exception takes effect.\n\n"
    "5.1 Vendor and Third-Party Risk\n"
    "Any vendor that will store, process, or transmit Confidential or Restricted "
    "data on Acme's behalf must complete a security questionnaire and pass a "
    "risk review before a contract is signed, and high-risk vendors are "
    "reassessed annually. Vendor contracts must include a data processing "
    "addendum specifying encryption, breach-notification, and data-deletion "
    "obligations consistent with this handbook.",
    # Page 5 -- short, clean fact; a control case where chunk size should
    # not matter since the whole section fits comfortably in a single
    # chunk at every configured size.
    "6. Security Incident Response\n"
    "6.1 Reporting\n"
    "All security incidents, including suspected phishing, malware infections, "
    "lost or stolen devices, and unauthorized access, must be reported to the "
    "Security Operations Center within 1 hour of detection. Reports may be "
    "submitted through the internal incident portal or by calling the 24/7 "
    "security hotline. Delayed reporting is itself treated as a policy "
    "violation, since containment effectiveness drops sharply once several "
    "hours have passed.\n\n"
    "6.2 Response and Containment\n"
    "Upon receiving a report, the on-call security engineer triages the "
    "incident within fifteen minutes and assigns a severity level. Severity 1 "
    "incidents (active data exfiltration or ransomware) trigger immediate "
    "network isolation of affected systems and a page to the full incident "
    "response team.",
    # Page 6
    "7. Employee Security Training\n"
    "All employees complete mandatory security awareness training within their "
    "first week of employment and an annual refresher thereafter. Training "
    "covers phishing recognition, password hygiene, data classification, and "
    "the incident reporting process described in Section 6. Completion is "
    "tracked in the learning management system, and managers receive an "
    "escalation notice for any direct report who has not completed the annual "
    "refresher within thirty days of its due date.\n\n"
    "8. Policy Review\n"
    "This handbook is reviewed annually by the Information Security team and "
    "approved by the Chief Information Security Officer. Material changes are "
    "communicated to all employees by email and take effect thirty days after "
    "publication unless an earlier effective date is required by law or by an "
    "active security incident.",
]


# ---------------------------------------------------------------------------
# Document B: WidgetX Technical Operations Manual (extends the existing
# short product_manual_widgetx.pdf with deeper technical/procedural detail)
# ---------------------------------------------------------------------------

TECHOPS_PAGES: list[str] = [
    # Page 1
    "WidgetX Technical Operations Manual\n"
    "Advanced Installation, Calibration, and Maintenance Procedures\n\n"
    "1. Overview\n"
    "This manual supplements the WidgetX Quick Start Guide with the detailed "
    "procedures required for industrial installation sites, including site "
    "preparation, sensor calibration, firmware maintenance, and troubleshooting "
    "of common fault codes. It is intended for certified field technicians and "
    "facility engineers responsible for keeping WidgetX units within their rated "
    "operating tolerances over a multi-year service life. Technicians must "
    "complete the WidgetX certification course before performing any procedure "
    "described in Sections 3 through 6 of this manual.\n\n"
    "2. Site Preparation\n"
    "Before installation, confirm that the mounting surface can support the unit "
    "weight of 42 kilograms plus a safety margin, that ambient temperature at "
    "the install location stays within 0 to 45 degrees Celsius year-round, and "
    "that a dedicated 20-amp circuit is available within 3 meters of the "
    "planned mounting point. Sites with ambient dust or particulate levels above "
    "the rated enclosure ingress protection rating require the optional sealed "
    "enclosure kit, ordered separately.",
    # Page 2
    "3. Electrical and Network Connections\n"
    "Connect the unit to a grounded 20-amp circuit using the supplied power "
    "cable; do not use an extension cord rated below 20 amps, as undervoltage "
    "at startup can trigger a false thermal fault. Network connectivity is "
    "provided via the rear Ethernet port and should be connected to a switch "
    "port with Power over Ethernet disabled, since the WidgetX unit does not "
    "accept PoE and enabling it can damage the network interface. Once powered "
    "and connected, the unit performs a self-test sequence that takes "
    "approximately ninety seconds, after which the status LED should show solid "
    "green.\n\n"
    "3.1 Common Fault Codes at Startup\n"
    "Fault code E-01 indicates a grounding fault and requires an electrician to "
    "verify the circuit before retrying. Fault code E-02 indicates the ambient "
    "temperature sensor reading is out of range, most often because the unit "
    "was powered on before reaching thermal equilibrium after transport. Fault "
    "code E-07 indicates a network configuration error and typically resolves "
    "after re-running the network setup wizard from the front panel.",
    # Page 3
    "4. Pressure Sensor Calibration Procedure\n"
    "4.1 Preparation\n"
    "Before calibrating the pressure sensor, allow the unit to run for at least "
    "thirty minutes under normal operating load so that internal components "
    "reach a stable operating temperature; calibrating a cold unit produces "
    "readings that drift once the unit warms up. Gather the certified reference "
    "gauge, the calibration adapter fitting included in the technician toolkit, "
    "and a laptop running the WidgetX Configuration Utility version 4.2 or "
    "later, since earlier versions do not support the automated calibration "
    "wizard described below.\n"
    "4.2 Calibration Steps\n"
    "Attach the reference gauge to the calibration port using the supplied "
    "adapter, taking care to fully seat the fitting to avoid a slow leak that "
    "will produce an unstable reading during the hold phase. Launch the "
    "Configuration Utility, select Calibration from the main menu, and follow "
    "the on-screen prompts to apply reference pressure at the three "
    "calibration points: zero, mid-range, and full-scale. At each point, hold "
    "the pressure steady for thirty seconds while the utility samples the "
    "sensor output and computes the correction offset; do not proceed to the "
    "next point until the utility reports a stable reading, since an unstable "
    "sample will silently produce an incorrect offset that only surfaces as "
    "measurement drift weeks later in the field.\n"
    "While the wizard samples each point, avoid touching the calibration "
    "adapter or the cable bundle running to the reference gauge, since even "
    "small mechanical vibration during the thirty-second hold can be picked up "
    "by the sensor and cause the utility to reject the sample and restart the "
    "hold timer for that point. On sites with significant floor vibration from "
    "nearby machinery, technicians are advised to perform calibration during a "
    "scheduled maintenance window when adjacent equipment is powered down, "
    "since repeated sample rejection can extend a routine calibration from its "
    "usual twenty minutes to well over an hour. The Configuration Utility logs "
    "every sample, accepted or rejected, along with a timestamp and the raw "
    "sensor voltage, which is useful when diagnosing an unusually long "
    "calibration session after the fact, and this log is bundled automatically "
    "into the calibration certificate described below so that a service center "
    "can review it if a unit is later returned under warranty.\n"
    "Once all three points are captured, the utility calculates a linear "
    "correction curve and writes it to the sensor's calibration memory, and a "
    "calibration certificate is generated automatically for the maintenance "
    "log. Taking all of the above warm-up, attachment, and three-point "
    "sampling steps into account, the recommended calibration interval for the "
    "WidgetX pressure sensor under normal industrial use is 500 operating "
    "hours, after which drift outside the rated accuracy band becomes "
    "statistically likely.",
    # Page 4
    "5. Firmware Maintenance\n"
    "Check the current firmware version from the Configuration Utility's System "
    "Information screen. Acme publishes firmware updates quarterly, and "
    "security-related patches are released out of cycle when needed. Before "
    "updating firmware, export the current calibration profile, since a "
    "firmware update resets calibration data to factory defaults and the unit "
    "must be recalibrated afterward using the procedure in Section 4. Firmware "
    "updates are applied through the Configuration Utility over the network "
    "connection and take approximately six minutes; do not power-cycle the unit "
    "during this window, as an interrupted update can require a factory reset "
    "by an authorized service center.\n\n"
    "6. Preventive Maintenance Schedule\n"
    "In addition to the pressure sensor calibration described in Section 4, "
    "technicians should inspect the intake filter every 250 operating hours and "
    "replace it if visibly clogged, and perform a full mechanical inspection of "
    "seals and fittings every 1,000 operating hours.",
    # Page 5
    "7. Troubleshooting Guide\n"
    "If the unit reports intermittent pressure readings after the warranty "
    "period, first confirm the calibration interval in Section 4 has not been "
    "exceeded, since drift is the most common root cause of intermittent "
    "readings in units older than one year. If readings remain unstable "
    "immediately after a fresh calibration, inspect the calibration port "
    "fitting for a slow leak, and check that the reference gauge itself is "
    "within its own calibration due date, since a drifted reference gauge will "
    "silently miscalibrate the unit.\n\n"
    "8. Warranty and Service\n"
    "The WidgetX unit carries a two-year limited warranty covering "
    "manufacturing defects, excluding damage from improper installation "
    "(see Section 2 and Section 3) or from skipped calibration intervals. "
    "Contact an authorized service center for any repair; opening the sealed "
    "enclosure voids the warranty.",
]


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _write_pdf("data_security_policy_handbook.pdf", SECURITY_PAGES)
    _write_pdf("widgetx_technical_operations_manual.pdf", TECHOPS_PAGES)


if __name__ == "__main__":
    main()
