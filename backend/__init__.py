"""SmartDoc backend package: FastAPI app and RAG pipeline modules.

Telemetry is disabled here, before any submodule imports ``chromadb``.
ChromaDB reads ``ANONYMIZED_TELEMETRY`` when its client is constructed, and the
installed posthog version is incompatible with its telemetry call, so leaving it
enabled prints a "capture() takes 1 positional argument" warning on every store
operation. Setting it inside a module that itself imports chromadb would be too
late -- the package initialiser is the only reliable place.
"""

import logging
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

# The setting above is not sufficient on chromadb 0.5.23: its posthog telemetry
# client attempts the call regardless and logs the failure at ERROR on every
# store operation. Silencing that one logger removes the noise without
# suppressing anything the application itself reports -- and nothing is
# transmitted either way, since telemetry is also disabled at the client (see
# backend.vectorstore.get_client).
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
