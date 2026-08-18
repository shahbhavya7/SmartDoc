"""Documents: list, upload, delete.

Every action is a call to the real endpoint. Delete in particular goes through
`DELETE /documents/{id}`, which removes the SQLite row AND the document's chunks
from the vector store -- the response reports both, so a partial delete is visible
here rather than silently leaving orphaned chunks behind.
"""

from __future__ import annotations

import streamlit as st

from streamlit_demo import api


def _fetch(token: str) -> dict | None:
    try:
        return api.list_documents(token)
    except api.ApiError as exc:
        st.error(str(exc))
        return None


def render(token: str) -> None:
    st.subheader("Documents")

    listing = _fetch(token)
    if listing is None:
        return

    docs = listing.get("documents", [])

    # -- Upload -----------------------------------------------------------
    uploaded = st.file_uploader(
        "Upload a PDF", type=["pdf"], accept_multiple_files=False
    )
    if uploaded is not None:
        upload_key = f"{uploaded.name}:{uploaded.size}"
        # Streamlit re-runs the script on every interaction and keeps the
        # uploaded file in the widget, so without this guard the same PDF would
        # be re-ingested on each rerun.
        if st.session_state.get("last_upload") != upload_key:
            with st.spinner(f"Parsing, chunking, and embedding {uploaded.name}"):
                try:
                    result = api.upload_document(
                        token, uploaded.name, uploaded.getvalue()
                    )
                except api.ApiError as exc:
                    st.error(str(exc))
                    result = None

            if result is not None:
                st.session_state["last_upload"] = upload_key
                for entry in result.get("files", []):
                    if entry.get("status") == "success":
                        st.success(
                            f"Indexed {entry.get('filename')} -- "
                            f"{entry.get('pages_parsed') or 0} pages, "
                            f"{entry.get('chunks_indexed') or 0} chunks searchable."
                        )
                    else:
                        st.error(
                            f"{entry.get('filename')}: "
                            f"{entry.get('error') or 'the server did not say why.'}"
                        )
                st.rerun()

    st.divider()

    # -- List -------------------------------------------------------------
    if not docs:
        st.info("No documents yet. Upload a PDF to get started.")
        return

    st.caption(
        f"{len(docs)} document(s), {listing.get('total_chunks', 0)} indexed chunks. "
        "Chunk count is shown because the API does not return a page count for "
        "already-indexed documents."
    )

    header = st.columns([4, 1, 2, 1])
    header[0].markdown("**Document**")
    header[1].markdown("**Chunks**")
    header[2].markdown("**Added**")
    header[3].markdown("**Action**")

    for doc in docs:
        row = st.columns([4, 1, 2, 1])
        row[0].text(doc.get("filename", "unknown"))
        row[1].text(str(doc.get("chunks") if doc.get("chunks") is not None else "-"))
        row[2].text((doc.get("created_at") or "")[:10])
        if row[3].button("Delete", key=f"delete-{doc['id']}"):
            try:
                result = api.delete_document(token, doc["id"])
            except api.ApiError as exc:
                st.error(str(exc))
            else:
                st.success(
                    f"Deleted {result.get('filename')} -- "
                    f"{result.get('chunks_deleted', 0)} chunks removed from the "
                    f"vector store, database row deleted: "
                    f"{result.get('row_deleted')}."
                )
                st.rerun()
