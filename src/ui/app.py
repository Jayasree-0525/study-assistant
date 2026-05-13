"""Streamlit study assistant app."""

import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.study_agent import ask, build_agent
from ingestion.pipeline import ingest_document

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide",
)

# ── initialise session state ──────────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state["agent"] = None

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

if "page" not in st.session_state:
    st.session_state["page"] = "upload"


# ── helper: build agent once ──────────────────────────────────────────────────
def get_agent():
    """Build the agent once and cache it in session state."""
    if st.session_state["agent"] is None:
        with st.spinner("Initialising study assistant..."):
            st.session_state["agent"] = build_agent()
    return st.session_state["agent"]


# ── helper: save uploaded file to disk ───────────────────────────────────────
def save_uploaded_file(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to data/raw/ and return the path."""
    save_dir = Path("data/raw")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(save_path)


# ── upload page ───────────────────────────────────────────────────────────────
def render_upload_page():
    st.title("📚 Study Assistant")
    st.markdown("Upload your lecture notes and I'll help you understand them.")

    st.divider()

    uploaded = st.file_uploader(
        "Upload lecture PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF lecture files. Max 50MB each.",
    )

    if uploaded:
        new_files = [f for f in uploaded if f.name not in st.session_state["uploaded_files"]]

        if new_files:
            st.info(f"Processing {len(new_files)} new file(s)...")
            progress = st.progress(0)

            for i, file in enumerate(new_files):
                with st.spinner(f"Ingesting {file.name}..."):
                    try:
                        file_path = save_uploaded_file(file)
                        result = ingest_document(
                            file_path,
                            extract_images=False,
                        )
                        st.session_state["uploaded_files"].append(file.name)
                        st.success(f"✅ {file.name} — " f"{result['total_chunks']} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ Failed to process {file.name}: {e}")

                progress.progress((i + 1) / len(new_files))

        else:
            st.info("These files are already uploaded.")

    if st.session_state["uploaded_files"]:
        st.divider()
        st.markdown("**Uploaded files:**")
        for fname in st.session_state["uploaded_files"]:
            st.markdown(f"- 📄 {fname}")

        st.divider()
        if st.button(
            "Start chatting →",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["page"] = "chat"
            st.rerun()
    else:
        st.markdown("_Upload at least one PDF to start chatting._")


# ── chat page ─────────────────────────────────────────────────────────────────
def render_chat_page():
    # sidebar
    with st.sidebar:
        st.markdown("### 📚 Study Assistant")
        st.divider()

        st.markdown("**Uploaded files:**")
        if st.session_state["uploaded_files"]:
            for fname in st.session_state["uploaded_files"]:
                st.markdown(f"- 📄 {fname}")
        else:
            st.markdown("_No files uploaded yet_")

        st.divider()

        if st.button("⬆️ Upload more files", use_container_width=True):
            st.session_state["page"] = "upload"
            st.rerun()

        if st.button(
            "🗑️ Clear conversation",
            use_container_width=True,
        ):
            st.session_state["messages"] = []
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.session_state["agent"] = None
            st.rerun()

        st.divider()
        st.caption("Powered by GPT-4o-mini · ChromaDB · LangGraph")

    # main chat area
    st.title("💬 Chat with your notes")

    if not st.session_state["uploaded_files"]:
        st.warning("No files uploaded yet. " "Go back to upload your lecture PDFs first.")
        if st.button("← Upload files"):
            st.session_state["page"] = "upload"
            st.rerun()
        return

    # display chat history
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # suggested questions if no messages yet
    if not st.session_state["messages"]:
        st.markdown("**Try asking:**")
        suggestions = [
            "What are the Five Forces in Porter's model?",
            "When is buyer power stronger?",
            "What are the most powerful barriers to entry?",
            "What forces are driving industry change?",
            "Summarise the key topics in these notes",
        ]
        cols = st.columns(2)
        for i, suggestion in enumerate(suggestions):
            if cols[i % 2].button(suggestion, use_container_width=True):
                st.session_state["messages"].append(
                    {
                        "role": "user",
                        "content": suggestion,
                    }
                )
                st.rerun()

    # chat input
    if prompt := st.chat_input("Ask a question about your notes..."):
        st.session_state["messages"].append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching your notes..."):
                agent = get_agent()
                response = ask(
                    prompt,
                    agent,
                    thread_id=st.session_state["thread_id"],
                )

            st.markdown(response)

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": response,
            }
        )


# ── router ────────────────────────────────────────────────────────────────────
def main():
    if st.session_state["page"] == "upload":
        render_upload_page()
    else:
        render_chat_page()


if __name__ == "__main__":
    main()
