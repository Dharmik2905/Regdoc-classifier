import streamlit as st
from backend.ingestion import process_file
from backend.classification import classify_document
from backend.storage import save_result, load_history


st.set_page_config(page_title="RegDoc Classifier", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Analyze", "History & Audit"])

st.sidebar.markdown("---")
st.sidebar.caption("Hitachi DS Datathon • AI-Powered Regulatory Classifier")


if page == "Upload & Analyze":
    st.title("Regulatory Document Classifier")
    st.caption(
        "Classify documents as **Public**, **Confidential**, **Highly Sensitive**, "
        "or **Unsafe**, with page-level evidence and human review."
    )

    uploaded_files = st.file_uploader(
        "Upload one or more documents (PDF or image)",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.markdown("---")
            st.subheader(f"File: {uploaded_file.name}")

            with st.spinner("Pre-processing document..."):
                doc_info = process_file(uploaded_file)

            col1, col2, col3 = st.columns(3)
            col1.metric("Pages", doc_info["num_pages"])
            col2.metric("Images", doc_info["num_images"])
            col3.metric("Legible", "Yes" if doc_info["legible"] else "Check manually")

            with st.expander("View page summaries"):
                for p in doc_info["pages"]:
                    st.markdown(f"**Page {p['page_num']}**")
                    st.write(p["text"][:500] or "*No text extracted*")

            with st.spinner("Running classification..."):
                result = classify_document(doc_info)

            st.markdown("### Classification Result")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Category", result["category"])
            c2.metric("Unsafe", "Yes" if result["unsafe"] else "No")
            c3.metric("Kid-safe", "Yes" if result["kid_safe"] else "No")
            c4.metric("Confidence", f"{result['confidence']*100:.0f}%")

            st.markdown("**Reasoning**")
            st.write(result["reasoning"])

            st.markdown("**Citations (pages and reasons)**")
            if result["citations"]:
                st.table(result["citations"])
            else:
                st.write("No specific citations generated yet.")

            st.markdown("### Human-in-the-Loop Review")

            override = st.selectbox(
                "Override AI category (optional)",
                [
                    "No override",
                    "Public",
                    "Confidential",
                    "Highly Sensitive",
                    "Unsafe",
                    "Confidential and Unsafe",
                ],
                key=f"override_{uploaded_file.name}",
            )
            comment = st.text_area(
                "Reviewer comment (optional)",
                key=f"comment_{uploaded_file.name}",
            )

            if st.button("Save Review", key=f"save_{uploaded_file.name}"):
                final_category = (
                    result["category"] if override == "No override" else override
                )
                save_result(
                    filename=uploaded_file.name,
                    doc_info=doc_info,
                    ai_result=result,
                    final_category=final_category,
                    reviewer_comment=comment,
                )
                st.success("Review saved to audit log.")

elif page == "History & Audit":
    st.title("History & Audit Trail")

    history = load_history()
    if not history:
        st.info("No documents processed yet. Go to 'Upload & Analyze', run a document, and click 'Save Review'.")
    else:
        import pandas as pd

        df = pd.DataFrame(history)

        st.subheader("Processed Documents")
        st.dataframe(
            df[[
                "timestamp",
                "filename",
                "pages",
                "images",
                "ai_category",
                "final_category",
                "unsafe",
                "kid_safe",
                "confidence",
            ]].sort_values("timestamp", ascending=False),
            use_container_width=True,
        )

        st.subheader("Category Distribution (Final)")
        st.bar_chart(df["final_category"].value_counts())

        st.subheader("Unsafe vs Kid-safe")
        col1, col2 = st.columns(2)
        col1.metric("Total Unsafe", int(df["unsafe"].sum()))
        col2.metric("Kid-safe Documents", int(df["kid_safe"].sum()))

        selected = st.selectbox(
            "Inspect a document",
            options=df.sort_values("timestamp", ascending=False)["filename"].unique(),
        )

        doc_rows = df[df["filename"] == selected].sort_values("timestamp", ascending=False)
        latest = doc_rows.iloc[0]

        st.markdown(f"### Latest review for: `{selected}`")
        st.write(f"**AI Category:** {latest['ai_category']}")
        st.write(f"**Final Category:** {latest['final_category']}")
        st.write(f"**Unsafe:** {latest['unsafe']}")
        st.write(f"**Kid-safe:** {latest['kid_safe']}")
        st.write(f"**Confidence:** {latest['confidence']:.2f}")
        st.write(f"**Reviewer comment:** {latest.get('reviewer_comment') or '—'}")
        st.write(f"**Timestamp (UTC):** {latest['timestamp']}")

