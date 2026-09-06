# document_indexer.py
"""
Azure Document Intelligence Ingestion Engine.
Handles PDF, DOCX, XLSX, TXT, and Image files using Azure Document Intelligence
(model: prebuilt-layout, API: 2024-11-30).
Chunks text with page/sheet citations, generates vector embeddings, and stores
in Azure AI Search and the Unified Knowledge Graph.
"""

import os
import time
import uuid
import requests
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

import config
import blob_storage
import knowledge_graph
from search_index import embed_text, ensure_index_exists


def _get_content_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    mapping = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".txt": "text/plain",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    return mapping.get(ext, "application/octet-stream")


def analyze_with_azure_doc_intel(file_path: str) -> dict:
    """Submits a document/image to Azure Document Intelligence and polls for results."""
    endpoint = (config.AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or "").rstrip("/")
    key = config.AZURE_DOCUMENT_INTELLIGENCE_KEY

    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence endpoint or key is not configured in .env")

    analyze_url = f"{endpoint}/documentintelligence/documentModels/prebuilt-layout:analyze?api-version=2024-11-30&outputContentFormat=markdown"
    content_type = _get_content_type(file_path)

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": content_type,
    }

    # Submit analysis request
    resp = requests.post(analyze_url, headers=headers, data=file_bytes, timeout=60)
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"Azure Document Intelligence submission failed ({resp.status_code}): {resp.text}")

    operation_location = resp.headers.get("Operation-Location")
    if not operation_location:
        # Some versions return result directly if small
        return resp.json().get("analyzeResult", {})

    # Poll operation
    for _ in range(60):  # Wait up to 2 minutes
        time.sleep(2.0)
        poll_resp = requests.get(operation_location, headers={"Ocp-Apim-Subscription-Key": key}, timeout=30)
        if poll_resp.status_code == 200:
            result_json = poll_resp.json()
            status = result_json.get("status")
            if status == "succeeded":
                return result_json.get("analyzeResult", {})
            elif status == "failed":
                raise RuntimeError(f"Azure Document Intelligence analysis failed: {result_json.get('error')}")
        else:
            time.sleep(1.0)

    raise TimeoutError("Azure Document Intelligence timed out while analyzing the document.")


def extract_document_chunks(doc_id: str, file_path: str, display_name: str) -> list[dict]:
    """
    Extracts text chunks with location citations (e.g. Page 1, Sheet 1, Image).
    Returns list of chunk dicts ready for indexing.
    """
    ext = os.path.splitext(file_path)[1].lower()
    source_type = ext.lstrip(".")
    if source_type in ["jpg", "jpeg", "png", "webp"]:
        source_type = "image"

    chunks = []

    # If it's a plain text file, process directly
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        words = raw_text.split()
        chunk_size = 200
        stride = 150
        for i in range(0, max(len(words), 1), stride):
            sub_words = words[i:i + chunk_size]
            if not sub_words:
                break
            part_num = (i // stride) + 1
            chunks.append({
                "doc_id": doc_id,
                "display_name": display_name,
                "source_type": "txt",
                "location": f"Part {part_num}",
                "page_num": float(part_num),
                "text": " ".join(sub_words),
            })
        return chunks

    # For PDF, DOCX, XLSX, Images: use Azure Document Intelligence
    print(f"[DocIntel] Analyzing '{display_name}' via Azure Document Intelligence...")
    analyze_result = analyze_with_azure_doc_intel(file_path)

    pages = analyze_result.get("pages", [])
    markdown_content = analyze_result.get("content", "")

    if pages:
        for p in pages:
            page_num = p.get("pageNumber", 1)
            # Combine lines on this page
            lines = [line.get("content", "") for line in p.get("lines", [])]
            page_text = "\n".join(lines).strip()
            
            if not page_text:
                continue

            # If page text is long (> 300 words), split into smaller chunks
            words = page_text.split()
            if len(words) > 300:
                chunk_size = 250
                stride = 180
                for j in range(0, len(words), stride):
                    sub = words[j:j + chunk_size]
                    if not sub:
                        break
                    part_str = f"Part {j//stride + 1}" if len(words) > chunk_size else ""
                    location_label = f"Page {page_num}" if not part_str else f"Page {page_num} ({part_str})"
                    if source_type == "xlsx":
                        location_label = f"Sheet Data - Page {page_num}"
                    elif source_type == "image":
                        location_label = f"Image: {display_name}"

                    chunks.append({
                        "doc_id": doc_id,
                        "display_name": display_name,
                        "source_type": source_type,
                        "location": location_label,
                        "page_num": float(page_num),
                        "text": " ".join(sub),
                    })
            else:
                location_label = f"Page {page_num}"
                if source_type == "xlsx":
                    location_label = f"Sheet Data - Page {page_num}"
                elif source_type == "image":
                    location_label = f"Image: {display_name}"

                    chunks.append({
                        "doc_id": doc_id,
                        "display_name": display_name,
                        "source_type": source_type,
                        "location": location_label,
                        "page_num": float(page_num),
                        "text": page_text,
                    })

    # Index tabular data with precise row & column citations if tables exist
    tables = analyze_result.get("tables", [])
    if tables:
        for t_idx, table in enumerate(tables):
            row_count = table.get("rowCount", 0)
            col_count = table.get("columnCount", 0)
            cells = table.get("cells", [])
            
            # Identify page of the table
            table_page = 1
            if table.get("boundingRegions"):
                table_page = table["boundingRegions"][0].get("pageNumber", 1)

            # Reconstruct table rows
            grid = {}
            for cell in cells:
                r = cell.get("rowIndex", 0)
                c = cell.get("columnIndex", 0)
                txt = cell.get("content", "").strip()
                if r not in grid:
                    grid[r] = {}
                grid[r][c] = txt

            # Build markdown table representation in blocks of up to 15 rows
            row_stride = 15
            for start_r in range(0, max(row_count, 1), row_stride):
                end_r = min(start_r + row_stride, row_count)
                row_lines = []
                for r in range(start_r, end_r):
                    cols = [grid.get(r, {}).get(c, "") for c in range(col_count)]
                    row_lines.append(" | ".join(cols))
                
                table_text = "\n".join(row_lines)
                if table_text.strip():
                    if source_type == "xlsx":
                        tbl_loc = f"Sheet (Rows {start_r + 1}-{end_r})"
                    else:
                        tbl_loc = f"Page {table_page} - Table (Rows {start_r + 1}-{end_r})"

                    chunks.append({
                        "doc_id": doc_id,
                        "display_name": display_name,
                        "source_type": source_type,
                        "location": tbl_loc,
                        "page_num": float(table_page),
                        "text": f"Table data for {display_name} ({tbl_loc}):\n{table_text}",
                    })

    # Fallback if pages structure was empty but full markdown exists
    if not chunks and markdown_content:
        chunks.append({
            "doc_id": doc_id,
            "display_name": display_name,
            "source_type": source_type,
            "location": f"Document: {display_name}",
            "page_num": 1.0,
            "text": markdown_content[:2000],
        })

    return chunks


def upload_document_chunks(chunks: list[dict]):
    """Uploads document chunks with vector embeddings to Azure AI Search."""
    ensure_index_exists()
    search_client = SearchClient(
        endpoint=config.SEARCH_ENDPOINT,
        index_name=config.SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(config.SEARCH_ADMIN_KEY),
    )

    docs = []
    for i, c in enumerate(chunks):
        docs.append({
            "id": f"{c['doc_id']}-{i}",
            "video_id": c["doc_id"],  # Maps to document ID
            "source_type": c["source_type"],
            "start_time": c.get("page_num", 1.0),
            "end_time": c.get("page_num", 1.0),
            "text": f"[{c['display_name']} | {c['location']}]\n{c['text']}",
            "keyframe_thumbnail_ids": [f"loc:{c['location']}", f"file:{c['display_name']}"],
            "text_vector": embed_text(c["text"]),
        })

    for i in range(0, len(docs), 1000):
        search_client.upload_documents(documents=docs[i:i + 1000])
    print(f"[DocIntel] Uploaded {len(docs)} chunks to Azure AI Search successfully.")


def ingest_document(local_path: str, display_name: str) -> tuple[str, str, str]:
    """
    End-to-end ingestion of any Document / Image / Spreadsheet:
    1. Upload to Azure Blob Storage
    2. Analyze via Azure Document Intelligence
    3. Generate chunks and embeddings into Azure AI Search
    4. Extract Knowledge Graph entities into knowledge_graph.json
    Returns (doc_id, blob_name, source_type).
    """
    doc_id = str(uuid.uuid4())
    ext = os.path.splitext(local_path)[1].lower()
    source_type = ext.lstrip(".")
    if source_type in ["jpg", "jpeg", "png", "webp"]:
        source_type = "image"

    # 1. Upload raw document to Azure Blob Storage for preview/archival
    blob_name = f"doc_{doc_id}{ext}"
    print(f"[DocIntel] [1/4] Uploading '{display_name}' to Azure Blob Storage as '{blob_name}'...")
    blob_storage.upload_blob(local_path, blob_name)

    # 2. Extract content & chunks using Azure Document Intelligence
    print(f"[DocIntel] [2/4] Extracting text & structure using Azure Document Intelligence...")
    chunks = extract_document_chunks(doc_id, local_path, display_name)

    # 3. Vector indexing in Azure AI Search
    print(f"[DocIntel] [3/4] Indexing {len(chunks)} chunks in Azure AI Search...")
    upload_document_chunks(chunks)

    # 4. Merge into Knowledge Graph
    print(f"[DocIntel] [4/4] Extracting entities and relations into Knowledge Graph...")
    try:
        # Adapt chunk objects for knowledge_graph.extract_and_merge
        kg_chunks = []
        for c in chunks:
            class DummyChunk:
                pass
            d = DummyChunk()
            d.start = c.get("page_num", 1.0)
            d.end = c.get("page_num", 1.0)
            d.text = c["text"]
            kg_chunks.append(d)
        knowledge_graph.extract_and_merge(doc_id, display_name, kg_chunks)
    except Exception as e:
        print(f"[DocIntel] Knowledge graph extraction warning: {e}")

    print(f"[DocIntel] Ingestion complete for '{display_name}' (ID: {doc_id})!")
    return doc_id, blob_name, source_type
