# search_index.py
"""
Creates (if needed) the Azure AI Search index and uploads chunks into it.
Embeddings are generated locally via sentence-transformers, so this never
depends on Azure OpenAI approval or billing.
"""

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    SearchField,
)
from sentence_transformers import SentenceTransformer

import config

_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def embed_text(text: str) -> list:
    return _get_embedder().encode(text).tolist()


def ensure_index_exists():
    index_client = SearchIndexClient(
        endpoint=config.SEARCH_ENDPOINT,
        credential=AzureKeyCredential(config.SEARCH_ADMIN_KEY),
    )
    existing = [i.name for i in index_client.list_indexes()]
    if config.SEARCH_INDEX_NAME in existing:
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="video_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="start_time", type=SearchFieldDataType.Double, filterable=True, sortable=True),
        SimpleField(name="end_time", type=SearchFieldDataType.Double, filterable=True),
        SearchableField(name="text", type=SearchFieldDataType.String),
        SimpleField(name="keyframe_thumbnail_ids", type=SearchFieldDataType.Collection(SearchFieldDataType.String)),
        SearchField(
            name="text_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=config.EMBEDDING_DIM,
            vector_search_profile_name="default-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        profiles=[VectorSearchProfile(name="default-profile", algorithm_configuration_name="default-hnsw")],
    )

    index = SearchIndex(name=config.SEARCH_INDEX_NAME, fields=fields, vector_search=vector_search)
    index_client.create_index(index)


def upload_chunks(chunks: list):
    """chunks: list[merge_and_chunk.Chunk]"""
    ensure_index_exists()
    search_client = SearchClient(
        endpoint=config.SEARCH_ENDPOINT,
        index_name=config.SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(config.SEARCH_ADMIN_KEY),
    )

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "id": f"{chunk.video_id}-{i}",
            "video_id": chunk.video_id,
            "source_type": "video",
            "start_time": chunk.start,
            "end_time": chunk.end,
            "text": chunk.text,
            "keyframe_thumbnail_ids": [k.thumbnail_id for k in chunk.keyframes],
            "text_vector": embed_text(chunk.text),
        })

    # Search accepts up to 1000 docs per batch; chunk lists are small per video
    # but this keeps it safe if you ever batch multiple videos together.
    for i in range(0, len(docs), 1000):
        search_client.upload_documents(documents=docs[i:i + 1000])


def search_top_chunks(query: str, video_id: str = None, video_map: dict = None, top_k: int = 3) -> list:
    """Returns the top_k matching chunk documents for a user question."""
    search_client = SearchClient(
        endpoint=config.SEARCH_ENDPOINT,
        index_name=config.SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(config.SEARCH_ADMIN_KEY),
    )
    query_vector = embed_text(query)
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="text_vector",
    )
    
    filter_expr = None
    if video_id:
        filter_expr = f"video_id eq '{video_id}'"
    elif video_map:
        valid_ids = list(video_map.keys())
        if valid_ids:
            filter_expr = " or ".join(f"video_id eq '{vid}'" for vid in valid_ids)

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        filter=filter_expr,
        select=["id", "video_id", "text", "start_time", "end_time", "keyframe_thumbnail_ids"],
    )
    return list(results)