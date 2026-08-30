# knowledge_graph.py
"""
Lightweight, 0-cost Local Knowledge Graph (GraphRAG) engine.
Extracts Entity-Relationship triples from transcripts/OCR and maintains a
local JSON graph (knowledge_graph.json) without extra cloud costs.
"""

import json
import os
import re
from openai import AzureOpenAI

GRAPH_FILE = os.path.join(os.path.dirname(__file__), "knowledge_graph.json")


def _get_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        timeout=30.0,
        max_retries=2,
    )


def load_graph() -> dict:
    if os.path.exists(GRAPH_FILE):
        try:
            with open(GRAPH_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"nodes": [], "edges": []}


def save_graph(graph_data: dict):
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2)


def populate_from_registry():
    """Populates Knowledge Graph for all existing videos in videos_registry.json."""
    registry_file = os.path.join(os.path.dirname(__file__), "videos_registry.json")
    if not os.path.exists(registry_file):
        return
    try:
        with open(registry_file, "r", encoding="utf-8") as f:
            registry = json.load(f)
    except Exception:
        return

    from video_indexer_client import VideoIndexerClient
    from merge_and_chunk import build_chunks

    vi = VideoIndexerClient()
    for local_id, info in registry.items():
        vi_video_id = info.get("vi_video_id")
        display_name = info.get("display_name", "Untitled")
        if not vi_video_id:
            continue
        try:
            print(f"[KnowledgeGraph] Indexing existing video: '{display_name}' ({vi_video_id})...")
            insights = vi.wait_for_processing(vi_video_id, timeout_seconds=10)
            chunks = build_chunks(vi_video_id, insights)
            extract_and_merge(vi_video_id, display_name, chunks)
        except Exception as e:
            print(f"[KnowledgeGraph] Skipping {display_name}: {e}")


def extract_and_merge(video_id: str, video_title: str, chunks: list) -> dict:
    """Extracts key entities & relations from video chunks and merges into knowledge_graph.json."""
    graph = load_graph()
    
    # Check if video node already exists
    video_node_id = f"video_{video_id}"
    nodes_dict = {n["id"]: n for n in graph.get("nodes", [])}
    edges_list = graph.get("edges", [])

    # Add/Update root video node
    nodes_dict[video_node_id] = {
        "id": video_node_id,
        "label": video_title,
        "group": "video",
        "shape": "hexagon",
        "color": {"background": "#DD1D21", "border": "#b91c1c", "highlight": {"background": "#FFD500", "border": "#DD1D21"}},
        "font": {"color": "#ffffff", "face": "Inter", "size": 14},
        "title": f"Video: {video_title}"
    }

    # Aggregate transcript text from chunks (up to 3000 chars)
    combined_text = "\n".join([c.text for c in chunks[:10]])[:3000]

    prompt = (
        "Extract up to 8 key concepts, tools, or topics discussed in this video text, "
        "and how they relate to each other or the main video topic.\n\n"
        f"Video Title: {video_title}\n"
        f"Text Context:\n{combined_text}\n\n"
        "Return a JSON object with this structure:\n"
        "{\n"
        '  "entities": [\n'
        '    {"name": "Concept Name", "category": "topic"}\n'
        '  ],\n'
        '  "relationships": [\n'
        '    {"source": "Concept Name", "relation": "explains / uses / configures", "target": "Other Concept Name"}\n'
        '  ]\n'
        "}\n"
        "Categories should be one of: 'topic', 'tool', 'action', 'setting'."
    )

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        data = json.loads(resp.choices[0].message.content)

        entities = data.get("entities", [])
        relationships = data.get("relationships", [])

        # Color mapping for entity groups
        group_colors = {
            "topic": {"bg": "#eff6ff", "border": "#3b82f6", "text": "#1d4ed8"},
            "tool": {"bg": "#f0fdf4", "border": "#22c55e", "text": "#15803d"},
            "action": {"bg": "#fef3c7", "border": "#f59e0b", "text": "#b45309"},
            "setting": {"bg": "#f3e8ff", "border": "#a855f7", "text": "#6b21a8"}
        }

        # Add entities to nodes
        for ent in entities:
            name = ent.get("name", "").strip()
            if not name:
                continue
            cat = ent.get("category", "topic").lower()
            colors = group_colors.get(cat, group_colors["topic"])
            
            node_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', name.lower())}"
            if node_id not in nodes_dict:
                nodes_dict[node_id] = {
                    "id": node_id,
                    "label": name,
                    "group": cat,
                    "shape": "box",
                    "margin": 10,
                    "color": {
                        "background": colors["bg"],
                        "border": colors["border"],
                        "highlight": {"background": "#ffffff", "border": "#DD1D21"}
                    },
                    "font": {"color": colors["text"], "face": "Inter", "size": 13},
                    "title": f"Category: {cat.capitalize()}"
                }
            
            # Connect video node to top entities
            edge_id = f"{video_node_id}_to_{node_id}"
            if not any(e.get("id") == edge_id for e in edges_list):
                edges_list.append({
                    "id": edge_id,
                    "from": video_node_id,
                    "to": node_id,
                    "label": "covers",
                    "color": {"color": "#cbd5e1", "highlight": "#DD1D21"},
                    "font": {"size": 10, "color": "#64748b"}
                })

        # Add relationship edges
        for rel in relationships:
            src = rel.get("source", "").strip()
            tgt = rel.get("target", "").strip()
            relation = rel.get("relation", "relates to").strip()

            if src and tgt:
                src_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', src.lower())}"
                tgt_id = f"ent_{re.sub(r'[^a-zA-Z0-9_]', '_', tgt.lower())}"

                if src_id in nodes_dict and tgt_id in nodes_dict:
                    edge_id = f"{src_id}_rel_{tgt_id}"
                    if not any(e.get("id") == edge_id for e in edges_list):
                        edges_list.append({
                            "id": edge_id,
                            "from": src_id,
                            "to": tgt_id,
                            "label": relation,
                            "arrows": "to",
                            "color": {"color": "#94a3b8", "highlight": "#3b82f6"},
                            "font": {"size": 10, "color": "#475569"}
                        })

    except Exception as e:
        print(f"Warning: Knowledge Graph extraction skipped/failed for {video_id}: {e}")

    updated_graph = {
        "nodes": list(nodes_dict.values()),
        "edges": edges_list
    }
    save_graph(updated_graph)
    return updated_graph
