"""Integration test for the RAG retrieval, generation, and parsing pipeline."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from rag.generator.review_generator import ReviewConfig, ReviewGenerator
from rag.retriever.hybrid import HybridRetriever
from rag.retriever.keyword_search import KeywordSearcher
from rag.retriever.vector_store import VectorStore

FIXED_LLM_RESPONSE = json.dumps(
    {
        "skills_feedback": {
            "content": "Strong Python and FastAPI experience.",
            "suggestions": ["Add automated tests."],
        }
    }
)


class FakeCompletions:
    """Return a deterministic response instead of calling a real LLM."""

    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeOpenAIClient:
    """Minimal OpenAI-compatible client used by ReviewGenerator."""

    def __init__(self, content: str):
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


@pytest.mark.integration
def test_rag_pipeline_with_fixed_llm_response(tmp_path, monkeypatch):
    """Run retrieval -> generation -> parsing without a real LLM request."""

    fixture_path = (
        Path(__file__).parent.parent / "fixtures" / "sample_profiles" / "basic_profile.json"
    )

    profile_data = json.loads(fixture_path.read_text())

    # Build deterministic chunks from the issue's sample profile.
    chunks = []

    for index, repo in enumerate(profile_data["repos"]):
        chunks.append(
            {
                "id": f"repo-{index}",
                "text": (
                    f"{repo['name']}\n" f"{repo['description']}\n" f"{repo['readme_content']}"
                ),
                "metadata": {
                    "source_id": repo["html_url"],
                    "chunk_index": index,
                    "section": "repository",
                },
            }
        )

    # Use deterministic embeddings so no embedding API is required.
    embedding_size = len(chunks)
    embeddings = []

    for index in range(embedding_size):
        embedding = [0.0] * embedding_size
        embedding[index] = 1.0
        embeddings.append(embedding)

    # Store the chunks in a real temporary ChromaDB vector store.
    vector_store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    profile_id = "integration-test"
    collection_name = f"profile_{profile_id}"
    collection = vector_store.get_collection(collection_name)

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        embeddings=embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )

    # Build the real keyword index from the same chunks.
    keyword_searcher = KeywordSearcher()
    keyword_searcher.index(chunks)

    retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_searcher=keyword_searcher,
        vector_weight=1.0,
        keyword_weight=0.0,
    )

    # Run the real retrieval path.
    retrieved_chunks = retriever.retrieve(
        query="Python FastAPI project",
        profile_id=profile_id,
        query_embedding=embeddings[0],
        max_chunks=len(chunks),
        min_score=0.0,
    )

    assert retrieved_chunks
    assert any("transit-delay-tracker" in chunk["text"] for chunk in retrieved_chunks)

    # Make a real OpenAI client creation fail immediately if the injected
    # client is accidentally ignored.
    def fail_if_live_client_is_created(*args, **kwargs):
        raise AssertionError("A real OpenAI client should not be created")

    monkeypatch.setattr(
        "rag.generator.review_generator.openai.OpenAI",
        fail_if_live_client_is_created,
    )

    fake_client = FakeOpenAIClient(FIXED_LLM_RESPONSE)

    config = ReviewConfig(
        api_key="unused",
        base_url="https://example.invalid",
        model="mock-model",
    )

    generator = ReviewGenerator(config=config, client=fake_client)

    # Run generation and parsing using the retrieved chunks.
    section = generator.generate_section(
        section_name="skills_feedback",
        context_chunks=retrieved_chunks,
        profile_data=profile_data,
    )

    # Verify generation used the fake LLM exactly once.
    assert len(fake_client.chat.completions.calls) == 1

    # Verify retrieved context actually reached the generation prompt.
    call = fake_client.chat.completions.calls[0]
    prompt = call["messages"][1]["content"]

    assert "transit-delay-tracker" in prompt

    # Verify the fixed LLM response was parsed into structured feedback.
    assert section.section_name == "skills_feedback"
    assert section.confidence == 0.9
    assert section.suggestions == ["Add automated tests."]
