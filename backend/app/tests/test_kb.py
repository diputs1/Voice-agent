import pytest

from app.embeddings import EmbeddingProvider
from app.ingestion import IngestedChunk
from app.kb import InMemoryKnowledgeBase, _adjusted_score, _diversify_categories, _normalize_text, _rrf_fuse

from datetime import UTC, datetime


def test_safari_price_chunk_beats_off_topic_price_chunk():
    query = "Giá vé hôm nay bao nhiêu tiền?"
    safari_row = {
        "title": "[MỚI] Bảng giá vé Vinpearl Safari Phú Quốc",
        "section": "Bảng giá vé Vinpearl Safari Phú Quốc mới nhất",
        "category": "price",
        "content": "Giá vé Vinpearl Safari Phú Quốc từ 710.000 VND đến 950.000 VND.",
        "source_url": "https://vinwonders.com/vi/wonderpedia/news/gia-ve-vinpearl-safari-phu-quoc/",
        "language": "vi",
    }
    nha_trang_row = {
        "title": "Ưu đãi ẩm thực VinWonders Nha Trang",
        "section": "Combo Vé Vào Cửa VinWonders Nha Trang",
        "category": "price",
        "content": "Combo buffet và vé vào cửa VinWonders Nha Trang giá 620.000 VNĐ.",
        "source_url": "https://vinwonders.com/vi/uu-dai/combo-am-thuc-vinwonders-nha-trang/",
        "language": "vi",
    }

    assert _adjusted_score(query, safari_row, 0.1) > _adjusted_score(query, nha_trang_row, 0.4)


def test_safari_show_chunk_beats_other_show_chunk():
    query = "Show biểu diễn động vật diễn ra lúc nào?"
    safari_row = {
        "title": "Vinpearl Safari Phú Quốc | Official Website",
        "section": "Must-see events",
        "category": "schedule",
        "content": "Animal Show 10:00 - 10:30 | 14:00 - 14:30.",
        "source_url": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        "language": "vi",
    }
    other_show_row = {
        "title": "Ưu đãi mở bán show Đất Nước Thiên Hùng Ca",
        "section": "Show diễn Đất Nước Thiên Hùng Ca",
        "category": "booking",
        "content": "Show diễn ra tại Vinpearl Theatre Ocean City.",
        "source_url": "https://vinwonders.com/vi/uu-dai/mo-ban-show-dat-nuoc-thien-hung-ca/",
        "language": "vi",
    }

    assert _adjusted_score(query, safari_row, 0.1) > _adjusted_score(query, other_show_row, 0.4)


def test_diversify_categories_does_not_duplicate_extra_picks():
    rows = [
        (1.0, {"id": "price-1", "category": "price"}),
        (0.9, {"id": "offer-1", "category": "offer"}),
        (0.8, {"id": "price-2", "category": "price"}),
    ]

    picked = _diversify_categories("giá vé", rows, 2)
    ids = [row["id"] for _, row in picked]

    assert ids == ["price-1", "offer-1", "price-2"]


def test_normalize_text_strips_vietnamese_diacritics():
    assert _normalize_text("Cầu Hôn Phú Quốc") == "cau hon phu quoc"


def test_rrf_fuse_deduplicates_and_preserves_confidence_score():
    row = {"id": "same", "title": "Cầu Hôn", "score": 0.2}
    fused = _rrf_fuse(
        [(0.9, {"id": "vector-only", "score": 0.9}, 0.9), (0.2, row, 0.2)],
        [(2.0, row, 0.2)],
        limit=10,
    )

    ids = [item[1]["id"] for item in fused]

    assert ids.count("same") == 1
    assert next(item[1] for item in fused if item[1]["id"] == "same")["score"] == 0.2


@pytest.mark.asyncio
async def test_in_memory_hybrid_search_matches_unaccented_exact_keyword():
    kb = InMemoryKnowledgeBase(EmbeddingProvider(None, "text-embedding-3-small"))
    now = datetime.now(UTC)
    await kb.upsert_chunks(
        [
            IngestedChunk(
                title="Cầu Hôn Phú Quốc",
                section="Điểm tham quan",
                category="experience",
                content="Thông tin về Cầu Hôn gần khu vực Phú Quốc.",
                source_url="https://vinwonders.com/vi/wonderpedia/news/cau-hon-phu-quoc/",
                content_hash="cau-hon",
                crawled_at=now,
                valid_until=None,
            ),
            IngestedChunk(
                title="Ẩm thực Nha Trang",
                section="Combo",
                category="offer",
                content="Ưu đãi ẩm thực tại Nha Trang.",
                source_url="https://vinwonders.com/vi/uu-dai/nha-trang/",
                content_hash="nha-trang",
                crawled_at=now,
                valid_until=None,
            ),
        ]
    )

    hits = await kb.search("cau hon", limit=2)

    assert hits[0].id == "cau-hon"
