from bs4 import BeautifulSoup

from app.crawling.ingestion import classify_category, seed_chunks, soup_to_chunks


def test_seed_chunks_have_required_metadata():
    chunks = seed_chunks()

    assert chunks
    assert all(chunk.source_url for chunk in chunks)
    assert all(chunk.section for chunk in chunks)
    assert all(chunk.crawled_at for chunk in chunks)


def test_classifies_time_sensitive_content():
    assert classify_category("WONDERPEDIA Nơi mở ra những vùng đất diệu kỳ WonderCulture WonderLand") == "wonderpedia"
    assert classify_category("Giá vé và ưu đãi mới nhất") == "price"
    assert classify_category("Hạn áp dụng voucher giảm 15%") == "offer"
    assert classify_category("Giờ mở cửa hằng ngày 09:00") == "schedule"
    assert classify_category("VinWonders Affiliate hoa hồng hấp dẫn") == "affiliate"
    assert classify_category("Hội viên VinClub hạng Gold Platinum Diamond") == "vinclub"
    assert classify_category("Đặt vé online tại booking.vinwonders.com") == "booking"


def test_soup_to_chunks_extracts_metadata():
    soup = BeautifulSoup(
        """
        <html><head><title>Ưu đãi VinClub</title><meta name="description" content="Giảm 15% cho hội viên VinClub"></head>
        <body><h1>Đặc quyền hội viên VinClub</h1><p>Hạn áp dụng: 01/01 - 31/12/2026. Giảm 15% khi mua vé VinWonders.</p></body></html>
        """,
        "html.parser",
    )

    chunks = soup_to_chunks(
        soup,
        source_url="https://vinwonders.com/vi/uu-dai/vinclub/",
        canonical_url="https://vinwonders.com/vi/uu-dai/vinclub/",
        seed_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        discovered_from=None,
        crawl_depth=1,
        render_mode="http",
        status_code=200,
        crawl_job_id="job-1",
    )

    assert chunks
    assert any(chunk.category == "vinclub" for chunk in chunks)
    assert chunks[0].metadata["crawl_job_id"] == "job-1"
    assert chunks[0].metadata["canonical_url"] == "https://vinwonders.com/vi/uu-dai/vinclub/"
