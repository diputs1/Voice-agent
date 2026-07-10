import httpx
import pytest

from app.crawling.firecrawl import FirecrawlClient, firecrawl_result_to_chunks, scrape_result_to_chunks


@pytest.mark.asyncio
async def test_firecrawl_client_starts_and_reads_crawl():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer fc-test"
        if request.url.path == "/v2/crawl" and request.method == "POST":
            payload = request.read().decode()
            assert '"crawlEntireDomain":true' in payload.replace(" ", "")
            return httpx.Response(200, json={"success": True, "id": "crawl-1", "url": "https://example.com"})
        if request.url.path == "/v2/crawl/crawl-1" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "status": "completed",
                    "total": 1,
                    "completed": 1,
                    "creditsUsed": 1,
                    "data": [
                        {
                            "markdown": "# VinClub\nƯu đãi lên tới 15% cho hội viên VinClub.",
                            "links": ["https://vinwonders.com/vi/uu-dai/"],
                            "metadata": {
                                "title": "Ưu đãi VinClub",
                                "sourceURL": "https://vinwonders.com/vi/uu-dai/vinclub/",
                                "url": "https://vinwonders.com/vi/uu-dai/vinclub/",
                                "statusCode": 200,
                            },
                        }
                    ],
                },
            )
        return httpx.Response(404)

    client = FirecrawlClient("fc-test", transport=httpx.MockTransport(handler))

    started = await client.start_crawl(url="https://vinwonders.com/vi/", max_depth=1, limit=5)
    result = await client.wait_for_crawl(started["id"], poll_interval=0.01)

    chunks = firecrawl_result_to_chunks(
        result=result,
        seed_url="https://vinwonders.com/vi/",
        crawl_job_id="job-1",
    )

    assert result["completed"] == 1
    assert chunks
    assert chunks[0].category == "vinclub"
    assert chunks[0].metadata["source_type"] == "firecrawl"


def test_scrape_result_to_chunks_handles_single_page_response():
    chunks = scrape_result_to_chunks(
        result={
            "success": True,
            "data": {
                "markdown": "# VinWonders Affiliate\nChương trình affiliate có hoa hồng hấp dẫn.",
                "metadata": {
                    "title": "VinWonders Affiliate",
                    "sourceURL": "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
                    "url": "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
                    "statusCode": 200,
                },
            },
        },
        seed_url="https://vinwonders.com/vi/",
        crawl_job_id="job-2",
    )

    assert chunks[0].category == "affiliate"
    assert "firecrawl_metadata" in chunks[0].metadata


def test_firecrawl_chunks_strip_image_and_related_noise():
    chunks = firecrawl_result_to_chunks(
        result={
            "data": [
                {
                    "markdown": "\n".join(
                        [
                            "# Giá vé Safari",
                            "Giới thiệu](https://vinwonders.com/demo#session-1) ![",
                            "Mã ưu đãi](https://vinwonders.com/demo#session-2) ![",
                            "Giới thiệu](https://vinwonders.com/demo#session-1) \\![ Mã ưu đãi](https://vinwonders.com/demo#session-2)",
                            "[![Ảnh vé](data:image/svg+xml,%3Csvg%3E)](https://vinwonders.com/vi/uu-dai/demo/)",
                            "[![[Onsite service] - Safari Ultimate Animal Tour](http://booking-static.vinpearl.com/tours/demo.jpg)\\ \\ [Onsite service] Safari Ultimate Animal Tour\\ \\ 14 USD](https://booking.vinwonders.com/en-USD/cart?ticketCode=demo)",
                            "Giá vé Vinpearl Safari Phú Quốc từ 650.000 VND đến 850.000 VND.",
                            "## Bài viết liên quan",
                            "[![Bài khác](data:image/svg+xml,%3Csvg%3E)](https://vinwonders.com/vi/wonderpedia/news/noise/)",
                            "Nội dung rác không nên vào knowledge base.",
                        ]
                    ),
                    "metadata": {
                        "title": "Giá vé Safari",
                        "sourceURL": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc-gia-ve-va-quy-dinh/",
                        "url": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc-gia-ve-va-quy-dinh/",
                        "statusCode": 200,
                    },
                }
            ]
        },
        seed_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        crawl_job_id="job-clean",
    )

    assert chunks
    content = "\n".join(chunk.content for chunk in chunks)
    assert "650.000 VND" in content
    assert "data:image/svg+xml" not in content
    assert "booking-static" not in content
    assert "![" not in content
    assert "Nội dung rác" not in content


def test_firecrawl_section_falls_back_to_title_for_body_fragments():
    chunks = firecrawl_result_to_chunks(
        result={
            "data": [
                {
                    "markdown": "ở hữu website | Trang web cung cấp nội dung phù hợp\nNội dung chính của trang.",
                    "metadata": {
                        "title": "VinWonders Affiliate",
                        "sourceURL": "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
                        "url": "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
                        "statusCode": 200,
                    },
                }
            ]
        },
        seed_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        crawl_job_id="job-section",
    )

    assert chunks[0].section == "VinWonders Affiliate"


def test_firecrawl_chunks_include_discovery_metadata():
    chunks = firecrawl_result_to_chunks(
        result={
            "data": [
                {
                    "markdown": "# VinClub\nƯu đãi hội viên VinClub.",
                    "metadata": {
                        "title": "VinClub",
                        "sourceURL": "https://vinwonders.com/vi/uu-dai/vinclub/",
                        "url": "https://vinwonders.com/vi/uu-dai/vinclub/",
                        "statusCode": 200,
                    },
                }
            ]
        },
        seed_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
        crawl_job_id="job-discovery",
        url_metadata={
            "https://vinwonders.com/vi/uu-dai/vinclub": {
                "discovery_source": "firecrawl_links",
                "url_category": "vinclub",
                "discovered_from": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc",
            }
        },
    )

    assert chunks[0].metadata["discovery_source"] == "firecrawl_links"
    assert chunks[0].metadata["url_category"] == "vinclub"
