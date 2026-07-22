import pytest

from app.crawling.link_discovery import LinkDiscoveryAgent, classify_url_category, normalize_url, skip_reason


def test_classifies_vin_url_categories():
    assert classify_url_category("https://vinwonders.com/vi/wonderpedia/") == "wonderpedia"
    assert (
        classify_url_category(
            "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate"
        )
        == "affiliate"
    )
    assert classify_url_category("https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub") == "vinclub"
    assert classify_url_category("https://vinwonders.com/vi/promotions/combo-voucher-hot") == "offer"
    assert classify_url_category("https://vinwonders.com/en/vinpearl-safari-phu-quoc-price-and-regulations") == "price"
    assert classify_url_category("https://vinwonders.com/vi/vinpearl-safari-phu-quoc") == "experience"


def test_filters_allowed_and_skipped_urls():
    assert skip_reason("https://vinwonders.com/vi/vinpearl-safari-phu-quoc") is None
    assert skip_reason("https://vinwonders.com/en/vinpearl-safari-phu-quoc") is None
    assert skip_reason("https://vinwonders.com/ko/vinpearl-safari-phu-quoc") == "unsupported_language"
    assert skip_reason("https://vinwonders.com/vi/login") == "auth_page"
    assert skip_reason("https://vinwonders.com/vi/banner.jpg") == "static_asset"
    assert skip_reason("https://example.com/vi/vinpearl-safari-phu-quoc") == "external_domain"


def test_discovers_firecrawl_links_without_static_fallbacks():
    result = {
        "data": [
            {
                "links": [
                    "https://vinwonders.com/vi/promotions/?utm_source=test",
                    "https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/",
                    "https://vinwonders.com/ru/promotions/",
                    "https://example.com/vi/vinpearl-safari-phu-quoc/",
                ],
                "metadata": {
                    "sourceURL": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                    "url": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                },
            }
        ]
    }

    discovery = LinkDiscoveryAgent(
        seed_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
    ).discover_from_firecrawl_result(result)

    selected = {item.url: item for item in discovery.selected_urls}
    assert normalize_url("https://vinwonders.com/vi/vinpearl-safari-phu-quoc/") in selected
    assert normalize_url("https://vinwonders.com/vi/wonderpedia/") in selected
    assert normalize_url("https://vinwonders.com/vi/promotions/?utm_source=test") in selected
    assert (
        normalize_url("https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/")
        in selected
    )
    assert "https://vinwonders.com/ru/promotions" in discovery.skip_reasons
    assert "https://example.com/vi/vinpearl-safari-phu-quoc" in discovery.skip_reasons


def test_discovers_firecrawl_map_urls_as_candidates():
    result = {
        "data": [
            {
                "links": [],
                "metadata": {
                    "sourceURL": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                    "url": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                },
            }
        ]
    }

    discovery = LinkDiscoveryAgent(
        seed_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
    ).discover_from_firecrawl_result(
        result,
        extra_urls=[
            "https://vinwonders.com/vi/wonderpedia/news/voucher-vinpearl-safari-phu-quoc/",
            "https://vinwonders.com/ko/wonderpedia/",
        ],
    )

    selected = {item.url: item for item in discovery.selected_urls}
    voucher_url = "https://vinwonders.com/vi/wonderpedia/news/voucher-vinpearl-safari-phu-quoc"
    assert voucher_url in selected
    assert selected[voucher_url].discovered_from == "https://vinwonders.com/vi/vinpearl-safari-phu-quoc"
    assert "https://vinwonders.com/ko/wonderpedia" in discovery.skip_reasons


@pytest.mark.asyncio
async def test_agent_scrapes_selected_links_and_expands_from_scraped_page():
    result = {
        "data": [
            {
                "links": [
                    "https://vinwonders.com/vi/promotions/",
                ],
                "metadata": {
                    "sourceURL": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                    "url": "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
                },
            }
        ]
    }
    scraper = FakeFirecrawlScraper(
        {
            "https://vinwonders.com/vi/wonderpedia": {
                "data": {
                    "markdown": "# WONDERPEDIA\nNơi mở ra những vùng đất diệu kỳ.",
                    "links": [],
                    "metadata": {
                        "sourceURL": "https://vinwonders.com/vi/wonderpedia/",
                        "url": "https://vinwonders.com/vi/wonderpedia/",
                    },
                }
            },
            "https://vinwonders.com/vi/promotions": {
                "data": {
                    "markdown": "# Promotions\nƯu đãi Safari.",
                    "links": ["https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/"],
                    "metadata": {
                        "sourceURL": "https://vinwonders.com/vi/promotions/",
                        "url": "https://vinwonders.com/vi/promotions/",
                    },
                }
            },
            "https://vinwonders.com/vi/uu-dai": {
                "data": {
                    "markdown": "# Ưu đãi\nCác chương trình ưu đãi VinWonders.",
                    "links": [],
                    "metadata": {
                        "sourceURL": "https://vinwonders.com/vi/uu-dai/",
                        "url": "https://vinwonders.com/vi/uu-dai/",
                    },
                }
            },
            "https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub": {
                "data": {
                    "markdown": "# VinClub\nƯu đãi hội viên.",
                    "links": [],
                    "metadata": {
                        "sourceURL": "https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/",
                        "url": "https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/",
                    },
                }
            },
        }
    )

    discovery, scraped_pages = await LinkDiscoveryAgent(
        seed_url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
    ).discover_and_scrape(result, scraper, max_scrapes=4)

    assert scraper.scraped_urls == [
        "https://vinwonders.com/vi/wonderpedia",
        "https://vinwonders.com/vi/promotions",
        "https://vinwonders.com/vi/uu-dai",
        "https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub",
    ]
    assert [page.discovered_url.url for page in scraped_pages] == scraper.scraped_urls
    assert discovery.scraped_urls == scraper.scraped_urls
    assert (
        discovery.skip_reasons.get("https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub")
        is None
    )
    assert {
        item.url: item.discovery_source for item in discovery.selected_urls
    }["https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub"] == "agent_scrape_links"


class FakeFirecrawlScraper:
    def __init__(self, responses):
        self.responses = responses
        self.scraped_urls = []

    async def scrape(self, url):
        self.scraped_urls.append(url)
        return self.responses[url]
