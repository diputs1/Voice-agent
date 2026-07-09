from app.link_discovery import LinkDiscoveryAgent, classify_url_category, normalize_url, skip_reason


def test_classifies_vin_url_categories():
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


def test_discovers_firecrawl_links_and_falls_back_when_too_few_selected():
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
        fallback_urls=[
            "https://vinwonders.com/vi/vinpearl-safari-phu-quoc-gia-ve-va-quy-dinh/",
            "https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/",
        ],
    ).discover_from_firecrawl_result(result)

    selected = {item.url: item for item in discovery.selected_urls}
    assert normalize_url("https://vinwonders.com/vi/vinpearl-safari-phu-quoc/") in selected
    assert normalize_url("https://vinwonders.com/vi/promotions/?utm_source=test") in selected
    assert (
        normalize_url("https://vinwonders.com/vi/wonderpedia/news/ra-mat-chuong-trinh-vinwonders-affiliate/")
        in selected
    )
    assert normalize_url("https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/") in selected
    assert selected[normalize_url("https://vinwonders.com/vi/uu-dai/vinwonders-uu-dai-15-hoi-vien-vinclub/")].discovery_source == "fallback"
    assert "https://vinwonders.com/ru/promotions" in discovery.skip_reasons
    assert "https://example.com/vi/vinpearl-safari-phu-quoc" in discovery.skip_reasons
