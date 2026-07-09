import pytest
from pydantic import ValidationError

from app.schemas import CrawlRequest


def test_crawl_request_schema_does_not_expose_scope_or_render():
    properties = CrawlRequest.model_json_schema()["properties"]

    assert set(properties) == {"url", "max_depth", "max_pages"}


def test_crawl_request_rejects_dead_scope_render_fields():
    with pytest.raises(ValidationError):
        CrawlRequest(
            url="https://vinwonders.com/vi/vinpearl-safari-phu-quoc/",
            scope="vin_ecosystem",
            render="hybrid",
        )
