"""Tests unitaires pour Step 2 - Product detection."""

from __future__ import annotations

import pytest

from ginjer_exercice.pipeline import step2_products
from ginjer_exercice.observability.runtime_warnings import collect_runtime_warnings
from ginjer_exercice.schemas.ad import Ad, AdText, Brand
from ginjer_exercice.schemas.products import Color, DetectedProduct
from ginjer_exercice.schemas.step_outputs import (
    DetectedProductList,
    DetectedProductLLM,
    UniverseDetection,
    UniverseResult,
)

from .pipeline_fakes import (
    FakeLLMProvider,
    FakeMediaFetcher,
    FakePromptRegistry,
    FakeTraceSpan,
    fake_image_content,
    fake_video_content,
)


@pytest.fixture
def chanel_ad() -> Ad:
    return Ad(
        platform_ad_id="test-ad-001",
        brand=Brand.CHANEL,
        texts=[AdText(title="CHANEL N5", body_text="The legend.")],
        media_urls=["https://example.com/n5.jpg"],
    )


@pytest.fixture
def image_fetcher() -> FakeMediaFetcher:
    return FakeMediaFetcher({
        "https://example.com/n5.jpg": fake_image_content("https://example.com/n5.jpg"),
    })


@pytest.fixture
def universe_result() -> UniverseResult:
    return UniverseResult(
        detected_universes=[
            UniverseDetection(universe="Fragrance", confidence=0.95, reasoning="Perfume bottle visible."),
        ]
    )


@pytest.fixture
def single_product_list() -> DetectedProductList:
    return DetectedProductList(
        products=[
            DetectedProductLLM(
                raw_description="Transparent glass bottle labelled N5",
                universe="Fragrance",
                color="Gold",
                importance=5,
            )
        ],
        overall_confidence=0.95,
    )


@pytest.fixture
def multi_product_list() -> DetectedProductList:
    return DetectedProductList(
        products=[
            DetectedProductLLM(
                raw_description="Main perfume bottle",
                universe="Fragrance",
                color="Gold",
                importance=5,
            ),
            DetectedProductLLM(
                raw_description="Background accessory",
                universe="Women",
                color="Black",
                importance=2,
            ),
        ],
        overall_confidence=0.85,
    )


@pytest.fixture
def zero_importance_list() -> DetectedProductList:
    return DetectedProductList(
        products=[
            DetectedProductLLM(
                raw_description="Barely visible item in background",
                universe="Women",
                color="Beige",
                importance=0,
            )
        ],
        overall_confidence=0.3,
    )


class TestStep2Execute:
    def test_returns_list_of_detected_products(
        self, chanel_ad, universe_result, single_product_list, image_fetcher
    ) -> None:
        result = step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=FakeLLMProvider([single_product_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
            media_fetcher=image_fetcher,
        )
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], DetectedProduct)

    def test_product_fields_correctly_mapped(
        self, chanel_ad, universe_result, single_product_list, image_fetcher
    ) -> None:
        result = step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=FakeLLMProvider([single_product_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
            media_fetcher=image_fetcher,
        )
        product = result[0]
        assert product.raw_description == "Transparent glass bottle labelled N5"
        assert product.universe == "Fragrance"
        assert product.color == Color.GOLD
        assert product.importance == 5

    def test_importance_zero_filtered(
        self, chanel_ad, universe_result, zero_importance_list, image_fetcher
    ) -> None:
        result = step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=FakeLLMProvider([zero_importance_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
            media_fetcher=image_fetcher,
        )
        assert result == []

    def test_results_sorted_by_importance_descending(
        self, chanel_ad, universe_result, multi_product_list, image_fetcher
    ) -> None:
        result = step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=FakeLLMProvider([multi_product_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
            media_fetcher=image_fetcher,
        )
        assert [product.importance for product in result] == [5, 2]

    def test_empty_product_list_returns_empty(
        self, chanel_ad, universe_result, image_fetcher
    ) -> None:
        result = step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=FakeLLMProvider([DetectedProductList(products=[], overall_confidence=0.9)]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
            media_fetcher=image_fetcher,
        )
        assert result == []

    def test_unknown_color_falls_back_to_multicolor(
        self, chanel_ad, universe_result, image_fetcher
    ) -> None:
        weird_color_list = DetectedProductList(
            products=[
                DetectedProductLLM(
                    raw_description="A product",
                    universe="Women",
                    color="Chartreuse",
                    importance=3,
                )
            ],
            overall_confidence=0.8,
        )
        result = step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=FakeLLMProvider([weird_color_list]),
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
            media_fetcher=image_fetcher,
        )
        assert result[0].color == Color.MULTICOLOR

    def test_prompt_fetched_by_name(
        self, chanel_ad, universe_result, single_product_list, image_fetcher
    ) -> None:
        fake_registry = FakePromptRegistry()
        step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=FakeLLMProvider([single_product_list]),
            prompt_registry=fake_registry,
            trace=FakeTraceSpan(),
            media_fetcher=image_fetcher,
        )
        assert "pipeline/products" in fake_registry.gets

    def test_downloads_and_injects_images(self, chanel_ad, universe_result, single_product_list) -> None:
        fake_llm = FakeLLMProvider([single_product_list])
        fetcher = FakeMediaFetcher({
            "https://example.com/n5.jpg": fake_image_content("https://example.com/n5.jpg", content=b"jpeg-bytes"),
        })

        step2_products.execute(
            chanel_ad,
            universe_result,
            llm_provider=fake_llm,
            prompt_registry=FakePromptRegistry(),
            trace=FakeTraceSpan(),
            media_fetcher=fetcher,
        )

        assert fetcher.downloaded_urls == ["https://example.com/n5.jpg"]
        sent_parts = fake_llm.calls[0]["messages"][0].parts
        assert sent_parts[0].type == "text"
        assert sent_parts[1].type == "media"
        assert sent_parts[1].media == b"jpeg-bytes"

    def test_videos_are_ignored_in_p0(self, universe_result, single_product_list) -> None:
        ad = Ad(
            platform_ad_id="test-ad-001",
            brand=Brand.CHANEL,
            texts=[AdText(title="CHANEL N5", body_text="The legend.")],
            media_urls=["https://example.com/video.mp4"],
        )
        fake_llm = FakeLLMProvider([single_product_list])
        fetcher = FakeMediaFetcher({
            "https://example.com/video.mp4": fake_video_content("https://example.com/video.mp4"),
        })

        with collect_runtime_warnings() as warnings:
            step2_products.execute(
                ad,
                universe_result,
                llm_provider=fake_llm,
                prompt_registry=FakePromptRegistry(),
                trace=FakeTraceSpan(),
                media_fetcher=fetcher,
            )

        assert fetcher.downloaded_urls == ["https://example.com/video.mp4"]
        assert len(fake_llm.calls[0]["messages"][0].media) == 0
        assert warnings == ["step2: video ignored (not supported in P0)", "step2: fallback to text-only, no usable image"]

    def test_falls_back_to_text_only_if_image_download_fails(
        self, chanel_ad, universe_result, single_product_list
    ) -> None:
        fake_llm = FakeLLMProvider([single_product_list])
        fetcher = FakeMediaFetcher(errors={"https://example.com/n5.jpg": RuntimeError("download failed")})

        with collect_runtime_warnings() as warnings:
            step2_products.execute(
                chanel_ad,
                universe_result,
                llm_provider=fake_llm,
                prompt_registry=FakePromptRegistry(),
                trace=FakeTraceSpan(),
                media_fetcher=fetcher,
            )

        assert fetcher.downloaded_urls == ["https://example.com/n5.jpg"]
        assert len(fake_llm.calls[0]["messages"][0].media) == 0
        assert warnings == ["step2: fallback to text-only, no usable image"]
