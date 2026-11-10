"""Argument contract for the slime-side Harbor bridge."""

from __future__ import annotations

import argparse

from harbor.slime_bridge.args import add_harbor_arguments


def _parse(*argv: str) -> argparse.Namespace:
    return add_harbor_arguments(argparse.ArgumentParser()).parse_args(argv)


def test_plural_service_urls_are_accepted() -> None:
    args = _parse(
        "--harbor-service-urls",
        "http://h20:9500",
        "http://h20-2:9500",
    )

    assert args.harbor_service_urls == [
        "http://h20:9500",
        "http://h20-2:9500",
    ]
    assert args.harbor_service_url is None


def test_legacy_singular_service_url_remains_supported() -> None:
    args = _parse("--harbor-service-url", "http://h20:9500")

    assert args.harbor_service_url == "http://h20:9500"
    assert args.harbor_service_urls is None


def test_service_deadline_has_a_finite_default() -> None:
    args = _parse()

    assert args.harbor_service_deadline_secs > 0
