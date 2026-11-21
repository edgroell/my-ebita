import unittest
from unittest.mock import Mock, patch

from services_api.alphavantage_service import AlphaVantageService


class AlphaVantageServiceTests(unittest.TestCase):
    def test_symbol_search_normalizes_valid_exchange_symbols(self):
        response = Mock()
        response.json.return_value = {
            "bestMatches": [
                {"1. symbol": "tsco.lon", "2. name": "Tesco PLC"},
                {"1. symbol": "INVALID"},
                "unexpected",
            ]
        }

        with (
            patch(
                "services_api.alphavantage_service.time.monotonic",
                return_value=10.0,
            ),
            patch(
                "services_api.alphavantage_service.requests.get",
                return_value=response,
            ) as get,
        ):
            service = AlphaVantageService("test-key")
            results = service.search_symbols("Tesco")

        self.assertEqual(
            results,
            [{"symbol": "TSCO.LON", "name": "Tesco PLC"}],
        )
        self.assertEqual(get.call_args.kwargs["params"]["function"], "SYMBOL_SEARCH")
        self.assertEqual(get.call_args.kwargs["params"]["keywords"], "Tesco")

    def test_requests_are_spaced_by_minimum_interval(self):
        response = Mock()
        response.json.return_value = {}

        with (
            patch(
                "services_api.alphavantage_service.time.monotonic",
                side_effect=[10.0, 10.0, 10.2, 11.1],
            ),
            patch("services_api.alphavantage_service.time.sleep") as sleep,
            patch(
                "services_api.alphavantage_service.requests.get",
                return_value=response,
            ) as get,
        ):
            service = AlphaVantageService("test-key")
            service._request_json({"function": "OVERVIEW"})
            service._request_json({"function": "EARNINGS_CALL_TRANSCRIPT"})

        get.assert_called()
        self.assertEqual(get.call_count, 2)
        sleep.assert_called_once()
        self.assertAlmostEqual(sleep.call_args.args[0], 0.9)


if __name__ == "__main__":
    unittest.main()
