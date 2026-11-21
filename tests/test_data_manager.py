import unittest

from flask import Flask

from data.data_manager import DataManager
from data.data_models import db


class DataManagerCompanySearchTests(unittest.TestCase):
    def test_company_search_results_are_cached_without_overwriting_names(self):
        app = Flask(__name__)
        app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite://"
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
        db.init_app(app)

        with app.app_context():
            db.create_all()
            manager = DataManager(db)

            companies = manager.cache_company_search_results(
                [
                    {"symbol": "tsco.lon", "name": "Tesco PLC"},
                    {"symbol": "TSCO.LON", "name": "Duplicate result"},
                ]
            )
            manager.cache_company_search_results(
                [{"symbol": "TSCO.LON", "name": "Updated result"}]
            )

            self.assertEqual(len(companies), 1)
            self.assertEqual(companies[0].ticker_symbol, "TSCO.LON")
            self.assertEqual(
                manager.get_company_by_ticker("TSCO.LON").company_name,
                "Tesco PLC",
            )
            db.session.remove()
            db.drop_all()


if __name__ == "__main__":
    unittest.main()
