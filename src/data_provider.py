import requests


class DataProvider:

    def get_quotes(self):
        resp = requests.get("https://api.kraken.com/0/public/Ticker")
        resp.raise_for_status()
        resp = resp.json()
        pairs = resp["result"]
        usd_paires = {k: v for k, v in pairs.items() if k[-3:] == "USD"}
        return usd_paires
