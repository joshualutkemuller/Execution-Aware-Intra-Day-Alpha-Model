from dataclasses import dataclass

@dataclass
class MarketAtSignal:
    def decide(self, state: dict) -> dict:
        # TODO: policy logic; placeholder always market
        return {"side": state.get("side","buy"), "type": "market", "improve_ticks": 0}
