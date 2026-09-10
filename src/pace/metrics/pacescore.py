import math

class PACEScore:
    def __init__(self, config: dict):
        self.config = config

    def calculate(self, sb: float, ss: float, sg: float) -> dict:
        """
        Calculates the composite PACE Score.
        """
        w_b = self.config.get("weights", {}).get("bronze", 0.3333)
        w_s = self.config.get("weights", {}).get("silver", 0.3333)
        w_g = self.config.get("weights", {}).get("gold", 0.3334)
        
        for name, value in (("bronze", sb), ("silver", ss), ("gold", sg)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {name} score: {value!r}")
        weights = (w_b, w_s, w_g)
        if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Weights must be finite and nonnegative")
        if not math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Weights must sum to one")
        score = (w_b * sb) + (w_s * ss) + (w_g * sg)
        
        status = "AT RISK"
        if score >= 0.85:
            status = "EXCELLENT"
        elif score >= 0.70:
            status = "ACCEPTABLE"
            
        return {
            "score": score,
            "status": status,
            "components": {
                "bronze": sb,
                "silver": ss,
                "gold": sg
            }
        }
