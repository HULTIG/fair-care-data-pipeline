class PACEScore:
    def __init__(self, config: dict):
        self.config = config

    def calculate(self, sb: float, ss: float, sg: float) -> dict:
        """
        Calculates the composite PACE Score.
        """
        import math
        values = {"bronze": sb, "silver": ss, "gold": sg}
        for name, value in values.items():
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} component must be a finite number in [0, 1]")

        weights = self.config.get("weights", {})
        w_b = weights.get("bronze", 0.3333)
        w_s = weights.get("silver", 0.3333)
        w_g = weights.get("gold", 0.3334)
        if any(not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0 for w in (w_b, w_s, w_g)):
            raise ValueError("readiness weights must be finite non-negative numbers")
        if not math.isclose(w_b + w_s + w_g, 1.0, rel_tol=0, abs_tol=1e-6):
            raise ValueError("readiness weights must sum to one")
        
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
