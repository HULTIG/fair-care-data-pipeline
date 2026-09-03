class PACEScore:
    def __init__(self, config: dict):
        self.config = config

    def calculate(self, sb: float, ss: float, sg: float) -> dict:
        """
        Calculates the composite PACE Score.
        """
        w_b = self.config.get("weights", {}).get("bronze", 0.33)
        w_s = self.config.get("weights", {}).get("silver", 0.33)
        w_g = self.config.get("weights", {}).get("gold", 0.33)
        
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
