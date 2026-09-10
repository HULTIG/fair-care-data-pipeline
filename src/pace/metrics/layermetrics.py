class BronzeMetrics:
    def calculate(self, metadata: dict) -> float:
        # SB = w1*Provenance + w2*PII + w3*Quality
        provenance_score = 1.0 if metadata.get("provenance_complete") else 0.5
        pii_score = 1.0 if not metadata.get("pii_found") else 0.0
        quality_score = metadata.get("quality_score", 0.0)
        
        return (provenance_score + pii_score + quality_score) / 3

class SilverMetrics:
    def calculate(self, metadata: dict) -> float:
        # SS = w1*Anonymization + w2*Causal
        # Removed predictive utility from silver to avoid double counting with Gold layer.
        
        technique = metadata.get("technique", "kanonymity")
        epsilon = metadata.get("epsilon")
        k = metadata.get("k")
        
        if technique in {"numeric_noise", "differentialprivacy"} and epsilon is not None and epsilon > 0 and epsilon != float('inf'):
            # Differential Privacy semantics: lower epsilon is better privacy
            anon_score = max(0.0, 1.0 - (epsilon / 10.0))
        elif technique in {"kanonymity", "ldiversity", "tcloseness"} and k is not None and k > 0:
            # K-Anonymity semantics: higher k is better privacy
            anon_score = min(1.0, k / 10.0)
        else:
            # Fallback to structural risk
            anon_score = max(0.0, 1.0 - metadata.get("risk", 1.0))
            
        causal_score = 1.0 if metadata.get("causal_validity") == "PASS" else 0.5
        
        return (anon_score + causal_score) / 2.0

class GoldMetrics:
    def calculate(self, metadata: dict) -> float:
        # SG = (Fairness + ModelUtility) / 2.0
        import math
        spd = metadata.get("statistical_parity_difference")
        
        # Handle NaN, None, or missing SPD
        if spd is None or (isinstance(spd, float) and math.isnan(spd)):
            raise ValueError("fairness score is unavailable")
        else:
            # Good fairness if SPD is close to 0. Heavily penalize structural bias.
            fairness_score = 1.0 if abs(spd) <= 0.1 else max(0.0, 1.0 - 2 * abs(spd))
        
        # Model utility is now only evaluated here to prevent double-counting
        model_utility = metadata.get("model_utility")
        if model_utility is None or not isinstance(model_utility, (int, float)) or not math.isfinite(model_utility):
            raise ValueError("model utility is unavailable")
        
        return (fairness_score + model_utility) / 2.0
