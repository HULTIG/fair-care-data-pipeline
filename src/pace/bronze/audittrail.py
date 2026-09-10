import json
import os
from datetime import datetime

class AuditTrail:
    def __init__(self, log_dir: str = "results/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "audit_log.json")

    def log_event(self, event_type: str, details: dict):
        """
        Logs an event to the audit trail.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
            
        print(f"Logged event: {event_type}")

    def verify_provenance(self) -> bool:
        """
        Verifies that provenance data (audit trail) is accessible and has entries.
        """
        return os.path.exists(self.log_file) and os.path.getsize(self.log_file) > 0

    def log_failure(self, run_id: str, stage: str, error: Exception):
        """Persist a machine-readable failure record for a run."""
        self.log_event("RUN_FAILURE", {
            "run_id": run_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "error": str(error),
        })
