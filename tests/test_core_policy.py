from __future__ import annotations

import unittest

from intentprobe.scanner.core import choose_runtime_decision


class CorePolicyTests(unittest.TestCase):
    def test_low_activation_only_score_does_not_warn(self) -> None:
        decision, reasons = choose_runtime_decision(
            activation_score=0.43,
            static_risk_score=0.0,
            risk_score=0.43,
            static_findings=[],
            warn_threshold=0.30,
            block_threshold=0.85,
        )

        self.assertEqual("allow", decision)
        self.assertIn("activation-only", reasons[0])

    def test_high_activation_only_score_still_warns(self) -> None:
        decision, reasons = choose_runtime_decision(
            activation_score=0.60,
            static_risk_score=0.0,
            risk_score=0.60,
            static_findings=[],
            warn_threshold=0.30,
            block_threshold=0.85,
        )

        self.assertEqual("warn", decision)
        self.assertIn("warn threshold", reasons[0])


if __name__ == "__main__":
    unittest.main()
