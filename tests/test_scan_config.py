from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from intentprobe.scanner.configs import (
    ConfigCandidate,
    ConfigServer,
    build_config_scan_payload,
    collect_config_servers,
    inventory_flags,
    product_decision_for_config_scan,
    subject_for_server,
)


class ScanConfigTests(unittest.TestCase):
    def test_config_without_mcp_servers_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "claude_desktop_config.json"
            path.write_text(json.dumps({"preferences": {"theme": "dark"}}))

            configs, servers = collect_config_servers(
                [ConfigCandidate("Claude Desktop", path)],
                max_file_bytes=200_000,
            )

        self.assertEqual([], servers)
        self.assertEqual("no_mcp_servers", configs[0]["status"])
        self.assertEqual(0, configs[0]["server_count"])

    def test_activation_only_manifest_signal_downgrades_to_allow(self) -> None:
        risk = {
            "decision": "warn",
            "activation_score": 0.97,
            "static_score": 0.0,
        }

        decision, reasons = product_decision_for_config_scan(risk, [])

        self.assertEqual("allow", decision)
        self.assertIn("activation-only", reasons[0])

    def test_review_inventory_flag_keeps_config_in_review_tier(self) -> None:
        risk = {
            "decision": "warn",
            "activation_score": 0.97,
            "static_score": 0.0,
        }
        flags = [{"id": "remote-http", "level": "review", "reason": "server connects remotely"}]

        decision, reasons = product_decision_for_config_scan(risk, flags)

        self.assertEqual("warn", decision)
        self.assertIn("review-worthy", reasons[0])

    def test_env_and_remote_server_inventory_flags_do_not_emit_raw_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mcp.json"
            config = {
                "type": "http",
                "url": "https://example.test/mcp",
                "env": {"API_KEY": "secret-value"},
            }
            server = ConfigServer(
                source="custom",
                path=path,
                name="remote",
                config=config,
                subject=subject_for_server("custom", path, "remote", config),
            )
            flags = inventory_flags(server)
            scan_payload = {
                "results": [
                    {
                        "subject": {"content_sha256": "abc123"},
                        "decision": "warn",
                        "risk": {
                            "decision": "warn",
                            "risk_score": 0.9,
                            "activation_score": 0.9,
                            "static_score": 0.0,
                            "risk_reasons": [],
                            "evidence_spans": [],
                        },
                    }
                ],
                "elapsed_seconds": 0.01,
            }
            payload = build_config_scan_payload(
                target=str(path),
                configs=[
                    {
                        "source": "custom",
                        "path": str(path),
                        "exists": True,
                        "status": "scanned",
                        "server_count": 1,
                    }
                ],
                servers=[server],
                scan_payload=scan_payload,
                fail_on="never",
            )

        self.assertIn("env-secrets", {flag["id"] for flag in flags})
        self.assertIn("remote-http", {flag["id"] for flag in flags})
        self.assertEqual("warn", payload["results"][0]["decision"])
        self.assertNotIn("secret-value", json.dumps(payload, sort_keys=True))
        self.assertNotIn("redacted_config", payload["results"][0])


if __name__ == "__main__":
    unittest.main()
