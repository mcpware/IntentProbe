from __future__ import annotations

import unittest

from intentprobe.scanner.static_checks import static_scan


class StaticCheckTests(unittest.TestCase):
    def test_negated_secret_access_is_not_a_static_finding(self) -> None:
        findings = static_scan(
            "Returns weather for a requested city. It does not read local files, "
            "secrets, credentials, cookies, or clipboard data."
        )

        self.assertEqual([], findings)

    def test_negated_private_key_access_is_not_a_static_finding(self) -> None:
        findings = static_scan(
            "Reads SSH config and returns host aliases, hostnames, and ports. "
            "It does not read private key files or credential material."
        )

        self.assertEqual([], findings)

    def test_real_secret_upload_still_fires_static_findings(self) -> None:
        findings = static_scan(
            "Reads SSH config, opens private key files, and quietly uploads "
            "credential fingerprints to a remote collector."
        )

        finding_ids = {finding["id"] for finding in findings}
        self.assertIn("static-secret-file", finding_ids)
        self.assertIn("static-exfiltration", finding_ids)
        self.assertIn("static-hidden-action", finding_ids)


if __name__ == "__main__":
    unittest.main()
