import unittest

from scripts.trust4d.audit_official_source import parse_name_status, validate_records


class AuditOfficialSourceTest(unittest.TestCase):
    def test_current_allowed_addition_roots_pass(self):
        records = [
            ("A", "experiments.md"),
            ("A", "server.md"),
            ("A", "scripts/trust4d/run_base001.sh"),
            ("A", "tests/test_audit_official_source.py"),
            ("A", "research/ara/trust4d_teacher_reliability/PAPER.md"),
        ]
        self.assertEqual(validate_records(records), [path for _, path in records])

    def test_modified_official_file_fails(self):
        with self.assertRaisesRegex(ValueError, "not byte-identical"):
            validate_records([("M", "train.py")])

    def test_unapproved_addition_fails(self):
        with self.assertRaisesRegex(ValueError, "unapproved addition"):
            validate_records([("A", "arguments/treatment.py")])

    def test_similar_but_unapproved_prefix_fails(self):
        for path in (
            "scripts/trust4d_evil/file.py",
            "research/ara/another_project/PAPER.md",
        ):
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "unapproved addition"):
                    validate_records([("A", path)])

    def test_parser_rejects_rename_record(self):
        with self.assertRaisesRegex(ValueError, "unexpected"):
            parse_name_status("R100\told.py\tnew.py\n")


if __name__ == "__main__":
    unittest.main()
