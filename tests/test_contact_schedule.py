import unittest

from models.contact_schedule import linear_contact_strength


class ContactScheduleTest(unittest.TestCase):
    def test_zero_schedule_preserves_hard_contact(self):
        self.assertEqual(linear_contact_strength(1, 1000, 0.0, 0.0), 1.0)

    def test_warmup_and_ramp_boundaries(self):
        self.assertEqual(linear_contact_strength(200, 1000, 0.2, 0.6), 0.0)
        self.assertAlmostEqual(
            linear_contact_strength(500, 1000, 0.2, 0.6), 0.5
        )
        self.assertEqual(linear_contact_strength(800, 1000, 0.2, 0.6), 1.0)
        self.assertEqual(linear_contact_strength(1000, 1000, 0.2, 0.6), 1.0)

    def test_invalid_schedule_is_rejected(self):
        invalid = ((-0.1, 0.5), (0.2, -0.1), (0.6, 0.5))
        for warmup, ramp in invalid:
            with self.subTest(warmup=warmup, ramp=ramp):
                with self.assertRaises(ValueError):
                    linear_contact_strength(1, 1000, warmup, ramp)
        with self.assertRaises(ValueError):
            linear_contact_strength(1, 0, 0.2, 0.6)


if __name__ == "__main__":
    unittest.main()
