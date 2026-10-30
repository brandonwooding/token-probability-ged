import unittest

from metrics import multilingual_metrics


class MetricsTest(unittest.TestCase):
    def test_low_probability_detector_and_micro_pooling(self):
        rows = [("en", .1, 1), ("en", .9, 0),
                ("de", .2, 1), ("de", .8, 0)]
        report = multilingual_metrics(rows, error_when_low=True)
        self.assertEqual(report["pooled_micro"]["tp"], 2)
        self.assertEqual(report["pooled_micro"]["fp"], 0)
        self.assertEqual(report["pooled_micro"]["f0.5"], 1.)

    def test_high_probability_detector_can_predict_every_item(self):
        rows = [("en", .1, 1), ("en", .2, 1)]
        report = multilingual_metrics(rows, error_when_low=False)
        self.assertEqual(report["pooled_micro"]["tp"], 2)
        self.assertEqual(report["pooled_micro"]["f0.5"], 1.)


if __name__ == "__main__":
    unittest.main()
