"""Deployment defaults must match the specific evaluated checkpoint."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from runtime_config import ROOT, confidence, image_size


class RuntimeDefaults(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='runtime-check-', dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root/'fixture.pt'
        self.weights.write_bytes(b'fixture, not a model')
        self.digest = hashlib.sha256(self.weights.read_bytes()).hexdigest()

    def metrics(self, threshold, digest=None, public=False):
        path = self.root/('reports/submission_evaluation/metrics.json' if public else 'reports/evaluation/metrics.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(weights_sha256=digest or self.digest,
                                       selected_confidence=threshold)),encoding='utf-8')

    def test_matching_public_metrics_determine_threshold(self):
        self.metrics(0.32, public=True)
        self.metrics(0.18)
        self.assertEqual(confidence(self.weights,None,self.root),0.32)

    def test_other_public_weights_do_not_override_local_threshold(self):
        self.metrics(0.32,digest='0'*64,public=True)
        self.metrics(0.18)
        self.assertEqual(confidence(self.weights,None,self.root),0.18)

    def test_explicit_threshold_overrides_reference(self):
        self.metrics(0.32)
        self.assertEqual(confidence(self.weights,0.6,self.root),0.6)

    def test_unknown_weights_use_default(self):
        self.metrics(0.32,digest='0'*64)
        self.assertEqual(confidence(self.weights,None,self.root),0.25)

    def test_invalid_threshold_is_rejected(self):
        for value in (True,0,float('nan'),1.01):
            with self.assertRaises(ValueError):
                confidence(self.weights,value,self.root)
        self.metrics(0)
        with self.assertRaises(ValueError):
            confidence(self.weights,None,self.root)

    def test_selected_resolution_and_stride(self):
        (self.root/'selection.json').write_text(json.dumps(dict(weights_sha256=self.digest,
            winner=dict(config=dict(imgsz=768)))),encoding='utf-8')
        self.assertEqual(image_size(self.weights,None,self.root),768)
        self.assertEqual(image_size(self.weights,640,self.root),640)
        with self.assertRaises(ValueError):
            image_size(self.weights,321,self.root)

    def test_public_metrics_resolution_when_selection_has_other_weights(self):
        (self.root/'selection.json').write_text(json.dumps(dict(weights_sha256='0'*64,
            winner=dict(config=dict(imgsz=640)))),encoding='utf-8')
        path = self.root/'reports/submission_evaluation/metrics.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(weights_sha256=self.digest,imgsz=768)),encoding='utf-8')
        self.assertEqual(image_size(self.weights,None,self.root),768)

    def test_unknown_resolution_uses_640(self):
        self.assertEqual(image_size(self.weights,None,self.root),640)

    def test_evaluated_resolution_takes_precedence_over_training_resolution(self):
        (self.root/'selection.json').write_text(json.dumps(dict(weights_sha256=self.digest,
            winner=dict(config=dict(imgsz=640)))),encoding='utf-8')
        path=self.root/'reports/submission_evaluation/metrics.json';path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(weights_sha256=self.digest,imgsz=960)),encoding='utf-8')
        self.assertEqual(image_size(self.weights,None,self.root),960)


if __name__ == '__main__':
    unittest.main()
