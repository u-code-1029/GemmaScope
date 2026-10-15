import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from server import Worker, bounded_float
from PIL import Image


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.worker = Worker(self.root / 'project', self.root / 'models')
        self.source = self.root / '원본 이미지'
        self.source.mkdir()
        Image.new('RGB', (32, 32), (60, 70, 80)).save(self.source / '시료.png')

    def tearDown(self):
        self.worker.cancel.set()
        if self.worker.thread:
            self.worker.thread.join(timeout=5)
        self.temp.cleanup()

    def test_async_import_then_label_and_metadata(self):
        events = []
        with patch('server.emit', events.append):
            result = self.worker.handle('import', {'path': str(self.source)})
            self.assertEqual(result['job']['status'], 'running')
            self.worker.thread.join(timeout=5)
        self.assertEqual(self.worker.job['status'], 'completed')
        items = self.worker.handle('images', {})['items']
        self.assertEqual(len(items), 1)
        self.assertEqual(self.worker.handle('label', {'ids': [items[0]['id']], 'label': 'SEM'})['updated'], 1)
        result = self.worker.handle('metadata', {'ids': [items[0]['id']], 'metadata': {'sample': 'S-01'}})
        self.assertEqual(result['updated'], 1)
        self.assertTrue(any(x.get('event', {}).get('job', {}).get('status') == 'completed' for x in events))

    def test_csv_import_only_registered_paths_and_keeps_provenance(self):
        self.worker.store.import_folder(self.source)
        source = self.root / 'metadata.csv'
        with source.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['path', 'sample', 'acquired_at'])
            writer.writerow(['원본 이미지/시료.png', '시료 A', '2026-10-15'])
            writer.writerow(['missing.png', 'not registered', ''])
        result = self.worker.handle('import_metadata', {'path': str(source)})
        self.assertEqual((result['updated'], result['skipped']), (1, 1))
        metadata = self.worker.store.all_records()[0]['metadata']
        self.assertEqual(metadata['sample'], '시료 A')
        self.assertIn('provenance', metadata)

    def test_project_switch_and_edit_rejected_during_job(self):
        self.worker.job = {'status': 'running'}
        with self.assertRaisesRegex(ValueError, '진행 중'):
            self.worker.handle('project', {'path': str(self.root / 'another')})
        with self.assertRaisesRegex(ValueError, '진행 중'):
            self.worker.handle('label', {'ids': [], 'label': 'SEM'})

    def test_invalid_threshold_and_unknown_model_are_rejected(self):
        for value in ['NaN', 'inf', -0.1, 1.1]:
            with self.assertRaises(ValueError):
                bounded_float(value, 0, 1, '통과 기준')
        with self.assertRaisesRegex(ValueError, '학습 모델'):
            self.worker.handle('search', {'run_id': '../another'})

    def test_known_path_cannot_reveal_unregistered_file(self):
        self.worker.store.import_folder(self.source)
        self.assertTrue(self.worker.handle('known_path', {'path': str(self.source / '시료.png')})['known'])
        self.assertFalse(self.worker.handle('known_path', {'path': str(self.root / 'secret')})['known'])

    def test_failed_training_is_persisted_and_not_searchable(self):
        with patch('server.training.train', side_effect=ValueError('독립 시료 그룹 부족')), patch('server.emit'), patch('server.traceback.print_exc'):
            self.worker.handle('train', {'mode': 'classifier'})
            self.worker.thread.join(timeout=5)
        runs = self.worker.store.runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]['status'], 'failed')
        self.assertIn('그룹 부족', runs[0]['error'])
        with self.assertRaisesRegex(ValueError, '완료된'):
            self.worker.run(runs[0]['id'])

    def test_symlink_import_root_rejected_before_resolving(self):
        alias = self.root / 'alias'
        try:
            alias.symlink_to(self.source, target_is_directory=True)
        except OSError:
            self.skipTest('symlink privilege unavailable')
        with self.assertRaises(ValueError):
            self.worker.handle('import', {'path': str(alias)})


if __name__ == '__main__':
    unittest.main()
