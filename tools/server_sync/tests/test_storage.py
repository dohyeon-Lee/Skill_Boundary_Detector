"""Filesystem capacity is read without walking datasets or querying SSH."""
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from storage import storage_snapshot


class StorageTest(unittest.TestCase):
    def test_same_filesystem_is_reported_once_and_missing_paths_skipped(self):
        with tempfile.TemporaryDirectory() as root:
            first = Path(root) / 'a'
            second = Path(root) / 'b'
            first.mkdir()
            second.mkdir()
            with patch('storage.shutil.disk_usage', wraps=shutil.disk_usage) as usage:
                data = storage_snapshot([('A', first), ('B', second), ('missing', Path(root) / 'missing')])
            self.assertTrue(data['hostname'])
            self.assertEqual(len(data['volumes']), 1)
            self.assertEqual(usage.call_count, 1)
            volume = data['volumes'][0]
            self.assertEqual(volume['name'], 'A')
            self.assertGreater(volume['total'], 0)
            self.assertGreaterEqual(volume['free'], 0)
            self.assertGreaterEqual(volume['used_percent'], 0)

    def test_unavailable_volume_is_reported_without_breaking_response(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('storage.shutil.disk_usage', side_effect=OSError('unavailable')):
                data = storage_snapshot([('volume', root)])
            self.assertIn('error', data['volumes'][0])

    def test_reserved_space_matches_df_available_space_percentage(self):
        with tempfile.TemporaryDirectory() as root:
            usage = SimpleNamespace(total=100, used=80, free=10)
            with patch('storage.shutil.disk_usage', return_value=usage):
                data = storage_snapshot([('volume', root)])
            self.assertEqual(data['volumes'][0]['used_percent'], 88.9)
