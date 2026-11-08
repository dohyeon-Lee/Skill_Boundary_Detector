"""Manual checkpoint upload uses completed steps and an in-memory Hub only."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hf_server import HFBackend
from hf_upload import select_checkpoints, upload_selected


class UploadHub:
    def __init__(self):
        self.created, self.uploaded = [], []

    def create_repo(self, *args, **kwargs):
        self.created.append((args, kwargs))

    def list_repo_files(self, *args, **kwargs):
        return ['README.md']

    def upload_folder(self, **kwargs):
        self.uploaded.append(kwargs)


class UploadTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='hf-manual-upload-')
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.run = 'outputs_filtered/skillVLA_PT/VSA/example'
        self.first = self.run + '/checkpoints/010000'
        self.second = self.run + '/checkpoints/020000'
        for step in ['010000', '020000', '030000']:
            p = self.project / self.run / 'checkpoints' / step
            (p / 'pretrained_model').mkdir(parents=True)
            (p / 'training_state').mkdir()
            (p / 'pretrained_model/model.safetensors').write_bytes(b'mock weights')
            (p / 'training_state/training_step.json').write_text(json.dumps({'step': int(step)}))
        checkpoints = self.project / self.run / 'checkpoints'
        (checkpoints / 'last').symlink_to('020000')
        (checkpoints / '.hf_pulled').write_text('010000\n')
        (self.project / self.run / 'private.log').write_text('not a checkpoint')
        self.api = UploadHub()
        self.backend = HFBackend({'project_root': str(self.project), 'hf_checkpoint_repo': 'me/checkpoints'}, api=self.api)

    def test_parent_selection_expands_completed_steps_and_deduplicates(self):
        result = select_checkpoints(self.project, [self.run, self.first])
        self.assertEqual([x['path'] for x in result], [self.first, self.second])
        self.assertEqual(result[0]['file_count'], 2)
        # A manual request can deliberately upload a checkpoint downloaded elsewhere.
        self.assertEqual(select_checkpoints(self.project, [self.first])[0]['path'], self.first)

    def test_incomplete_symlink_and_outside_paths_rejected(self):
        for path in [self.run + '/checkpoints/030000', self.run + '/checkpoints/last',
                     '../outside', 'models/pi05_base', self.first + '/pretrained_model']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                select_checkpoints(self.project, [path])
        (self.project / self.first / 'secret').symlink_to(self.project / self.run / 'private.log')
        with self.assertRaisesRegex(ValueError, '링크'):
            select_checkpoints(self.project, [self.first])

    def test_dry_run_and_invalid_selection_never_contact_hub(self):
        api = Mock()
        upload_selected(self.project, 'me/checkpoints', [self.first], dry_run=True, api=api)
        self.assertEqual(api.mock_calls, [])
        with self.assertRaises(ValueError):
            upload_selected(self.project, 'me/checkpoints', [self.run + '/checkpoints/030000'], api=api)
        self.assertEqual(api.mock_calls, [])

    def test_upload_keeps_paths_training_state_and_local_metadata(self):
        before = sorted(str(p) for p in self.project.rglob('*'))
        upload_selected(self.project, 'me/checkpoints', [self.run], api=self.api)
        self.assertEqual([u['path_in_repo'] for u in self.api.uploaded], [self.first, self.second])
        for item in self.api.uploaded:
            self.assertEqual(item['repo_type'], 'model')
            self.assertTrue((item['folder_path'] / 'training_state/training_step.json').is_file())
        self.assertEqual(self.api.created[0][1]['private'], False)
        self.assertEqual(before, sorted(str(p) for p in self.project.rglob('*')))
        self.assertEqual((self.project / self.run / 'checkpoints/last').readlink(), Path('020000'))

    def test_local_browser_and_manual_plan_are_read_only(self):
        root = self.backend.browse('local-checkpoints')
        self.assertEqual([p['name'] for p in root['entries']], ['outputs_filtered'])
        listing = self.backend.browse('local-checkpoints', self.run + '/checkpoints')
        self.assertEqual(next(p for p in listing['entries'] if p['name'] == 'last')['kind'], 'symlink')
        plan = self.backend.plan('local-checkpoints', 'hub-checkpoints', [self.run], operation='checkpoints-push')
        self.assertEqual([p['path'] for p in plan['items']], [self.first, self.second])
        self.assertTrue(all(p['route'] == 'checkpoints-push' for p in plan['items']))
        self.assertEqual(plan['collapsed'], 0)
        self.assertEqual(self.api.created, [])
        command = self.backend.commands(plan['items'][0])
        self.assertTrue(command[2].endswith('hf_upload.py'))
        self.assertEqual(json.loads(command[command.index('--paths-json') + 1]), [self.first])
        shutil.rmtree(self.project / self.first)
        with self.assertRaises(ValueError):
            self.backend.commands(plan['items'][0])

    def test_watch_route_remains_compatible_and_operation_must_match(self):
        old = self.backend.plan('local-checkpoints', 'hub-checkpoints', ['outputs_filtered'])
        self.assertEqual(old['items'][0]['route'], 'watch')
        self.assertIn('watch', self.backend.commands(old['items'][0]))
        with self.assertRaisesRegex(ValueError, '방향'):
            self.backend.plan('local-checkpoints', 'hub-checkpoints', [self.first], operation='checkpoints-pull')


if __name__ == '__main__':
    unittest.main()
