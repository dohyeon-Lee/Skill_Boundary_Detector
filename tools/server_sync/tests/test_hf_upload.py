"""Manual checkpoint upload uses completed steps and an in-memory Hub only."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hf_server import HFBackend
from hf_upload import configure_http_upload, select_checkpoints, upload_selected


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

    def test_eval_upload_uses_original_public_path_without_training_state(self):
        selected = select_checkpoints(self.project, [self.first], pretrained_only=True)
        self.assertEqual(selected[0]['file_count'], 1)
        upload_selected(self.project, 'me/checkpoints', [self.first], api=self.api, pretrained_only=True)
        upload = self.api.uploaded[0]
        self.assertEqual(upload['repo_type'], 'model')
        self.assertEqual(upload['path_in_repo'], self.first + '/pretrained_model')
        self.assertEqual(upload['folder_path'], self.project / self.first / 'pretrained_model')
        self.assertFalse((upload['folder_path'] / 'training_state').exists())
        self.assertTrue((self.project / self.first / 'training_state/training_step.json').is_file())

    def test_eval_upload_option_survives_desk_plan_and_command(self):
        item = self.backend.plan('local-checkpoints', 'hub-checkpoints', [self.first],
            {'pretrained_only': True}, operation='checkpoints-push')['items'][0]
        self.assertEqual(item['file_count'], 1)
        command = self.backend.commands(item)
        self.assertIn('--pretrained-only', command)
        self.assertNotIn('--watch', command)
        with self.assertRaises(ValueError):
            self.backend.plan('local-checkpoints', 'hub-checkpoints', [self.first],
                {'pretrained_only': 'true'}, operation='checkpoints-push')

    def test_eval_cli_selects_http_before_sdk_import_and_respects_explicit_override(self):
        uploader = Path(__file__).resolve().parents[1] / 'hf_upload.py'
        script = '''import runpy,sys
from pathlib import Path
path=Path(sys.argv[1]); expected=sys.argv[2]=='1'
sys.path.insert(0,str(path.parent));sys.argv=[str(path),'--pretrained-only','--help']
try:runpy.run_path(str(path),run_name='__main__')
except SystemExit as error:assert error.code==0
from huggingface_hub import constants
assert constants.HF_HUB_DISABLE_XET is expected
'''
        for override, expected in ((None, '1'), ('0', '0')):
            env = dict(os.environ)
            if override is None:
                env.pop('HF_HUB_DISABLE_XET', None)
            else:
                env['HF_HUB_DISABLE_XET'] = override
            result = subprocess.run([sys.executable, '-c', script, str(uploader), expected],
                env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_http_upload_has_finite_inactivity_timeouts(self):
        from huggingface_hub import set_client_factory
        from huggingface_hub.utils._http import default_client_factory, get_session
        try:
            configure_http_upload()
            timeout = get_session().timeout
            self.assertEqual((timeout.connect, timeout.read, timeout.write, timeout.pool), (20, 120, 120, 20))
        finally:
            set_client_factory(default_client_factory)

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
