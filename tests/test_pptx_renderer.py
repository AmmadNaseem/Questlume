"""Renderer boundary tests run offline and do not require the Node runtime."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src.core.config import RenderingSettings
from src.export.pptx_renderer import PresentationRenderError, render_pptx
from src.schemas.presentation import PresentationPlan


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        with patch.dict(os.environ, {}, clear=True):
            self.settings = RenderingSettings(_env_file=None, render_template_dir=root,
                presentation_output_dir=root / 'outputs', render_artifact_module=str(root / 'module.mjs'),
                render_skill_dir=str(root / 'skill'))
        (root / 'module.mjs').write_text('')
        helper = root / 'skill/container_tools/artifact_tool_utils.mjs'
        helper.parent.mkdir(parents=True)
        helper.write_text('')
        self.registry = root / 'reference.json'
        self.registry.write_text(json.dumps({'template_file': 'reference.pptx'}))
        with zipfile.ZipFile(root / 'reference.pptx', 'w') as archive:
            archive.writestr('[Content_Types].xml', '<Types/>')
        self.plan = PresentationPlan.model_validate(dict(topic='Types', day=15,
            source_mode='document', template_id='reference',
            sources=[dict(kind='document', source_id='s1', document_id='d1', chunk_id='c1',
                          filename='notes.pdf', page=1, excerpt='Annotations describe types.')],
            slides=[dict(number=1, layout='cover', purpose='Introduce', title='Types')] +
                   [dict(number=i, layout='concept', purpose='Explain', title=f'Types {i}',
                         bullets=['Annotations describe types.'], source_ids=['s1']) for i in range(2,11)]))

    def test_filenames_and_plan_budget(self):
        for filename in ('../escape.pptx', 'bad.txt', 'nested/file.pptx'):
            with self.assertRaises(ValueError):
                render_pptx(self.plan, filename, self.settings)
        with self.assertRaises(ValueError):
            render_pptx(self.plan, 'ok.pptx', self.settings.model_copy(update={'render_max_plan_bytes': 1}))

    def test_unknown_template_and_traversal(self):
        with self.assertRaises(PresentationRenderError):
            render_pptx(self.plan.model_copy(update={'template_id': 'missing'}), 'ok.pptx', self.settings)
        self.registry.write_text(json.dumps({'template_file': '../outside.pptx'}))
        with self.assertRaises(PresentationRenderError):
            render_pptx(self.plan, 'ok.pptx', self.settings)

    def test_invalid_template_and_runtime(self):
        with self.assertRaises(PresentationRenderError):
            render_pptx(self.plan, 'ok.pptx', self.settings.model_copy(update={'render_artifact_module': ''}))
        (self.registry.parent / 'reference.pptx').write_bytes(b'')
        with self.assertRaises(PresentationRenderError):
            render_pptx(self.plan, 'ok.pptx', self.settings)

    def test_existing_file_is_preserved(self):
        output = self.settings.presentation_output_dir / 'existing.pptx'
        output.parent.mkdir()
        output.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            render_pptx(self.plan, output.name, self.settings)
        self.assertEqual(output.read_bytes(), b'preserve')

    def test_failure_and_timeout_publish_nothing(self):
        for result in (subprocess.CompletedProcess([], 1), subprocess.TimeoutExpired('node', 1)):
            with patch('src.export.pptx_renderer.subprocess.run') as run:
                if isinstance(result, Exception): run.side_effect = result
                else: run.return_value = result
                with self.assertRaises(PresentationRenderError):
                    render_pptx(self.plan, 'failed.pptx', self.settings)
                self.assertFalse((self.settings.presentation_output_dir / 'failed.pptx').exists())

    def test_publication_and_safe_subprocess_arguments(self):
        def fake_render(args, **kwargs):
            self.assertFalse(kwargs.get('shell', False))
            self.assertEqual(kwargs['timeout'], self.settings.render_timeout_seconds)
            work = Path(args[-2])
            target = work / 'final/validated.pptx'
            target.parent.mkdir()
            with zipfile.ZipFile(target, 'w') as archive:
                archive.writestr('[Content_Types].xml', '<Types/>')
            return subprocess.CompletedProcess(args, 0)
        with patch('src.export.pptx_renderer.subprocess.run', side_effect=fake_render):
            output = render_pptx(self.plan, 'result.pptx', self.settings)
        self.assertTrue(zipfile.is_zipfile(output))

    def test_no_validated_file_is_rejected(self):
        with patch('src.export.pptx_renderer.subprocess.run', return_value=subprocess.CompletedProcess([], 0)):
            with self.assertRaises(PresentationRenderError):
                render_pptx(self.plan, 'missing.pptx', self.settings)

    def test_failed_copy_removes_partial_output(self):
        def fake_render(args, **kwargs):
            target = Path(args[-2]) / 'final/validated.pptx'
            target.parent.mkdir()
            with zipfile.ZipFile(target, 'w') as archive:
                archive.writestr('[Content_Types].xml', '<Types/>')
            return subprocess.CompletedProcess(args, 0)
        with patch('src.export.pptx_renderer.subprocess.run', side_effect=fake_render), \
             patch('src.export.pptx_renderer.shutil.copyfileobj', side_effect=OSError('Disk full')):
            with self.assertRaises(OSError):
                render_pptx(self.plan, 'partial.pptx', self.settings)
        self.assertFalse((self.settings.presentation_output_dir / 'partial.pptx').exists())


if __name__ == '__main__':
    unittest.main()
