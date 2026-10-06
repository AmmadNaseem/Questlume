"""Headless Streamlit interaction tests; no LLM, search or embedding downloads."""

import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.core.config import PROJECT_ROOT, Settings
from src.export.pptx_renderer import PresentationRenderError
from src.schemas.interview import InterviewResult
from src.schemas.presentation import PresentationPlan
from src.services.generation import GenerationResult


class UITests(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {}, clear=True):
            self.settings = Settings(_env_file=None, llm_provider='groq', llm_fallback_providers=(),
                groq_model='fake', groq_api_key='fake')
        source = dict(kind='document', source_id='s1', document_id='d1', chunk_id='c1',
                      filename='notes.pdf', page=1, excerpt='Annotations describe types.')
        self.interview = InterviewResult.model_validate(dict(topic='Types', source_mode='document', sources=[source],
            items=[dict(question='What are type hints?', category='Python', difficulty='beginner',
                question_type='conceptual', answer='They describe expected types.',
                key_points=['They describe types.'], source_ids=['s1'], confidence='high')]))
        self.plan = PresentationPlan.model_validate(dict(topic='Types', day=15, source_mode='document',
            template_id='reference', sources=[source], slides=[dict(number=1, layout='cover',
            purpose='Introduce', title='Types')]+[dict(number=i, layout='concept', purpose='Teach',
            title=f'Types {i}', bullets=['Annotations describe types.'], source_ids=['s1']) for i in range(2,11)]))

    def app(self):
        return AppTest.from_file(str(PROJECT_ROOT / 'streamlit_app.py')).run()

    def test_initial_page_and_all_four_forms(self):
        app = self.app()
        self.assertFalse(app.exception)
        for source in ('Uploaded PDFs', 'Online research'):
            for output in ('Interview questions', 'Presentation'):
                app.radio[0].set_value(source)
                app.radio[1].set_value(output).run()
                self.assertFalse(app.exception)
                labels = [field.label for field in app.text_input]
                self.assertIn('Topic', labels)
                self.assertIn('Audience' if output == 'Presentation' else 'Job role', labels)

    def test_blank_topic_does_not_call_generation(self):
        with patch('src.ui.app.get_settings', return_value=self.settings), patch('src.ui.app.generate') as generate:
            app = self.app()
            app.button[0].click().run()
            generate.assert_not_called()
            self.assertTrue(app.error)
            self.assertIn('Topic', app.error[0].value)
            self.assertFalse(app.exception)

    def test_missing_model_shows_safe_configuration_field(self):
        from pydantic import ValidationError
        with patch.dict(os.environ, {}, clear=True):
            try:
                Settings(_env_file=None, llm_provider='groq', llm_fallback_providers=(),
                         groq_model='', groq_api_key='secret-key-do-not-display')
            except ValidationError as error:
                failure = error
        with patch('src.ui.app.get_settings', side_effect=failure), patch('src.ui.app.generate') as generate:
            app = self.app()
            app.radio[0].set_value('Online research').run()
            app.text_input[0].set_value('Types')
            app.button[0].click().run()
            self.assertIn('GROQ_MODEL', app.error[0].value)
            self.assertNotIn('secret-key-do-not-display', app.error[0].value)
            generate.assert_not_called()

    def test_success_rerun_and_failed_new_request(self):
        with patch('src.ui.app.get_settings', return_value=self.settings), \
             patch('src.ui.app.generate', return_value=GenerationResult(self.interview, ())) as generate:
            app = self.app()
            app.radio[0].set_value('Online research').run()
            app.text_input[0].set_value('Types')
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generate.call_count, 1)
            self.assertIn('generation_result', app.session_state)
            app.radio[1].set_value('Presentation').run()
            self.assertEqual(generate.call_count, 1)
            generate.side_effect = RuntimeError('secret-provider-message')
            app.button[0].click().run()
            self.assertTrue(app.error)
            self.assertNotIn('secret-provider-message', app.error[0].value)
            self.assertNotIn('generation_result', app.session_state)

    def test_failed_render_keeps_plan_and_success_keeps_download_bytes(self):
        with patch('src.ui.app.get_settings', return_value=self.settings), \
             patch('src.ui.app.generate', return_value=GenerationResult(self.plan, ())), \
             patch('src.ui.app.presentation_bytes', side_effect=PresentationRenderError('runtime')) as render:
            app = self.app()
            app.radio[0].set_value('Online research').run()
            app.radio[1].set_value('Presentation').run()
            app.text_input[0].set_value('Types')
            app.button[0].click().run()
            next(button for button in app.button if button.label == 'Create PowerPoint').click().run()
            self.assertTrue(app.error)
            self.assertIn('generation_result', app.session_state)
            render.side_effect = None
            render.return_value = b'PPTX-fixture'
            next(button for button in app.button if button.label == 'Create PowerPoint').click().run()
            self.assertEqual(app.session_state['pptx_bytes'], b'PPTX-fixture')
            self.assertFalse(app.exception)

    def test_missing_upload_is_actionable_and_stops_before_configuration(self):
        with patch('src.ui.app.get_settings') as settings:
            app = self.app()
            app.text_input[0].set_value('Types')
            app.button[0].click().run()
            self.assertIn('Upload at least one PDF', app.error[0].value)
            settings.assert_not_called()

    def test_question_limit_shows_actual_maximum(self):
        with patch('src.ui.app.get_settings', return_value=self.settings), patch('src.ui.app.generate') as generate:
            app = self.app()
            app.radio[0].set_value('Online research').run()
            app.text_input[0].set_value('Types')
            app.number_input[0].set_value(self.settings.interview_max_questions + 1)
            app.button[0].click().run()
            self.assertIn(f'up to {self.settings.interview_max_questions}', app.error[0].value)
            generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
