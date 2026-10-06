"""Error explanations distinguish known failures and never echo unknown raw data."""

import unittest
from src.ui.errors import error_message
from src.ingestion.pdf_loader import PDFIngestionError, PDFTextUnavailableError
from src.export.pptx_renderer import PresentationRenderError


class ErrorMessageTests(unittest.TestCase):
    def test_provider_status_codes_have_specific_actions(self):
        for code, action in ((401, 'API key'), (403, 'permissions'), (404, 'model ID'),
                             (429, 'quota'), (413, 'size'), (503, 'service error')):
            error = RuntimeError('private-provider-payload')
            error.status_code = code
            message = error_message(error)
            self.assertIn(action, message)
            self.assertNotIn('private-provider-payload', message)

    def test_specific_pdf_recovery_actions(self):
        cases = [
            (PDFIngestionError('Encrypted PDFs are not supported.'), 'unlocked copy'),
            (PDFIngestionError('Duplicate PDF content is not allowed.'), 'Remove the duplicate'),
            (PDFTextUnavailableError('private filename'), 'OCR'),
            (PDFIngestionError('PDF page count exceeds the configured limit.'), 'Split'),
        ]
        for error, action in cases:
            self.assertIn(action, error_message(error))

    def test_unknown_messages_are_not_exposed(self):
        for error in (RuntimeError('secret-token'), ValueError('secret-token'),
                      PDFIngestionError('Cannot parse PDF: confidential.pdf.')):
            message = error_message(error)
            self.assertNotIn('secret-token', message)
            self.assertNotIn('confidential.pdf', message)

    def test_renderer_configuration_keeps_json_available(self):
        message = error_message(PresentationRenderError('Configure RENDER_ARTIFACT_MODULE for the installed Artifact Tool runtime.'))
        self.assertIn('not configured', message)
        self.assertIn('JSON', message)


if __name__ == '__main__':
    unittest.main()
