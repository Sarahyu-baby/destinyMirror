import sys
import os
import pytest
from unittest.mock import MagicMock, patch, mock_open

# Mock libs before import so the app doesn't crash on load
sys.modules['face_analyzer'] = MagicMock()
sys.modules['face_visualizer'] = MagicMock()
sys.modules['destiny_predictor'] = MagicMock()

import screens


class TestResultScreen:

    @pytest.fixture
    def screen(self):
        # Basic screen init
        s = screens.ResultScreen()

        # FIX: Don't overwrite s.ids with a raw MagicMock() or Kivy breaks.
        # Instead, inject our mocks into the existing dictionary.
        s.ids['result_grid'] = MagicMock()
        s.ids['result_image'] = MagicMock()

        return s

    def test_export_features_to_csv_success(self, screen):
        """Check if CSV export works without touching disk."""

        # Dummy data
        screen.current_raw_stats = {
            'Eye_Distance': 0.45,
            'Jaw_Width': 0.60
        }

        # Mock IO to stop real file creation.
        # Force 'exists' to False so makedirs() actually triggers.
        with patch("builtins.open", mock_open()) as mocked_file, \
                patch("os.makedirs") as mocked_mkdirs, \
                patch("os.path.exists", return_value=False), \
                patch.object(screen, 'show_fortune_popup') as mocked_popup:
            screen.export_features_to_csv()

            # Did we try to make the dir?
            mocked_mkdirs.assert_called_with("feature_data")

            # Did we write the data?
            handle = mocked_file()
            handle.write.assert_any_call("Eye_Distance,0.45\r\n")
            handle.write.assert_any_call("Jaw_Width,0.6\r\n")

            # Check success popup
            args, _ = mocked_popup.call_args
            assert "exported successfully" in args[1]
