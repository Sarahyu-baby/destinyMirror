from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from kivy.utils import platform

# Import screens and layout from screens module
from screens import MainScreen, ResultScreen, KV_LAYOUT

class DestinyMirror(App):
    """
    Main Application class.
    """
    def build(self):
        print("Destiny Mirror is starting...") # Debug print

        # Request permissions on Android devices
        if platform == 'android':
            from android.permissions import request_permissions, Permission
            request_permissions(
                [Permission.CAMERA, Permission.WRITE_EXTERNAL_STORAGE, Permission.READ_EXTERNAL_STORAGE]
            )

        # Load the KV Layout defined in screens.py
        Builder.load_string(KV_LAYOUT)

        # Setup Screen Manager
        sm = ScreenManager()
        sm.add_widget(MainScreen(name='main'))
        sm.add_widget(ResultScreen(name='result'))
        return sm

    def on_stop(self):
        """Handle app closure cleanup."""
        if self.root:
            main = self.root.get_screen('main')
            main.stop_camera()


if __name__ == '__main__':
    try:
        DestinyMirror().run()
    except Exception as e:
        print(f"App Crashed: {e}")
        import traceback
        traceback.print_exc()