import os
import datetime
import cv2
import csv
import re
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.factory import Factory
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.utils import get_color_from_hex

from face_analyzer import FaceAnalyzer
from face_visualizer import FaceVisualizer
from destiny_predictor import DestinyPredictor
# --- UI Layout Definition ---
KV_LAYOUT = '''
#:import get_color_from_hex kivy.utils.get_color_from_hex

# Custom Rounded Button definition
<RoundedButton@Button>:
    background_color: (0,0,0,0)  # Hide default background
    background_normal: ''        # Ensure no image is used
    background_down: ''          # Ensure no image is used on press
    canvas.before:
        Color:
            # Auto-darken color when pressed
            rgba: (self.background_color[0]*0.8, self.background_color[1]*0.8, self.background_color[2]*0.8, 1) if self.state == 'down' else self.background_color
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [15,]

# Main Screen definition
<MainScreen>:
    name: 'main'
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 15
        canvas.before:
            # Deep midnight blue background
            Color:
                rgba: get_color_from_hex('#0F172A')
            Rectangle:
                pos: self.pos
                size: self.size

        # Title Bar
        Label:
            text: "DESTINY MIRROR"
            size_hint_y: 0.12
            font_size: '28sp'
            font_name: 'Roboto' 
            bold: True
            color: get_color_from_hex('#38BDF8') # Neon Cyan
            canvas.before:
                Color:
                    rgba: get_color_from_hex('#1E293B')
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [0, 0, 15, 15]

        # Camera preview area (with border)
        BoxLayout:
            id: cam_layout
            size_hint_y: 0.65
            padding: 5
            canvas.before:
                Color:
                    rgba: get_color_from_hex('#334155')
                Line:
                    rectangle: self.x, self.y, self.width, self.height
                    width: 1.5

            Image:
                id: camera_preview
                allow_stretch: True
                keep_ratio: True

        # Status Label
        Label:
            id: status_label
            text: "Loading AI Core..."
            size_hint_y: 0.05
            font_size: '14sp'
            color: get_color_from_hex('#94A3B8')

        # Capture Button
        RoundedButton:
            id: capture_btn
            text: "CAPTURE & PREDICT"
            size_hint_y: 0.15
            background_color: get_color_from_hex('#0EA5E9') # Bright Blue
            font_size: '20sp'
            bold: True
            disabled: True 
            on_release: root.capture_and_analyze()

<ResultScreen>:
    name: 'result'
    BoxLayout:
        orientation: 'vertical'
        padding: 20
        spacing: 10
        canvas.before:
            Color:
                rgba: get_color_from_hex('#0F172A')
            Rectangle:
                pos: self.pos
                size: self.size

        Label:
            text: "YOUR DESTINY"
            size_hint_y: 0.08
            font_size: '26sp'
            bold: True
            color: get_color_from_hex('#F472B6') # Pink-Purple

        # --- NEW VIP SUBSCRIBE BUTTON (Prominent & Large) ---
        RoundedButton:
            text: "SUBSCRIBE VIP ($5/mo)"
            size_hint_y: 0.12
            background_color: get_color_from_hex('#F59E0B') # Gold/Amber
            font_size: '20sp'
            bold: True
            on_release: root.subscribe_vip()
            canvas.before:
                Color:
                    rgba: (1, 0.84, 0, 0.3) # Gold Glow effect
                BoxShadow:
                    pos: self.pos
                    size: self.size
                    offset: 0, -2
                    spread_radius: 5, 5
                    border_radius: [15, 15, 15, 15]

        # Result Image Display
        BoxLayout:
            size_hint_y: 0.30
            padding: 2
            canvas.before:
                Color:
                    rgba: (1, 1, 1, 0.2)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [10,]

            Image:
                id: result_image
                allow_stretch: True
                keep_ratio: True
                source: ''

        Label:
            text: "Select a card to reveal:"
            size_hint_y: 0.05
            font_size: '14sp'
            italic: True
            color: get_color_from_hex('#CBD5E1')

        # Button Grid
        ScrollView:
            size_hint_y: 0.35
            GridLayout:
                id: result_grid
                cols: 2
                spacing: 12
                padding: 5
                size_hint_y: None
                height: self.minimum_height
                row_default_height: '65dp'

        # Bottom Action Bar
        BoxLayout:
            size_hint_y: 0.10
            spacing: 15

            # Updated Save Button Text
            RoundedButton:
                text: "SAVE (VIP ONLY)"
                background_color: get_color_from_hex('#10B981') # Emerald Green
                font_size: '14sp'
                bold: True
                on_release: root.save_results_to_csv()

            RoundedButton:
                text: "BACK"
                background_color: get_color_from_hex('#EF4444') # Warning Red
                font_size: '14sp'
                bold: True
                on_release: app.root.current = 'main'
'''
# --- Python Logic ---

class FortunePopup(Popup):
    """
    Custom Popup class to display detailed fortune results.
    Handles dynamic text wrapping and scrolling.
    """

    def __init__(self, title, content_text, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.title_size = '20sp'
        self.title_color = get_color_from_hex('#38BDF8')
        self.size_hint = (0.85, 0.6)
        self.auto_dismiss = True

        # Popup styling
        self.separator_color = get_color_from_hex('#38BDF8')

        layout = BoxLayout(orientation='vertical', padding=20, spacing=15)

        scroll = ScrollView()

        self.label = Label(
            text=content_text,
            size_hint_y=None,
            halign='center',
            valign='top',
            font_size='18sp',
            line_height=1.5,
            color=(1, 1, 1, 1)
        )

        # Bind width to text_size to ensure proper wrapping
        self.label.bind(width=lambda *x: self.label.setter('text_size')(self.label, (self.label.width, None)))
        # Bind texture_size to height to ensure the scrollview works correctly
        self.label.bind(texture_size=self.update_height)

        scroll.add_widget(self.label)

        close_btn = Factory.RoundedButton(
            text="Close",
            size_hint_y=None,
            height='45dp',
            background_color=get_color_from_hex('#475569')
        )
        close_btn.bind(on_release=self.dismiss)

        layout.add_widget(scroll)
        layout.add_widget(close_btn)
        self.content = layout

    def update_height(self, instance, size):
        instance.height = size[1]
class MainScreen(Screen):
    """
    Main screen for camera preview and capturing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.analyzer = FaceAnalyzer()
        self.visualizer = FaceVisualizer()
        self.update_event = None
        self.current_frame = None

        self.predictor = None
        Clock.schedule_once(self.init_predictor, 0.5)

    def init_predictor(self, dt):
        """Initialize the ML predictor in a scheduled event."""
        self.predictor = DestinyPredictor()
        if self.predictor.is_ready:
            self.ids.status_label.text = "AI Core Online. Ready."
            self.ids.capture_btn.disabled = False
        else:
            self.ids.status_label.text = "Error: AI Core Offline (Run train_and_save.py)"

    def on_enter(self):
        self.start_camera()

    def on_leave(self):
        self.stop_camera()

    def start_camera(self):
        if self.capture is None:
            self.capture = cv2.VideoCapture(0)
            if not self.capture.isOpened():
                self.capture = cv2.VideoCapture(1)

        if self.update_event is None and self.capture.isOpened():
            self.update_event = Clock.schedule_interval(self.update, 1.0 / 30.0)
        elif not self.capture.isOpened():
            self.ids.status_label.text = "Error: Camera Access Denied"

    def stop_camera(self):
        if self.capture:
            self.capture.release()
            self.capture = None
        if self.update_event:
            self.update_event.cancel()
            self.update_event = None

    def update(self, dt):
        """Update loop for camera frame."""
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                self.current_frame = frame
                display_frame = cv2.flip(frame, 1)
                buf = cv2.flip(display_frame, 0).tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.ids.camera_preview.texture = texture

    def capture_and_analyze(self):
        if self.current_frame is None:
            return

        self.ids.status_label.text = "Processing Neural Data..."
        Clock.schedule_once(self.do_process, 0.1)

    def save_captured_image(self, frame, prefix="face"):
        save_dir = "captures"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"{prefix}_{timestamp}.png")
        cv2.imwrite(filename, frame)
        return filename

    def do_process(self, dt):
        """Process the image, run predictions, and transition to result screen."""
        stats = self.analyzer.process_image(self.current_frame)

        if stats:
            fortune_results = self.predictor.predict_fortune(stats)
            lms_data = self.analyzer.landmarks_np
            custom_pts_data = self.analyzer.custom_points

            img_to_draw = self.current_frame.copy()
            img_with_dots = self.visualizer.draw_landmarks(img_to_draw, lms_data)
            final_visualized_img = self.visualizer.draw_custom_points(img_with_dots, custom_pts_data)

            saved_path = self.save_captured_image(final_visualized_img, prefix="analyzed")

            app = App.get_running_app()
            result_screen = app.root.get_screen('result')
            result_screen.display_data(fortune_results, saved_path)
            app.root.current = 'result'
            self.ids.status_label.text = "Analysis Complete"
        else:
            self.ids.status_label.text = "No Face Detected"
class ResultScreen(Screen):
    """
    Screen to display prediction results with interactive buttons.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_fortune_results = {}

        # Color Palette (Hex Codes)
        self.color_map = {
            'LOVE': '#EC4899',  # Hot Pink
            'WEALTH': '#EAB308',  # Golden Yellow
            'HEALTH': '#22C55E',  # Bright Green
            'CAREER': '#3B82F6',  # Royal Blue
            'AUTHORITY': '#A855F7',  # Purple
            'LATER-LIFE': '#64748B',  # Slate Gray
            'SOCIAL': '#F97316',  # Orange
            'CHILDREN': '#14B8A6',  # Teal
            'DEFAULT': '#06B6D4'  # Cyan
        }

    def subscribe_vip(self):
        """Triggered when the Subscribe button is pressed."""
        self.show_fortune_popup(
            "VIP Subscription",
            "Subscription Successful!\n\nYou have unlocked exclusive VIP insights and the ability to save your destiny."
        )

    def display_data(self, fortune_results, img_path):
        """Populate the grid with result buttons."""
        self.current_fortune_results = fortune_results

        if img_path:
            self.ids.result_image.source = img_path
            self.ids.result_image.reload()

        grid = self.ids.result_grid
        grid.clear_widgets()

        for category, data in fortune_results.items():
            raw_label = data['label']
            full_sentence = data['sentence']

            key_upper = category.upper().replace("_", "-")

            # --- Generic VIP Logic ---
            # Use Regex to strip any trailing numbers (e.g., SOCIAL2 -> SOCIAL)
            base_key = re.sub(r'\d+$', '', key_upper)

            display_text = raw_label
            # If the key changed (SOCIAL2 != SOCIAL), it's a secondary VIP metric
            if base_key != key_upper:
                display_text = f"{base_key} (VIP Content)"

            # Use base_key to find the color (So SOCIAL2 gets SOCIAL's Orange color)
            hex_color = self.color_map.get(base_key, self.color_map.get('DEFAULT'))
            bg_color = get_color_from_hex(hex_color)

            # Create a RoundedButton using the Factory
            btn = Factory.RoundedButton(
                text=display_text,
                font_size='14sp' if "(VIP" in display_text else '16sp',
                bold=True,
                background_color=bg_color,
                color=(1, 1, 1, 1)
            )
            btn.bind(on_release=lambda instance, t=display_text, c=full_sentence: self.show_fortune_popup(t, c))
            grid.add_widget(btn)

    def show_fortune_popup(self, title, content):
        popup = FortunePopup(title=title, content_text=content)
        popup.open()

    def save_results_to_csv(self):
        """Save current predictions to a CSV file."""
        if not self.current_fortune_results:
            return

        save_dir = "fortune_results"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"fortune_{timestamp}.csv")

        try:
            with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['Category', 'Prediction']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for category, data in self.current_fortune_results.items():
                    writer.writerow({'Category': data['label'], 'Prediction': data['sentence']})

            # VIP Prompt logic
            self.show_fortune_popup(
                "VIP Access",
                f"Processing payment...\n$5.00 charged.\n\nDestiny archived to:\n{filename}"
            )
            print(f"Results saved to {filename}")

        except Exception as e:
            self.show_fortune_popup("Error", f"Failed to save:\n{e}")
