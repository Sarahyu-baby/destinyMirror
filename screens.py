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
