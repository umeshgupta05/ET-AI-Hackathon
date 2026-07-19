"""Official RBI Mahatma Gandhi (New) Series feature specifications.

These specifications describe expected features and required capture channels.
They are not training labels and do not turn a research image model into a
certified banknote authentication device.
"""

from __future__ import annotations

from typing import Any


RBI_REFERENCE_URLS = (
    "https://www.rbi.org.in/scripts/FAQView.aspx?Id=136",
    "https://www.rbi.org.in/CommonPerson/english/scripts/Notification.aspx?Id=3005",
    "https://rbi.org.in/scripts/PublicationsView.aspx?Id=18086",
)


COMMON_FEATURES = (
    "see_through_register",
    "gandhi_portrait",
    "electrotype_watermark",
    "microlettering",
    "windowed_security_thread",
    "ascending_serial_number_panels",
)


RBI_MGNS_SPECIFICATIONS: dict[str, dict[str, Any]] = {
    "10": {
        "dimensions_mm": [123, 63],
        "base_colour": "chocolate_brown",
        "reverse_motif": "Konark Sun Temple",
        "microtext": ["RBI", "BHARAT", "INDIA", "10"],
        "thread": "windowed_demetallised",
        "thread_colour_shift": False,
        "latent_image": False,
        "identification_mark": None,
        "angular_bleed_lines": 0,
    },
    "20": {
        "dimensions_mm": [129, 63],
        "base_colour": "greenish_yellow",
        "reverse_motif": "Ellora Caves",
        "microtext": ["RBI", "BHARAT", "INDIA", "20"],
        "thread": "windowed_demetallised",
        "thread_colour_shift": False,
        "latent_image": False,
        "identification_mark": None,
        "angular_bleed_lines": 0,
    },
    "50": {
        "dimensions_mm": [135, 66],
        "base_colour": "fluorescent_blue",
        "reverse_motif": "Hampi with Chariot",
        "microtext": ["RBI", "BHARAT", "INDIA", "50"],
        "thread": "windowed_demetallised",
        "thread_colour_shift": False,
        "latent_image": False,
        "identification_mark": None,
        "angular_bleed_lines": 0,
    },
    "100": {
        "dimensions_mm": [142, 66],
        "base_colour": "lavender",
        "reverse_motif": "Rani ki Vav",
        "microtext": ["RBI", "BHARAT", "INDIA", "100"],
        "thread": "windowed_colour_shifting",
        "thread_colour_shift": True,
        "latent_image": True,
        "identification_mark": "triangle",
        "angular_bleed_lines": 4,
    },
    "200": {
        "dimensions_mm": [146, 66],
        "base_colour": "bright_yellow",
        "reverse_motif": "Sanchi Stupa",
        "microtext": ["RBI", "BHARAT", "INDIA", "200"],
        "thread": "windowed_colour_shifting",
        "thread_colour_shift": True,
        "latent_image": True,
        "identification_mark": "raised_H",
        "angular_bleed_lines": 4,
        "bleed_line_circles": 2,
        "denomination_ink_colour_shift": True,
    },
    "500": {
        "dimensions_mm": [150, 66],
        "base_colour": "stone_grey",
        "reverse_motif": "Red Fort",
        "microtext": ["RBI", "BHARAT", "INDIA", "500"],
        "thread": "windowed_colour_shifting",
        "thread_colour_shift": True,
        "latent_image": True,
        "identification_mark": "circle",
        "angular_bleed_lines": 5,
        "denomination_ink_colour_shift": True,
    },
    "2000": {
        "dimensions_mm": [166, 66],
        "base_colour": "magenta",
        "reverse_motif": "Mangalyaan",
        "microtext": ["RBI", "2000"],
        "thread": "windowed_colour_shifting",
        "thread_colour_shift": True,
        "latent_image": True,
        "identification_mark": "horizontal_rectangle",
        "angular_bleed_lines": 7,
        "denomination_ink_colour_shift": True,
        "circulation_note": "withdrawn_from_circulation_but_remains_legal_tender",
    },
}


def get_rbi_specification(denomination: str | None) -> dict[str, Any] | None:
    normalized = str(denomination or "").replace("INR", "").replace("Rs", "").strip()
    specification = RBI_MGNS_SPECIFICATIONS.get(normalized)
    if not specification:
        return None
    width, height = specification["dimensions_mm"]
    return {
        "series": "Mahatma Gandhi New Series",
        "denomination": normalized,
        "expected_aspect_ratio": round(width / height, 4),
        "common_features": list(COMMON_FEATURES),
        "capture_requirements": {
            "rgb_front": ["microlettering", "serial_panels", "identification_mark", "bleed_lines"],
            "rgb_back": ["reverse_motif", "year_of_printing", "language_panel"],
            "transmitted": ["watermark", "electrotype", "see_through_register", "continuous_thread"],
            "uv": ["thread_fluorescence", "number_panel_fluorescence", "optical_fibres"],
            "tilt_rgb": ["thread_colour_shift", "denomination_ink_colour_shift"],
            "machine_sensor": ["magnetic_thread", "double_feed", "thickness", "physical_dimensions"],
        },
        "source_urls": list(RBI_REFERENCE_URLS),
        **specification,
    }
