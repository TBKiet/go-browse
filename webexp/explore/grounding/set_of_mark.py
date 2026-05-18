"""Set-of-Mark (SoM) visual grounding.

Ported from Explorer (ACL 2025) and adapted for BrowserGym.
Draws colored numbered tags on screenshots corresponding to interactive
elements in the accessibility tree. Enables vision-language models to
reference elements by number (e.g., "click [42]") instead of raw bids.

Core drawing logic from Explorer's traj_gen/set_of_mark.py.
"""

import random
from PIL import Image, ImageDraw, ImageFont

TOP_NO_LABEL_ZONE = 20  # Don't print labels too close to the top


def add_set_of_mark(screenshot, rois: dict) -> tuple:
    """Add Set-of-Mark overlay to a screenshot.

    Args:
        screenshot: PIL Image or numpy array of the page screenshot.
        rois: Dict mapping element_id (str) -> {"rects": [{"left","top","right","bottom"}]}
              Same format as Explorer's getInteractiveRects() output.

    Returns:
        tuple: (annotated_image, visible_rect_ids, rects_above, rects_below)
    """
    if not isinstance(screenshot, Image.Image):
        screenshot = Image.fromarray(screenshot)
    return _add_set_of_mark(screenshot, rois)


def _add_set_of_mark(screenshot: Image.Image, rois: dict):
    visible_rects = []
    rects_above = []
    rects_below = []

    fnt = ImageFont.load_default()
    base = screenshot.convert("L").convert("RGBA")
    overlay = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(overlay)

    for elem_id in rois:
        rects = rois[elem_id].get("rects", [])
        for rect in rects:
            if not rect:
                continue
            w = rect.get("right", 0) - rect.get("left", 0)
            h = rect.get("bottom", 0) - rect.get("top", 0)
            if w * h == 0:
                continue

            mid_x = (rect["right"] + rect["left"]) / 2.0
            mid_y = (rect["top"] + rect["bottom"]) / 2.0

            if 0 <= mid_x < base.size[0]:
                if mid_y < 0:
                    rects_above.append(elem_id)
                elif mid_y >= base.size[1]:
                    rects_below.append(elem_id)
                else:
                    visible_rects.append(elem_id)
                    _draw_roi(draw, int(elem_id), fnt, rect)

    comp = Image.alpha_composite(base, overlay)
    overlay.close()
    return comp, visible_rects, rects_above, rects_below


def _draw_roi(draw, idx: int, font, rect: dict):
    color = _color(idx)
    luminance = color[0] * 0.3 + color[1] * 0.59 + color[2] * 0.11
    text_color = (0, 0, 0, 255) if luminance > 90 else (255, 255, 255, 255)

    roi = [(rect["left"], rect["top"]), (rect["right"], rect["bottom"])]

    label_location = (rect["right"], rect["top"])
    label_anchor = "rb"

    if label_location[1] <= TOP_NO_LABEL_ZONE:
        label_location = (rect["right"], rect["bottom"])
        label_anchor = "rt"

    draw.rectangle(
        roi, outline=color, fill=(color[0], color[1], color[2], 48), width=2
    )

    bbox = draw.textbbox(
        label_location, str(idx), font=font, anchor=label_anchor, align="center"
    )
    bbox = (bbox[0] - 3, bbox[1] - 3, bbox[2] + 3, bbox[3] + 3)
    draw.rectangle(bbox, fill=color)

    draw.text(
        label_location,
        str(idx),
        fill=text_color,
        font=font,
        anchor=label_anchor,
        align="center",
    )


def _color(identifier):
    rnd = random.Random(int(identifier))
    color = [rnd.randint(0, 255), rnd.randint(125, 255), rnd.randint(0, 50)]
    rnd.shuffle(color)
    color.append(255)
    return tuple(color)
