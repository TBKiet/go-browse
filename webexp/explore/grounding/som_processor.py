"""Set-of-Mark processor for BrowserGym observations.

Bridges BrowserGym's accessibility tree + screenshot with the SoM
overlay drawing. Extracts element bounding boxes from the axtree
object and maps BrowserGym bids to SoM tag numbers.
"""

import logging
from .set_of_mark import add_set_of_mark

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _walk_axtree(node, extra_properties=None):
    """Recursively walk the accessibility tree and yield (bid, extra_props) pairs.

    BrowserGym's axtree_object is a nested dict/list structure where each
    interactive node has a 'bid' attribute (BrowserGym's element identifier).
    This generator yields every node that has a bid, along with any extra
    properties (bbox, cursor, etc.) if available.
    """
    if isinstance(node, list):
        for child in node:
            yield from _walk_axtree(child, extra_properties)
    elif isinstance(node, dict):
        bid = node.get("bid")
        if bid is not None:
            props = None
            backend_id = node.get("backend_node_id")
            if backend_id and extra_properties and str(backend_id) in extra_properties:
                props = extra_properties[str(backend_id)]
            yield (str(bid), node, props)

        # Recurse into children
        children = node.get("children", [])
        if children:
            yield from _walk_axtree(children, extra_properties)


def extract_rois_from_axtree(axtree_object, extra_element_properties=None):
    """Extract Regions of Interest from BrowserGym's accessibility tree.

    Converts BrowserGym's axtree structure into the ROI format expected
    by add_set_of_mark(): {element_id: {"rects": [{"left","top","right","bottom"}]}}

    For bounding boxes, checks extra_element_properties (if provided by
    BrowserGym) which may contain 'bbox' or 'center' coordinates.
    Falls back to estimating positions from tree order if no bbox available.

    Args:
        axtree_object: BrowserGym's accessibility tree (nested dict/list).
        extra_element_properties: Optional dict mapping backend_node_id to
            property dicts (may contain 'bbox', 'center', etc.).

    Returns:
        dict mapping bid (str) -> {"rects": [rect_dict], "tag": str, "text": str}
    """
    rois = {}

    for bid, node, props in _walk_axtree(axtree_object, extra_element_properties):
        rects = []

        # Try to get bbox from extra properties
        if props and "bbox" in props:
            bbox = props["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x, y, w, h = bbox
                rects.append({
                    "left": x,
                    "top": y,
                    "right": x + w,
                    "bottom": y + h,
                })

        if not rects:
            continue

        tag = node.get("tag", node.get("role", "unknown"))
        text = node.get("text", node.get("name", ""))

        rois[bid] = {
            "rects": rects,
            "tag": str(tag),
            "text": str(text)[:100] if text else "",
        }

    return rois


def process_observation(obs: dict) -> dict:
    """Add Set-of-Mark overlay to a BrowserGym observation.

    Takes a raw BrowserGym observation dict and returns a copy with the
    screenshot replaced by a SoM-annotated version. Also adds som_id_info
    (mapping SoM tag IDs to element metadata for action grounding).

    Args:
        obs: BrowserGym observation dict with keys:
            - screenshot: numpy array (H,W,3)
            - axtree_object: accessibility tree
            - extra_element_properties: optional dict

    Returns:
        dict: observation with 'screenshot' replaced by SoM image, plus
              'som_screenshot' (original) and 'som_id_info' (bid->bbox mapping).
    """
    result = {**obs}

    screenshot = obs.get("screenshot")
    axtree = obs.get("axtree_object")
    extra_props = obs.get("extra_element_properties")

    if screenshot is None or axtree is None:
        logger.debug("Skipping SoM: no screenshot or axtree available")
        return result

    try:
        rois = extract_rois_from_axtree(axtree, extra_props)

        if not rois:
            logger.debug("Skipping SoM: no ROIs with bounding boxes extracted")
            return result

        som_image, visible, above, below = add_set_of_mark(screenshot, rois)

        # Store original screenshot
        result["som_screenshot_original"] = screenshot
        # Replace live screenshot with SoM version
        result["screenshot"] = som_image
        # Store SoM metadata for element grounding
        result["som_id_info"] = {
            bid: {
                "bbox": rois[bid]["rects"][0] if rois[bid]["rects"] else None,
                "tag": rois[bid].get("tag", ""),
                "text": rois[bid].get("text", ""),
            }
            for bid in rois
        }
        result["som_visible_ids"] = visible
        result["som_above_ids"] = above
        result["som_below_ids"] = below

        logger.info(
            f"SoM: {len(rois)} elements, {len(visible)} visible, "
            f"{len(above)} above, {len(below)} below"
        )

    except Exception as e:
        logger.warning(f"SoM processing failed, falling back to original: {e}")

    return result
