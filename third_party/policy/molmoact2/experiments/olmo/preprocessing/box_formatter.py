"""
Box Formatter for 2D Object Detection

Formats bounding boxes in <boxes> tag format similar to pointing format:
- Pointing: <points coords="1 50 60">person</points>
- Detection: <boxes coords="1 10 15 25 30">car</boxes>

Coordinates are in (idx x1 y1 x2 y2) format, scaled to 0-100 range.
"""

from typing import List, Optional, Tuple
import numpy as np


class BoxFormatter:
    """
    Formats bounding boxes for detection output.

    Output format: <boxes coords="idx x1 y1 x2 y2 ...">label</boxes>
    """

    def __init__(self, coordinate_scale: str = "100", sort_order: str = "xy"):
        """
        Args:
            coordinate_scale: "100" for 0-100 range, "1000" for 0-1000 range
            sort_order: "xy" to sort boxes by position, "random" for random order
        """
        self.coordinate_scale = coordinate_scale
        self.sort_order = sort_order

    def format_image_boxes(
        self,
        boxes: np.ndarray,
        scale: float,
        label: str,
        alt_text: Optional[str] = None,
        mode: str = "detection",
        rng=None
    ) -> str:
        """
        Format bounding boxes for a single image.

        Args:
            boxes: Array of shape (N, 4) with (x1, y1, x2, y2) coordinates
            scale: Scale factor (coordinates are already in 0-100 if scale=100)
            label: Text label for the boxes
            alt_text: Alternative text (optional)
            mode: Output mode ("detection" returns just the box string)
            rng: Random number generator for shuffling

        Returns:
            Formatted string like "<boxes coords=\"1 10 15 25 30\">car</boxes>"
        """
        if len(boxes) == 0:
            return "There are none."

        # 1 for the frame index of the first image
        coord_str = "1 " + self.build_single_image_box_coordinates(rng, boxes, scale)
        box_str = self.build_box_str_from_coord_str(label, alt_text, coord_str)
        return self.build_box_output(box_str, len(boxes), mode)

    def build_single_image_box_coordinates(
        self,
        rng,
        boxes: np.ndarray,
        scale: float
    ) -> str:
        """
        Build coordinate string for boxes: idx x1 y1 x2 y2 ...

        Args:
            rng: Random number generator
            boxes: Array of shape (N, 4) with (x1, y1, x2, y2) coordinates
            scale: Scale factor for normalization

        Returns:
            Coordinate string like "1 10 15 25 30 2 35 40 50 60"
        """
        boxes = np.array(boxes)
        if len(boxes) == 0:
            return ""

        # Scale boxes if needed
        if scale != 1.0:
            # Boxes are already in 0-scale range, normalize to target coordinate scale
            if self.coordinate_scale == "100":
                # boxes are already in 0-100 if scale=100
                scaled_boxes = boxes
            else:
                # Scale to 0-1000
                scaled_boxes = boxes * (1000 / scale)
        else:
            scaled_boxes = boxes * (100 if self.coordinate_scale == "100" else 1000)

        # Round to integers
        scaled_boxes = np.round(scaled_boxes).astype(int)

        # Clip to valid range
        max_val = 100 if self.coordinate_scale == "100" else 1000
        scaled_boxes = np.clip(scaled_boxes, 0, max_val)

        # Sort boxes by position (top-left corner)
        if self.sort_order == "xy":
            # Sort by y1 first, then x1
            sort_idx = np.lexsort((scaled_boxes[:, 0], scaled_boxes[:, 1]))
        elif self.sort_order == "random" and rng is not None:
            sort_idx = rng.permutation(len(scaled_boxes))
        else:
            sort_idx = np.arange(len(scaled_boxes))

        scaled_boxes = scaled_boxes[sort_idx]

        # Format coordinates
        if self.coordinate_scale == "100":
            text_format = "%02d"
        else:
            text_format = "%03d"

        box_strs = []
        for idx, (x1, y1, x2, y2) in enumerate(scaled_boxes, start=1):
            coords = " ".join([text_format % c for c in [x1, y1, x2, y2]])
            box_strs.append(f"{idx} {coords}")

        return " ".join(box_strs)

    def build_box_str_from_coord_str(
        self,
        label: str,
        alt_text: Optional[str],
        coord_str: str
    ) -> str:
        """
        Build <boxes coords="...">label</boxes> string.

        Args:
            label: Text label
            alt_text: Alternative text (optional)
            coord_str: Coordinate string

        Returns:
            Formatted box string
        """
        if not coord_str:
            return ""

        text = "<boxes"
        if alt_text is not None and alt_text != label:
            text += f' alt="{alt_text}"'
        text += f' coords="{coord_str}"'
        return text + ">" + label + "</boxes>"

    def build_box_output(self, box_str: str, count: int, mode: str = "detection") -> str:
        """
        Build final box output based on mode.

        Args:
            box_str: Formatted box string
            count: Number of boxes
            mode: Output mode

        Returns:
            Final output string
        """
        if count == 0:
            return "There are none."

        if mode in ["detection", "detect", "box", "boxes"]:
            return box_str
        elif mode == "count":
            return str(count)
        elif mode == "detect_count":
            return f"Counting the {box_str} shows a total of {count}."
        else:
            return box_str
