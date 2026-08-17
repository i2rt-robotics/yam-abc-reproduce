import dataclasses
import json
import logging
import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Callable, List, Tuple, Union, Optional

import numpy as np

from olmo.data.video_object_tracking_datasets import PointTrajectoryEntry
from olmo.util import parse_timestamp


class DefaultTimestampFormatter:
    def __call__(self, time):
        return "%0.1f" % time


class PointFormatter:
    """Class that converts points to output text"""

    def format_video_tracks(self, frames_data: List[PointTrajectoryEntry], scale, label, alt_text=None, rng=None,
                            start_end_only=False, single_point_track=False, from_initial_points=None):
        """
        Format video tracks

        start_end_only: Only track the first and last point of each object
        first_object: Only track the first object (object id 0 in the raw data)
        from_initial_points: Only track the objects that start from `from_initial_points`
        """
        raise NotImplementedError()

    def format_video_points(self, timestamps: List[float], points: List[Tuple[float, float]], scale, label, alt_text=None, mode="point_then_count", rng=None):
        """
        Format video points
        """
        raise NotImplementedError()

    def format_image_points(self, points, scale, label, alt_text=None, mode="point_then_count", rng=None):
        """
        Format points for a single image
        """
        raise NotImplementedError()

    def format_multi_image_points(self, image_indices, points, scale, label, alt_text=None, mode="point_then_count", rng=None):
        """
        Format multi-image points
        """
        raise NotImplementedError()

    def extract_points(self, text,  image_w, image_h) -> List[Tuple[float, float]]:
        raise NotImplementedError()

    def extract_multi_image_points(self, text, image_w, image_h) -> List[Tuple[float, float, float]]:
        raise NotImplementedError()

    def extract_trajectories(self, text, width, height, video_fps, format=None) -> List[PointTrajectoryEntry]:
        raise NotImplementedError()

    def extract_tracks(self, text, width, height, video_fps, format=None) -> List[PointTrajectoryEntry]:
        return self.extract_trajectories(text, width, height, video_fps, format=format)


# Universal point extraction methods, its a bit of hack to try and support this universally
# but our eval code currently does not have access to the specific Formatter used
# so we live with it
def extract_points(text, image_w, image_h) -> List[Tuple[float, float]]:
    for cls in [UnifiedPointFormatter(), PointFormattingV1]:
        points = cls.extract_points(text, image_w, image_h)
        if points:
            return points
    return []


def extract_any_points(text, metadata) -> List[Tuple[int, float, float, float]]:
    if "image_size" in metadata:
        w, h = metadata["image_size"]
    elif "image_group_size" in metadata:
        w, h = zip(*metadata["image_group_size"])
    else:
        return []
    points = UnifiedPointFormatter().extract_multi_image_points(text, w, h, extract_ids=True)
    # put example id first
    if "image_group_size" in metadata:
        # Text uses one-based indexing, switch to 0 based
        points = [[int(ix), t-1, x, y] for t, ix, x, y in points]
    else:
        points = [[int(ix), t, x, y] for t, ix, x, y in points]
    if points:
        return points
    return []


def extract_multi_image_points(text, image_w, image_h) -> List[Tuple[float, float, float]]:
    for cls in [UnifiedPointFormatter(), PointFormattingV1]:
        points = cls.extract_multi_image_points(text, image_w, image_h)
        if points:
            return points
    return []


def extract_trajectories(text, width, height, video_fps, format=None) -> List[PointTrajectoryEntry]:
    for cls in [UnifiedPointFormatter(), PointFormattingV1]:
        points = cls.extract_trajectories(text, width, height, video_fps, format)
        if points:
            return points
    return []


def extract_tracks(text, width, height, video_fps, format=None) -> List[PointTrajectoryEntry]:
    return extract_trajectories(text, width, height, video_fps, format=format)


@dataclasses.dataclass
class UnifiedPointFormatter(PointFormatter):
    format_output_timestamp: Callable = DefaultTimestampFormatter()
    coordinate_scale: str = "1000"
    sort_order: str = "xy"
    sort_by_object_id: bool = True
    sort_tracking: bool = True
    point_str_format: str = "coord_str"
    image_sep: str = "\t"

    @staticmethod
    def build_for_format(pointing_format: str="html-v1", format_output_timestamp=DefaultTimestampFormatter()):
        if pointing_format == "html-v1":
            return UnifiedPointFormatter(format_output_timestamp)
        elif pointing_format == "html-v2":
            return UnifiedPointFormatter(format_output_timestamp, image_sep=";")
        else:
            raise NotImplementedError(pointing_format)

    def __post_init__(self):
        assert self.image_sep in ["\t", ":", ",", ";"]
        if self.point_str_format == "coord_str":
            self.coord_regex = re.compile(rf"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
        elif self.point_str_format == "tab_sep":
            self.coord_regex = re.compile(rf"<(?:points|tracks)\t[^\t]+\t[^\t]+\t(.*)/?>")
        else:
            raise NotImplementedError(self.point_str_format)

        self.frame_regex = re.compile(rf"(?:^|\t|:|,|;)([0-9\.]+) ([0-9\. ]+)")

        if self.coordinate_scale == "1000":
            self.points_regex = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
        elif self.coordinate_scale == "100":
            self.points_regex = re.compile(rf"([0-9]+) ([0-9]\.[0-9][0-9]) ([0-9]\.[0-9][0-9])")
        else:
            raise NotImplementedError(self.coordinate_scale)

    def _points_from_num_str(self, text, image_w, image_h, extract_ids=False):
        all_points = []
        for points in self.points_regex.finditer(text):
            ix, x, y = points.group(1), points.group(2), points.group(3)
            if self.coordinate_scale == "1000":
                x, y = float(x)/1000*image_w, float(y)/1000*image_h
            elif self.coordinate_scale == "100":
                x, y = float(x)/100*image_w, float(y)/100*image_h
            else:
                raise NotImplementedError(self.coordinate_scale)
            if 0 <= x <= image_w and 0 <= y <= image_h:
                yield int(ix), x, y

    def extract_points(self, text: str,  image_w: float, image_h: float) -> List[Tuple[float, float]]:
        points = self.extract_multi_image_points(text, image_w, image_h)
        return [(x, y) for _, x, y in points]

    def extract_multi_image_points(self, text, image_w, image_h, extract_ids=False) -> List[Tuple[float]]:
        # TODO will need to handle multiple w/h values for the multi-image case
        all_points = []
        if isinstance(image_w, (list, tuple)) and isinstance(image_h, (list, tuple)):
            assert len(image_w) == len(image_h)
            diff_res = True
        else:
            diff_res = False
        for coord in self.coord_regex.finditer(text):
            for point_grp in self.frame_regex.finditer(coord.group(1)):
                frame_id = int(point_grp.group(1)) if diff_res else float(point_grp.group(1))
                w, h = (image_w[frame_id-1], image_h[frame_id-1]) if diff_res else (image_w, image_h)
                for idx, x, y in self._points_from_num_str(point_grp.group(2), w, h):
                    if extract_ids:
                        all_points.append((frame_id, idx, x, y))
                    else:
                        all_points.append((frame_id, x, y))
        return all_points

    def extract_trajectories(self, text, width, height, video_fps, format=None) -> List[PointTrajectoryEntry]:
        points = self.extract_multi_image_points(text, width, height, extract_ids=True)
        grouped_by_timestamp = defaultdict(list)
        for t, ix, x, y in points:
            grouped_by_timestamp[t].append((ix, x, y))

        out: List[PointTrajectoryEntry] = []
        for timestamp in sorted(grouped_by_timestamp):
            frame = round(timestamp * video_fps)
            points = {}
            for ix, x, y in grouped_by_timestamp[timestamp]:
                if str(ix) in points:
                    # Formatting error, objects should only appear once in each frame
                    continue
                points[str(ix)] = dict(point=[x, y])
            out.append(dict(
                time=timestamp,
                frame=frame,
                points=points
            ))
        return out

    def format_video_tracks(self, frames_data, scale, label, alt_text=None, rng=None,
                            start_end_only=False, single_point_track=False, from_initial_points=None):
        """
        Build video tracking output text
        """

        if len(frames_data) == 0:
            return "No tracks available."

        if start_end_only:
            frames_data = self._filter_all_but_start_end(frames_data)
        # if single_point_track:
        #     frames_data = self._filter_initial_point(frames_data)
        if from_initial_points is not None:
            frames_data = self._filter_for_initial_points(frames_data, from_initial_points)

        coord_str = self.build_video_track_coordinates(frames_data, scale)
        point_str = self.build_point_str_from_coord_str(label, alt_text, coord_str, True)
        # tracking not used for counting questions so mode is fixed as `point`
        return self.build_point_output(point_str, None, mode="point")

    def format_video_points(self, timestamps, points, scale, label, alt_text=None, mode="point_then_count", rng=None):
        """
        Build video pointing output text
        """
        coord_str = self._build_multi_image_coordinates(rng, timestamps, points, scale, multi_image=False)
        point_str = self.build_point_str_from_coord_str(label, alt_text, coord_str, False)
        return self.build_point_output(point_str, sum(len(x) for x in points), mode)

    def format_image_points(self, points, scale, label, alt_text=None, mode="point_then_count", rng=None):
        """
        Build singe image pointing output text
        """
        # 1 for the frame index of the first image, to match the multi-image format
        coord_str = "1 " + self.build_single_image_coordinates(rng, points, scale)
        point_str = self.build_point_str_from_coord_str(label, alt_text, coord_str, False)
        return self.build_point_output(point_str, len(points), mode)

    def format_multi_image_points(self, image_indices, points, scale, label, alt_text=None, mode="point_then_count", rng=None):
        """
        Build multi-image pointing output text
        image_indices: List of image indices/timestamps (one per image)
            Format: List[float] or List[int]
            Example: [1.0, 2.0, 3.0] or [1, 2, 3]
        points: List of point lists (one list per image)
            Format: List[List[Tuple[float, float]]]
            Each inner list contains (x, y) coordinate tuples for that image
            Example: [[(100, 200), (300, 400)], [(150, 250)], []]
        scale: Scale factor for coordinate normalization for each image
            Format: List of floats or Tuple[float, float] for each image
        label: Text label for the points
            Format: str
            Example: "person", "red car"
        """
        assert len(scale) == len(points)
        coord_str = self._build_multi_image_coordinates(rng, image_indices, points, scale, multi_image=True)
        point_str = self.build_point_str_from_coord_str(label, alt_text, coord_str, False)
        return self.build_point_output(point_str, count=sum(len(p) for p in points), mode=mode)

    def build_video_track_coordinates(self, frames_data: List[PointTrajectoryEntry], scale, skip_occluded=True) -> str:
        frame_strings = []
        point_id_to_index = {}
        for frame_data in sorted(frames_data, key=lambda x: x["time"]):
            if not frame_data["points"]:
                continue

            points_dict = frame_data["points"]

            # Convert to tuples with scaled points
            point_list = []
            for obj_id, point_info in points_dict.items():
                occluded = point_info.get("occluded", False)
                if occluded:
                    if skip_occluded:
                        continue
                point = self._scale_point(point_info["point"], scale)
                point_list.append([obj_id, point, occluded])

            if not point_list:
                # All occluded or original list was empty
                continue

            # Sort by the scaled points
            if self.sort_tracking:
                if self.sort_order == "xy":
                    point_list.sort(key=lambda x: x[1])
                else:
                    raise NotImplementedError(self.sort_order)

                # Remap point_id so it matches the new, sorted order
                for point in point_list:
                    point_id = point[0]
                    if point_id not in point_id_to_index:
                        point_id_to_index[point_id] = len(point_id_to_index)
                    point[0] = point_id_to_index[point_id] + 1

            if self.sort_by_object_id:
                # Sort by object id
                assert self.sort_order == "xy"
                point_list.sort(key=lambda x: x[0])  # sort is stable so point order is preserved

            # Convert to a string
            idx, points, occludeds = list(zip(*point_list))  # transpose to seperate lists
            frame_points_str = self._format_image_points(
                idx, points, None if skip_occluded else occludeds)

            # Add time stamp prefix
            time_str = self.format_output_timestamp(parse_timestamp(frame_data["time"]))
            frame_strings.append(f"{time_str} {frame_points_str}")
        return self.image_sep.join(frame_strings)

    @staticmethod
    def _filter_initial_point(frames_data: List[PointTrajectoryEntry]) -> List[PointTrajectoryEntry]:
        filtered = []
        for frame_data in frames_data[1:]:
            if not frame_data["points"]:
                continue
            points = frame_data["points"]
            if 0 in points:
                filtered.append(dict(frame_data, points={0: points[0]}))
        return filtered

    @staticmethod
    def _filter_for_initial_points(frames_data: List[PointTrajectoryEntry], initial_points) -> List[PointTrajectoryEntry]:
        """
        Filter frames_data to only include points AFTER initial_points
        Args:
            frames_data: List of frame data with points
            initial_points: List of initial points with 'id' and 'frame' keys
        """
        point_id_mapping = {p['id']: i for i, p in enumerate(initial_points)}
        filtered = []
        for frame_data in frames_data:
            if not frame_data["points"]:
                continue

            # Only include points that are in initial_points and after their initial frame
            # {point_idx: {"point": [x,y], "occluded": bool}}
            points_dict = {point_idx: frame_data["points"][point_id] for point_id, point_idx in point_id_mapping.items()
                           if frame_data["time"] > initial_points[point_idx]["time"] and point_id in frame_data["points"]}
            if points_dict:
                # filtered.append(dict(frame_data, points=points_dict))
                filtered.append(dict(frame_data))
            # filtered.append(points_dict)
        return filtered

    @staticmethod
    def _filter_all_but_start_end(frames_data: List[PointTrajectoryEntry]) -> List[PointTrajectoryEntry]:
        object_frames = defaultdict(list)
        for i, frame_data in enumerate(frames_data):
            for obj_id, point_info in frame_data["points"].items():
                 object_frames[obj_id].append(i)

        # Maps time -> frame_data, we might get data for this task due to how the source data
        # works so merge `frame_data` with same time here as well
        filtered_frame_data = {}
        for i, frame_data in enumerate(frames_data):
            filtered_points = {}
            for obj_id, point_info in frame_data["points"].items():
                if i == object_frames[obj_id][0] or i == object_frames[obj_id][-1]:
                    filtered_points[obj_id] = point_info
            if filtered_points:
                time = frame_data["time"]
                if time not in filtered_frame_data:
                    filtered_frame_data[time] = dict(frame_data, points={})
                filtered_frame_data[time]["points"].update(filtered_points)
        return list(filtered_frame_data.values())

    def _build_multi_image_coordinates(
        self, rng, timestamps, frame_points, scales, multi_image, indices=None) -> str:
        on = 1
        frame_strs = []
        for i in range(len(frame_points)):
            points = frame_points[i]
            timestamp = timestamps[i]
            if multi_image:
                scale = scales[i]
            else:
                scale = scales
            frame_str = self.build_single_image_coordinates(rng, points, scale, on)
            on += len(points)
            if multi_image:
                prefix = str(timestamp)  # timestamp is a frame index
            else:
                prefix = self.format_output_timestamp(timestamp)
            frame_strs.append(f"{prefix} {frame_str}")
        return self.image_sep.join(frame_strs)

    def _scale_point(self, point, scale):
        if isinstance(scale, (tuple, list)):
            x_scale, y_scale = scale
        else:
            x_scale, y_scale = scale, scale
        x, y = float(point[0])/x_scale, float(point[1])/y_scale
        if not 0 <= x <= 1.0 and 0 <= y <= 1.0:
            logging.warning(f"Out of bound points {point}")
            x, y = (max(0, min(x, 1.0)), max(0, min(y, 1.0)))
        if self.coordinate_scale == "1000":
            return round(1000*float(x)), round(1000*float(y))
        elif self.coordinate_scale == "100.0":
            return round(100*float(x), 1), round(100*float(y), 1)
        else:
            raise NotImplementedError(self.coordinate_scale)

    def build_single_image_coordinates(
        self, rng, points: List, scale: Union[float, Tuple[float, float], None], idx_start=1) -> str:
        """Build coordinates for a single image"""
        scaled_points = [self._scale_point(point, scale) for point in points]
        # Sort after rounding and scaling to be 100% sure sort order matches text values.
        # Otherwise, we can get subtle errors when points become tied on a value after rounding
        # that they were not tied on before rounding
        if self.sort_order == "xy":
            ix = sorted(range(len(scaled_points)), key=lambda x: scaled_points[x])
        elif self.sort_order == "random":
            ix = rng.permutation(len(scaled_points))
        else:
            raise NotImplementedError(self.sort_order)
        scaled_points = [scaled_points[i] for i in ix]
        idx = range(idx_start, idx_start+len(scaled_points))
        return self._format_image_points(idx, scaled_points)

    def _format_image_points(self, idx, scaled_points, occluded=None) -> str:
        """Convert image points that are already sorted and scaled into a string """
        if self.coordinate_scale == "1000":
            text_format = "%03d"
        elif self.coordinate_scale == "100.0":
            text_format = "%0.1f"
        else:
            raise NotImplementedError(self.coordinate_scale)
        norm_points = [[text_format%x, text_format%y] for x, y in scaled_points]
        if occluded:
            occluded = [occluded[i] for i in idx]
            return " ".join([
                f"{idx} {x} {y} {'y' if occ else 'n'}" for idx, (x, y), occ in zip(idx, norm_points, occluded)
            ])
        else:
            return " ".join([
                f"{idx} {x} {y}" for idx, (x, y) in zip(idx, norm_points)
            ])

    def build_point_str_from_coord_str(
        self, label: str, alt_text: Optional[str], coord_str: str, is_track: bool) -> str:
        """Building pointing string using the coordinate string"""
        if not coord_str:
            return ""
        if self.point_str_format == "tab_sep":
            alt = "-" if (alt_text is None or alt_text == label) else alt_text
            return f"<points {label}\t{alt}\t{coord_str}/>"
        elif self.point_str_format == "coord_str":
            prefix = "tracks" if is_track else "points"
            text = f"<{prefix}"
            if alt_text is not None:
                text += f" alt=\"{alt_text}\""
            text += f" coords=\"{coord_str}\""
            return text + ">" + label + f"</{prefix}>"
        else:
            raise NotImplementedError(self.point_str_format)

    def build_point_output(self, point_str: str, count: int, mode: str="point") -> str:
        """Building pointing output using the pointing string"""
        if count == 0:
            return "There are none."
        elif mode in ["point_then_count", "point_count"]:
            return f"Counting the {point_str} shows a total of {count}."
        elif mode in ["count_then_point", "count_point"]:
            return f"There are {count} {point_str}."
        elif mode == "count":
            return str(count)
        elif mode is None or mode in ["point", "pointing"]:
            return point_str
        else:
            raise NotImplementedError(mode)


"""
Below is the old legacy pointing code, its a bit of mess but we keep it around
for backwards-compatibility with older models
"""

def normalize_points(points, scale, decimal_places=1) -> list:
    """
    Convert absolute pixel coordinates to normalized coordinates.

    Args:
          points: Points to normalize (numpy array or list)
          scale: Scale factor - can be single value or [x_scale, y_scale]
          decimal_places: Number of decimal places for rounding

      Returns:
          List of normalized and rounded points in: [[x1, y1], [x2, y2], ...] or [x, y] format
    """
    points = np.array(points)
    if len(points) == 0:
        return []

    if isinstance(scale, (tuple, list)):
        points = points / np.array(scale)[None, :]
    else:
        points = points * (100/scale)

        # Round to specified decimal places
    if points.ndim == 1:
        return [round(x, decimal_places) for x in points.tolist()]
    else:
        return [[round(x, decimal_places) for x in point] for point in points.tolist()]


def seconds_to_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60  # Keep decimal
    # keep two decimal places for seconds by default
    formatted = f"{hours:02}:{minutes:02}:{seconds:05.2f}"
    return formatted


def format_time(time_value, format="seconds"):
    """
    TODO [QUESTION]: Merge with `format_timestamps` in DataFormatter?
    Format time value for model input/output.

    Args:
        time_value: Time in various formats (string "MM:SS.FF", float, int)
        format: "seconds" -> "2.50", "timestamp" -> "00:02.50"

    Returns:
        Formatted time string
    """
    # Parse input to float if needed
    if isinstance(time_value, str):
        if ':' in time_value:  # MM:SS.FF format
            try:
                time_obj = datetime.strptime(time_value, "%M:%S.%f")
                time_value = time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
            except ValueError:
                return time_value  # Return as-is if can't parse
        else:
            time_value = float(time_value)

    time_value = float(time_value)

    if format == "seconds": # e.g. "2.50"
        return f"{time_value:.2f}"
    elif format == "timestamp": # e.g. "00:02.50"
        return seconds_to_timestamp(time_value)
    else:
        raise ValueError(f"Unknown format: {format}")


@dataclasses.dataclass
class PointFormattingV1(PointFormatter):
    pointing_format: str = "default"
    points_decimal_places: int = 1

    @classmethod
    def extract_multi_image_points(cls, text, video_w, video_h):
        try:
            all_triplets = []
            # Find each <points ... /> block
            blocks = re.findall(r"<points\s+([^/>]*?)\s*/>", text)
            for block in blocks:
                tokens = block.strip().split()
                if not tokens:
                    continue
                # First token is the timestamp
                ts = float(tokens[0])
                # pts = []
                # Remaining tokens are repeating: counter, x, y
                # We ignore the counter (1, 2, ...) and only collect (x, y)
                triplet = tokens[1:]
                if len(triplet) % 3 != 0:
                    raise ValueError(f"Malformed points triplets for timestamp {ts}: {triplet!r}")
                for i in range(0, len(triplet), 3):
                    x = float(triplet[i + 1])
                    y = float(triplet[i + 2])
                    point = np.array([x, y])
                    if np.max(point) > 100:
                        # treat as invalid point
                        continue
                    if video_h is not None and video_w is not None:
                        point /= 100.0
                        point = point * np.array([video_w, video_h])
                    all_triplets.append((ts, point[0], point[1]))
            return all_triplets
        except Exception as e:
            return None

    @classmethod
    def extract_points(cls, text, image_w, image_h):
        all_points = []
        for match in re.finditer(r"Click\(([0-9]+\.[0-9]), ?([0-9]+\.[0-9])\)", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        if all_points:
            return all_points

        for match in re.finditer(r"[0-9]+ ([0-9]{3}) ([0-9]{3})", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 1000:
                    # Treat as an invalid output
                    continue
                point /= 1000.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        if all_points:
            return all_points

        for match in re.finditer(r"[0-9]+ ([0-9]+\.[0-9]) ([0-9]+\.[0-9])", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        if all_points:
            return all_points

        for match in re.finditer(r"\(([0-9]+\.[0-9]),? ?([0-9]+\.[0-9])\)", text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        for match in re.finditer(r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"', text):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        for match in re.finditer(r'(?:\d+|p)\s*=\s*([0-9]{3})\s*,\s*([0-9]{3})', text):
            try:
                point = [int(match.group(i)) / 10.0 for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                point = point * np.array([image_w, image_h])
                all_points.append(point)
        return all_points

    @classmethod
    def extract_trajectories(cls, text, width, height, video_fps, format=None):
        from olmo.eval.point_tracking_utils import PointTrackingParser
        from olmo.eval.object_tracking_utils import ObjectTrackingParser

        if format == "video_point_track_all_frames_with_occlusion":
            return PointTrackingParser.parse_prediction(text, width, height, video_fps, format=format)
        else:
            return ObjectTrackingParser.parse_prediction(text, width, height, video_fps, format=format)

    def _normalize_and_round_points(self, points, scale,):
        """Helper to normalize and round points."""
        norm_points = normalize_points(points, scale, self.points_decimal_places)

        # If originally single point and returned as 2D array, convert to 1D
        if np.array(points).ndim == 1 and np.array(norm_points).ndim == 2:
            return norm_points[0]

        return norm_points

    def _format_video_points_per_frame(self, points_dict, scale, skip_occluded=True):
        """
        Format points for a single frame in video as {id: [x, y], ...}
        Shared by:
            - `_format_video_point_track_per_frame`
            - `_format_video_point_track_all_frames_with_occlusion`

        Input:
            points_dict: {point_id: {"point": [x,y], "occluded": bool}}
            scale: scale factor for normalizing points
            skip_occluded: if True, skip occluded points
        """
        formatted = {}
        for obj_id, point_info in points_dict.items():
            occluded = point_info.get("occluded", False)
            if occluded:
                if skip_occluded:
                    continue
            point = self._normalize_and_round_points(point_info["point"], scale)
            formatted[obj_id] = (point, occluded)

        if len(formatted) == 0:
            return None

        formatted = dict(sorted(formatted.items(), key=lambda x: x[0]))

        if skip_occluded:  # {id: [x, y], ...} without quotes
            pairs = [f"{k}: [{v[0][0]}, {v[0][1]}]" for k, v in formatted.items()]
            return "{" + ", ".join(pairs) + "}"
        else:  # {id: [x, y, 'yes' if occluded], ...} without quotes
            pairs = [f"{k}: [{v[0][0]}, {v[0][1]}, yes]" if v[1]
                     else f"{k}: [{v[0][0]}, {v[0][1]}]"
                     for k, v in formatted.items()]
            return "{" + ", ".join(pairs) + "}"

    def _format_video_point_track_per_frame(self, frames_data, scale):
        """Format as: time {t}\n{id: [x, y], ...}"""
        output_parts = []

        for frame_data in frames_data:
            if not frame_data["points"]:
                continue

            frame_points_formatted: str = self._format_video_points_per_frame(frame_data["points"], scale)
            if frame_points_formatted:
                time_str = format_time(frame_data["time"], "seconds")
                output_parts.append(f"time {time_str}")
                output_parts.append(frame_points_formatted)

        return "\n".join(output_parts)

    def _format_video_point_start_end(self, frames_data, scale):
        """Format as: id: ([x1,y1,t1], [x2,y2,t2])"""
        # Group by object ID
        object_frames = {}
        for frame_data in frames_data:
            for obj_id, point_info in frame_data["points"].items():
                if obj_id not in object_frames:
                    object_frames[obj_id] = []

                time_str = format_time(frame_data["time"], "seconds")
                object_frames[obj_id].append({
                    'time': time_str,
                    'point': point_info["point"]
                })

        # Generate object-centric JSON output
        output_parts = []
        for obj_id, frames in object_frames.items():
            if len(frames) < 2:
                continue  # Need at least start and end

            frames.sort(key=lambda x: float(x['time']))
            first = frames[0]
            last = frames[-1]

            first_point = self._normalize_and_round_points(first['point'], scale)
            last_point = self._normalize_and_round_points(last['point'], scale)

            output_parts.append(
                f'{obj_id}: ([{first_point[0]}, {first_point[1]}, {first["time"]}], '
                f'[{last_point[0]}, {last_point[1]}, {last["time"]}])'
            )

        return "\n".join(output_parts)

    def _format_video_single_point_track(self, frames_data, scale):
        """Format as: [x1,y1,t1], [x2,y2,t2], ..."""
        output_parts = []

        # Skip first frame (initial point)
        for frame_data in frames_data[1:]:
            if not frame_data["points"]:
                continue

            point_info = frame_data["points"][0]  # Single point
            if point_info.get("occluded", False):
                continue

            time_str = format_time(frame_data["time"], "seconds")
            point = self._normalize_and_round_points(point_info["point"], scale)

            output_parts.append(f'[{point[0]}, {point[1]}, {time_str}]')

        return ", ".join(output_parts)

    def _format_video_point_track_all_frames_with_occlusion(self, frames_data, scale, initial_points):
        """Format as time {t}\n{id: [x, y, occluded], ...}"""

        point_id_mapping = {p['id']: i for i,p in enumerate(initial_points)}
        output_parts = []

        for frame_data in frames_data:
            if not frame_data["points"]:
                continue

            # Only include points that are in initial_points and after their initial frame
            # {point_idx: {"point": [x,y], "occluded": bool}}
            points_dict = {point_idx: frame_data["points"][point_id] for point_id, point_idx in point_id_mapping.items()
                           if frame_data["frame"] > initial_points[point_idx]["frame"] and point_id in frame_data["points"]}

            frame_points_formatted: str = self._format_video_points_per_frame(points_dict, scale, skip_occluded=False)
            if frame_points_formatted:
                time_str = format_time(frame_data["time"], "seconds")
                output_parts.append(f"time {time_str}")
                output_parts.append(frame_points_formatted)

        return "\n".join(output_parts)

    def points_to_text(self, points, scale, label_text, alt_text):
        if isinstance(scale, (tuple, list)):
            points = points / np.array(scale)[None, :]
        else:
            points = points * (100/scale)
        points = [[round(x, 1), round(y, 1)] for x, y in points]
        points.sort(key=lambda x: x[0]*10000 + x[1])
        if self.pointing_format == "compact_v1":
            label_text = label_text.replace("\t", " ")
            if alt_text is None or alt_text == label_text:
                alt_text = "-"
            else:
                alt_text = alt_text.replace("\t", " ")
            return f"<points {label_text}\t{alt_text}\t" + " ".join(f"{ix+1} {x:0.1f} {y:0.1f}" for ix, (x, y) in enumerate(points)) + "/>"
        elif self.pointing_format == "compact_v2":
            # Using 0-999 instead of 0-99 to save two tokens on the decimal points
            label_text = label_text.replace("\t", " ")
            if alt_text is None or alt_text == label_text:
                alt_text = "-"
            else:
                alt_text = alt_text.replace("\t", " ")
            return f"<points {label_text}\t{alt_text}\t" + " ".join(f"{ix+1} {int(x*10):03d} {int(y*10):03d}" for ix, (x, y) in enumerate(points)) + "/>"
        elif self.pointing_format == "default":
            if len(points) == 1:
                x_str, y_str = points[0]
                return f"<point x=\"{x_str:0.1f}\" y=\"{y_str:0.1f}\" alt=\"{alt_text}\">{label_text}</point>"
            point_text = []
            for ix, (x, y) in enumerate(points, start=1):
                point_text.append(f"x{ix}=\"{x:0.1f}\"")
                point_text.append(f"y{ix}=\"{y:0.1f}\"")
            point_text = " ".join(point_text)
            return f"<points {point_text} alt=\"{alt_text}\">{label_text}</points>"
        else:
            raise NotImplementedError(self.pointing_format)

    def format_video_tracks(self, frames_data, scale, label, alt_text=None, rng=None,
                            start_end_only=False, single_point_track=False, from_initial_points=None):
        if len(frames_data) == 0:
            return "There are none."
        if isinstance(scale, (tuple, list)):
            scale = [x/100 for x in scale]
        if from_initial_points is not None:
            return self._format_video_point_track_all_frames_with_occlusion(frames_data, scale, from_initial_points)
        if start_end_only:
            return self._format_video_point_start_end(frames_data, scale)
        elif single_point_track:
            return self._format_video_single_point_track(frames_data, scale)
        else:
            return self._format_video_point_track_per_frame(frames_data, scale)

    def format_video_points(self, timestamps, points, scale, label, alt_text=None, mode="point_then_count", rng=None):
        if len(points) == 0:
            return "There are none."
        assert scale == 100
        points_dict = {}
        points_list = []
        pt_ix = 0
        for ix, points in enumerate(points):
            timestamp = timestamps[ix]
            if self.pointing_format == "default":
                point_text = f"<points t{ix+1}={timestamp}"
            elif self.pointing_format == "paragraph":
                point_text = f"{timestamp}\n"
            elif self.pointing_format == "compact_v1":
                point_text = f"<points {timestamp}\t"
            else:
                raise NotImplementedError(self.pointing_format)

            for x, y in points:
                if self.pointing_format == "json":
                    if timestamp not in points_dict:
                        points_dict[timestamp] = []
                    points_dict[timestamp].append((round(float(x), 1), round(float(y), 1)))
                elif self.pointing_format == "default":
                    point_text += f" x{pt_ix+1}={x:0.1f} y{pt_ix+1}={y:0.1f}"
                elif self.pointing_format == "paragraph":
                    point_text += f"{pt_ix+1} {x:0.1f} {y:0.1f}\n"
                elif self.pointing_format == "compact_v1":
                    point_text += f"{pt_ix+1} {x:0.1f} {y:0.1f} "
                else:
                    raise NotImplementedError(self.pointing_format)
                pt_ix += 1
            if self.pointing_format in ["default", "compact_v1"]:
                point_text = point_text.rstrip() + "/>"
                points_list.append(point_text)
            elif self.pointing_format == "paragraph":
                points_list.append(point_text)

        total_points = pt_ix
        if self.pointing_format == "json":
            point_text = json.dumps(points_dict)
        elif self.pointing_format in ["default", "paragraph", "compact_v1"]:
            point_text = " ".join(points_list)
            point_text = point_text.strip()
        else:
            raise NotImplementedError(self.pointing_format)

        if mode == "point_count":
            return f"The \"{label}\" are located at: {point_text}.\nCounting shows the total number is: {total_points}."
        elif mode == "count_point":
            return f"The total number of \"{label}\" is: {total_points}.\nThey are located at: {point_text}."
        elif mode == "count":
            return f"The total number of \"{label}\" is: {total_points}."
        elif mode in ["point", "pointing"]:
            return point_text
        else:
            raise NotImplementedError(mode)

    def format_image_points(self, points, scale, label, alt_text=None, mode="point_then_count", rng=None):
        if isinstance(scale, (tuple, list)):
            scale = [x/100 for x in scale]
        point_txt = self.points_to_text(points, scale, label, alt_text)
        if mode in ["point_count", "point_then_count"]:
            return f"Counting the {point_txt} shows a total of {len(points)}."
        else:
            return point_txt


LegacyPointFormatting = PointFormattingV1
