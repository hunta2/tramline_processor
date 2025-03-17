from typing import List, Set, Tuple, Any
from dataclasses import dataclass, field
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, GeometryCollection, Polygon
from shapely.ops import split
from gobbler.lib.tools.constants import GEOGRAPHIC, PREV_GEOMETRY
from loguru import logger


@dataclass
class TramlineProcessor:
    """A class that generates smoothed point bearings and flags deviations of points from the main tramline directions.
    Attributes:
        machine_data (gpd.GeoDataFrame): The input GeoDataFrame containing machine point data.
        angle_threshold (int): The threshold angle between points to break the polyline into segments. Default is 5.
        distance_threshold (int): The threshold distance between points to break the polyline into segments. Default is 3.
    """

    logger: Any
    machine_data: gpd.GeoDataFrame
    time_column: str
    angle_threshold: int = 5
    distance_threshold: int = 3
    lookahead: int = 3
    segments: List[LineString] = field(default_factory=list)
    final_segments: List[LineString] = field(default_factory=list)

    def calculate_bearing(self, point1: Point, point2: Point) -> float:
        delta_x = point2.x - point1.x
        delta_y = point2.y - point1.y
        angle = np.arctan2(delta_y, delta_x)
        bearing = np.degrees(angle)
        return bearing if bearing >= 0 else bearing + 360

    def adjust_distance_threshold(self):
        """Adjusts the distance threshold based on the median spatial distance between points."""
        utm_epsg = self.machine_data.estimate_utm_crs()
        self.machine_data.to_crs(utm_epsg, inplace=True)
        self.machine_data["distance"] = self.machine_data["geometry"].distance(self.machine_data["geometry"].shift())
        self.machine_data.to_crs(epsg=GEOGRAPHIC, inplace=True)
        median_distance = self.machine_data["distance"].median() + self.machine_data["distance"].std()
        if median_distance > self.distance_threshold:
            self.distance_threshold = median_distance

    def break_polyline(self, lines: List[LineString]) -> List[LineString]:
        """
        Breaks a list of LineStrings into multiple segments based on a threshold angle between points with a look-ahead mechanism.
        Args:
            lines (List[LineString]): The list of LineStrings to be broken.
        Returns:
            List[LineString]: A list of LineStrings representing the broken segments.
        """
        # Generate distance threshold based on median spatial distance between points
        self.adjust_distance_threshold()
        segments = []

        for line in lines:
            coords = list(line.coords)
            current_segment = [coords[0]]
            used_points = set([coords[0]])

            for i in range(1, len(coords)):
                point1 = Point(coords[i - 1])
                point2 = Point(coords[i])
                bearing = self.calculate_bearing(point1, point2)
                distance = point1.distance(point2)

                if distance > self.distance_threshold:
                    if len(current_segment) > 1:
                        segments.append(LineString(current_segment))
                    current_segment = [coords[i]]
                    used_points.add(coords[i])
                    continue

                if i > 1:
                    prev_bearing = self.calculate_bearing(Point(coords[i - 2]), point1)
                    if abs(bearing - prev_bearing) > self.angle_threshold:
                        # Check if skipping the current point results in a straighter line
                        if i + 1 < len(coords):
                            next_point = Point(coords[i + 1])
                            next_bearing = self.calculate_bearing(point1, next_point)
                            if abs(next_bearing - prev_bearing) <= self.angle_threshold:
                                continue  # Skip the current point

                        # Look ahead to see if the polyline can continue straight
                        lookahead_valid = True
                        for j in range(1, self.lookahead + 1):
                            if i + j < len(coords):
                                future_point = Point(coords[i + j])
                                future_bearing = self.calculate_bearing(point1, future_point)
                                future_distance = point1.distance(future_point)
                                if (
                                    abs(future_bearing - prev_bearing) > self.angle_threshold
                                    or future_distance > self.distance_threshold
                                ):
                                    lookahead_valid = False
                                    break
                        if not lookahead_valid:
                            if len(current_segment) > 1:
                                segments.append(LineString(current_segment))
                            current_segment = [coords[i - 1]]
                            used_points.add(coords[i - 1])

                if coords[i] not in used_points:
                    current_segment.append(coords[i])
                    used_points.add(coords[i])
            if len(current_segment) > 1:
                segments.append(LineString(current_segment))

        return segments

    def remove_line_crossings(self, segments: List[LineString]) -> List[LineString]:
        """Removes line crossings from a list of segments using sweep line algo"""
        events = []
        for i, segment in enumerate(segments):
            events.append((segment.bounds[0], "start", i, segment))
            events.append((segment.bounds[2], "end", i, segment))

        events.sort()

        active_segments = set()
        final_segments = []

        for event in events:
            x, event_type, i, segment = event
            if event_type == "start":
                for j in active_segments:
                    other_segment = segments[j]
                    if segment.crosses(other_segment):
                        split_segments = self.split_segment_at_intersection(segment, other_segment)
                        final_segments.extend(split_segments)
                        break
                else:
                    active_segments.add(i)
                    final_segments.append(segment)
            elif event_type == "end":
                if i in active_segments:
                    active_segments.remove(i)
                else:
                    # TODO: Add logging for this error
                    self.logger.debug(f"Warning: Trying to remove segment {i} which is not in active_segments")

        return final_segments

    def split_segment_at_intersection(self, segment: LineString, other_segment: LineString) -> List[LineString]:
        """Splits a segment at its intersection with another segment"""
        result = split(segment, other_segment)
        split_segments = [geom for geom in result.geoms if isinstance(geom, LineString)]
        return split_segments

    def time_gap(self) -> float:
        data_copy = self.machine_data.copy()
        data_copy = data_copy.sort_values(by=self.time_column).reset_index(drop=True)
        data_copy["time_diff_s"] = data_copy[self.time_column].diff().dt.total_seconds()
        return data_copy["time_diff_s"].median() + data_copy["time_diff_s"].mean()

    def generate_segments(self, data, time_column, time_gap) -> List[LineString]:
        data["delta_time"] = data[time_column].diff().dt.total_seconds().fillna(0)
        data["gap"] = data["delta_time"] > time_gap
        data["cluster"] = (data["gap"].cumsum() + 1).astype(int)

        # Make list of linestrings based on the cluster
        lines = (
            data.groupby("cluster")["geometry"]
            .apply(lambda x: LineString(x.tolist()) if len(x) > 1 else Point(x.tolist()[0]))
            .tolist()
        )

        # Separate LineStrings and Points
        line_strings = [line for line in lines if isinstance(line, LineString)]
        points = [point for point in lines if isinstance(point, Point)]

        # Create segments
        segments = self.break_polyline(line_strings)
        segments.extend([LineString([point, point]) for point in points if isinstance(point, Point)])
        return segments

    def process_polyline(self) -> List[LineString]:
        """
        Generates linestring segments derived from machine point data. Processes the polyline by breaking it into segments based on the angle threshold, removing line crossings, and returning the final segments.
        Returns:
            List[LineString]: The final segments of the processed polyline.
        """

        MAX_ITERATIONS = 10  # Set a maximum number of iterations to prevent endless loop
        # TODO drop this as we can use the created iso time for this probably
        time_gap = self.time_gap()
        self.segments = self.generate_segments(self.machine_data, self.time_column, time_gap)
        if len(self.machine_data[self.time_column].unique()) < 2:
            logger.warning("Time column has less than 2 unique values, will try to sort data on objectid type")
            segments_index = self.segments
            original_index = self.machine_data.index
            self.machine_data = self.machine_data.sort_values(by=self.machine_data.columns[0]).reset_index(drop=True)
            time_gap = self.time_gap()
            segments_objid = self.generate_segments(self.machine_data, self.time_column, time_gap)
            if len(segments_objid) < len(segments_index):
                logger.info("Will sort data on left most coloumn")
                self.segments = segments_objid
            else:
                self.machine_data = self.machine_data.loc[original_index].reset_index(drop=True)

        iteration_count = 0

        while iteration_count < MAX_ITERATIONS:
            new_segments = self.remove_line_crossings(self.segments)
            if len(new_segments) == len(self.segments):
                break
            self.segments = new_segments
            iteration_count += 1

        # Additional processing to remove line crossings
        no_cross_segments = []
        cross_segments = []
        for segment in self.segments:
            has_cross = False
            for other_segment in self.segments:
                if segment != other_segment and segment.crosses(other_segment):
                    cross_segments.append(segment)
                    has_cross = True
                    break
            if not has_cross:
                no_cross_segments.append(segment)

        self.final_segments = no_cross_segments
        self.final_segments.extend(cross_segments)
        self.final_segments = self.remove_line_crossings(self.final_segments)
        return self.final_segments

    def normalize_bearings(self):
        self.machine_data["smoothed_bearing"] = self.machine_data["smoothed_bearing"] % 360

    def assign_bearing_to_segments(self) -> gpd.GeoDataFrame:
        """
        Assigns bearing values to each segment in the final_segments list and merges them with the existing GeoDataFrame.
        Returns:
            gpd.GeoDataFrame: The updated GeoDataFrame with bearing values assigned to each segment.
        """

        segment_bearings = []
        point_bearing_map = {}

        for segment_id, segment in enumerate(self.final_segments):
            segment_points = list(segment.coords)
            segment_bearing = self.calculate_bearing(Point(segment_points[0]), Point(segment_points[-1]))
            for point in segment_points:
                if point in point_bearing_map:
                    # Compare segment lengths and keep the bearing from the longer segment
                    existing_segment_id = point_bearing_map[point][1]
                    existing_segment = self.final_segments[existing_segment_id]
                    if segment.length > existing_segment.length:
                        point_bearing_map[point] = (segment_bearing, segment_id)
                else:
                    point_bearing_map[point] = (segment_bearing, segment_id)

        for point, (bearing, segment_id) in point_bearing_map.items():
            segment_bearings.append((point[0], point[1], bearing))

        return self.geometry_process_segment(segment_bearings, "smoothed_bearing")

    def flag_deviations(self, main_directions: np.ndarray, tolerance: int = 10) -> gpd.GeoDataFrame:
        """
        Flags deviations in the GeoDataFrame based on the given main directions and tolerance.
        Parameters:
            main_directions (np.ndarray): An array of main directions.
            tolerance (int, optional): The tolerance value for flagging deviations. Defaults to 10.
        Returns:
            gpd.GeoDataFrame: The GeoDataFrame with the flagged deviations.
        """
        self.machine_data["deviation"] = self.machine_data.apply(
            lambda row: (
                min(
                    abs(row["smoothed_bearing"] - main_directions[0]),
                    abs(row["smoothed_bearing"] - main_directions[1]),
                )
                > tolerance
                if row["smoothed_bearing"] is not None
                else False
            ),
            axis=1,
        )

    def identify_main_directions(self) -> gpd.GeoDataFrame:
        """
        Identifies the main directions in the GeoDataFrame.
        Returns:
            gpd.GeoDataFrame: The modified GeoDataFrame with flagged deviations.
        """

        num_bins = 36
        hist, bin_edges = np.histogram(self.machine_data["smoothed_bearing"], bins=num_bins, range=(0, 360))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        main_directions = bin_centers[np.argsort(hist)[-2:]]
        self.flag_deviations(main_directions)

    def assign_segment_ids(self) -> gpd.GeoDataFrame:
        """
        Assigns segment IDs to the points in the GeoDataFrame based on the final segments.
        Returns:
            GeoDataFrame: The modified GeoDataFrame with segment IDs assigned to the points.
        """

        segment_ids = []
        point_segment_map = {}

        for segment_id, segment in enumerate(self.final_segments):
            segment_points = list(segment.coords)
            for point in segment_points:
                if point in point_segment_map:
                    # Compare segment lengths and keep the point in the longer segment
                    existing_segment_id = point_segment_map[point]
                    existing_segment = self.final_segments[existing_segment_id]
                    if segment.length > existing_segment.length:
                        point_segment_map[point] = segment_id
                else:
                    point_segment_map[point] = segment_id

        for point, segment_id in point_segment_map.items():
            segment_ids.append((point[0], point[1], segment_id))

        return self.geometry_process_segment(segment_ids, "segment_id")

    def geometry_process_segment(self, segment_df, seg_column: str):
        bearings_df = pd.DataFrame(segment_df, columns=["x", "y", seg_column])
        self.machine_data["x"] = self.machine_data.geometry.apply(lambda geom: geom.x)
        self.machine_data["y"] = self.machine_data.geometry.apply(lambda geom: geom.y)
        self.machine_data = self.machine_data.merge(bearings_df, on=["x", "y"], how="left")
        self.machine_data.drop(columns=["x", "y"], inplace=True)

    def flag_small_islands(self, min_size: int = 20) -> pd.DataFrame:
        """
        Flags small islands in the GeoDataFrame based on the given minimum size.
        Args:
            min_size (int, optional): The minimum size of an island to be considered as small. Defaults to 20.
        Returns:
            pd.DataFrame: The modified GeoDataFrame with the flagged small islands removed.
        """
        if self.machine_data["distance"].median() > 2:
            min_size /= self.machine_data["distance"].median()
        self.machine_data["is_small_island"] = (
            self.machine_data.groupby("segment_id")["segment_id"].transform("size") <= min_size
        )

        self.machine_data["is_small_island_no_deviation"] = (
            self.machine_data.groupby(["segment_id", "deviation"])["segment_id"].transform("size") <= min_size
        ) & ~self.machine_data["deviation"]
        # TODO check if a nan segment id is a problem or not
        self.machine_data["is_segment_id_nan"] = self.machine_data["segment_id"].isna()

        self.machine_data["tramline_deviation"] = (
            self.machine_data["deviation"]
            | self.machine_data["is_small_island_no_deviation"]
            | self.machine_data["is_segment_id_nan"]
        )
        self.machine_data.drop(
            columns=[
                PREV_GEOMETRY,
                "is_small_island",
                "is_small_island_no_deviation",
                "is_segment_id_nan",
                "deviation",
                "segment_id",
                "gap",
                "cluster",
                "delta_time",
                "distance",
            ],
            inplace=True,
        )

    def run(self):
        self.logger.info("Tramline-processor: Processing polyline")
        self.process_polyline()
        self.logger.info("Tramline-processor: Assign bearing to segments")
        self.assign_bearing_to_segments()
        self.logger.info("Tramline-processor: Normalizing bearings")
        self.normalize_bearings()
        self.logger.info("Tramline-processor: Identifying main directions")
        self.identify_main_directions()
        self.logger.info("Tramline-processor: Segmenting points")
        self.assign_segment_ids()
        self.logger.info("Tramline-processor: flagging small islands")
        self.flag_small_islands()
        return self.machine_data


def hexagon(x_center: float, y_center: float, size: int) -> List[Tuple]:
    """
    Generates the coordinates of a hexagon.
    """
    angle = np.linspace(0, 2 * np.pi, 7)
    x = x_center + size * np.cos(angle)
    y = y_center + size * np.sin(angle)
    return list(zip(x, y))


def create_hexagon_tessellation(gdf: gpd.GeoDataFrame, hex_width: int) -> gpd.GeoDataFrame:
    """
    Generates a hexagonal tessellation from the bounding box of the input data.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame.
        hex_width (float): The width of the hexagons in the tessellation.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the hexagons.
    """
    # Calculate the radius of the hexagon
    hex_radius = hex_width / 2
    horiz_spacing = hex_radius * 3 / 2
    vert_spacing = hex_radius * np.sqrt(3)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = maxx - minx
    height = maxy - miny
    num_cols = int(width / horiz_spacing) + 2
    num_rows = int(height / vert_spacing) + 2
    tessellation = []
    for row in range(num_rows):
        for col in range(num_cols):
            x = minx + col * horiz_spacing
            y = miny + row * vert_spacing
            # Offset every other column
            if col % 2 == 1:
                y += vert_spacing / 2

            hex_coords = hexagon(x, y, hex_radius)
            tessellation.append({"x": x, "y": y, "coords": hex_coords})
    hexagons = [Polygon(hex["coords"]) for hex in tessellation]
    return gpd.GeoDataFrame({"geometry": hexagons}, crs=gdf.crs)


def spatial_neighborhood_cleaning(gdf: gpd.GeoDataFrame, yield_col: str, hex_width: int = 10) -> pd.Series:
    """
    Identifies outliers in the GeoDataFrame based on a hexagonal tessellation.

    Args:
        gdf (GeoDataFrame): The input GeoDataFrame.
        hex_width (float): The width of the hexagons in the tessellation.
        yield_col (str): The column containing the yield data.

    Returns:
        pd.Series: A boolean Series indicating the outliers.
    """
    hex_grid = create_hexagon_tessellation(gdf, hex_width)
    joined_gdf = gpd.sjoin(gdf, hex_grid, how="left", predicate="within")

    # Ensure all rows from gdf are retained
    joined_gdf = joined_gdf.reset_index(drop=True)
    joined_gdf.index = gdf.index

    hex_aggregated = joined_gdf.groupby("index_right")[yield_col].agg(["mean", "std"]).reset_index()
    hex_grid = hex_grid.join(hex_aggregated.set_index("index_right"))
    joined_gdf = joined_gdf.merge(hex_grid[["mean", "std"]], left_on="index_right", right_index=True, how="left")

    # Fill NaN values for rows that did not match any hexagon
    joined_gdf["mean"] = joined_gdf["mean"].fillna(joined_gdf[yield_col].mean())
    joined_gdf["std"] = joined_gdf["std"].fillna(joined_gdf[yield_col].std())

    mask = (joined_gdf[yield_col] > joined_gdf["mean"] + 3 * joined_gdf["std"]) | (
        joined_gdf[yield_col] < joined_gdf["mean"] - 3 * joined_gdf["std"]
    )
    return mask
