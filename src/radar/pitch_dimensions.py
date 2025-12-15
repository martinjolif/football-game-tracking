import math

class PitchDimensions:
    """
    Class representing the dimensions of a standard football pitch.
    https://publications.fifa.com/fr/football-stadiums-guidelines/technical-guideline/stadium-guidelines/pitch-dimensions-and-surrounding-areas/
    """
    def __init__(self):
        self.width = 6800  # Width of the pitch in centimeters
        self.length = 10500  # Length of the pitch in centimeters
        self.penalty_area_length = 1650  # Length of the penalty area in centimeters
        self.penalty_area_width = 4032  # Width of the penalty area in centimeters
        self.goal_area_length = 550  # Length of the goal area in centimeters
        self.goal_area_width = 1832  # Width of the goal area in centimeters
        self.center_circle_radius = 915  # Radius of the center circle in centimeters
        self.penalty_circle_radius = 915 # Radius of the penalty circle in centimeters
        self.penalty_spot_distance = 1100  # Distance from the goal line to the penalty spot in centimeters

        # edges is a list of (start_vertex_id, end_vertex_id) pairs using 1-based vertex IDs.
        self.edges = [
            (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
            (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
            (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
            (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
            (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
            (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
        ]

        # colors[i] gives the color for edge vertex[i].
        self.colors = [
            "#B0E0E6", "#B0E0E6", "#B0E0E6", "#B0E0E6", "#B0E0E6", "#B0E0E6",
            "#B0E0E6", "#B0E0E6", "#B0E0E6", "#B0E0E6", "#B0E0E6", "#B0E0E6",
            "#B0E0E6", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#0000FF",
            "#0000FF", "#0000FF", "#0000FF", "#0000FF", "#0000FF", "#0000FF",
            "#0000FF", "#0000FF", "#0000FF", "#0000FF", "#0000FF", "#0000FF",
            "#00BFFF", "#00BFFF"
        ]

        # labels[i] gives the official pitch vertex ID for vertices[i].
        # Non-sequential values (e.g., '14', '19' at the end) correspond to special pitch points
        # that are appended last in the vertices list for plotting convenience.
        self.labels = [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
            "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
            "14", "19"
        ]

    def penalty_arc_offset(self):
        """
        Compute the distance from the projection of the penalty spot on the penalty area to the circle arc vertices.
        """
        return math.sqrt(
            (self.penalty_circle_radius ** 2) -
            (self.penalty_area_length - self.penalty_spot_distance) ** 2
        )

    def get_vertices(self):
        """Return the vertices of the pitch as a list of (x, y) tuples."""
        return [
            (0, 0), #1
            (0, (self.width - self.penalty_area_width) / 2), #2
            (0, (self.width - self.goal_area_width) / 2), #3
            (0, (self.width + self.goal_area_width) / 2), #4
            (0, (self.width + self.penalty_area_width) / 2), #5
            (0, self.width), #6
            (self.goal_area_length, (self.width - self.goal_area_width) / 2), #7
            (self.goal_area_length, (self.width + self.goal_area_width) / 2), #8
            (self.penalty_spot_distance, self.width / 2), #9
            (self.penalty_area_length, (self.width - self.penalty_area_width) / 2), #10
            (self.penalty_area_length, self.width / 2 - self.penalty_arc_offset()), #11
            (self.penalty_area_length, self.width / 2 + self.penalty_arc_offset()), #12
            (self.penalty_area_length, (self.width + self.penalty_area_width) / 2), #13
            (self.length / 2, 0), #15
            (self.length / 2, self.width / 2 - self.center_circle_radius),  # 16
            (self.length / 2, self.width / 2 + self.center_circle_radius), #17
            (self.length / 2, self.width), #18
            (self.length - self.penalty_area_length, (self.width - self.penalty_area_width) / 2), #20
            (self.length - self.penalty_area_length, self.width / 2 - self.penalty_arc_offset()), #21
            (self.length - self.penalty_area_length, self.width / 2 + self.penalty_arc_offset()), #22
            (self.length - self.penalty_area_length, (self.width + self.penalty_area_width) / 2), #23
            (self.length - self.penalty_spot_distance, self.width / 2), #24
            (self.length - self.goal_area_length, (self.width - self.goal_area_width) / 2), #25
            (self.length - self.goal_area_length, (self.width + self.goal_area_width) / 2), #26
            (self.length, 0), #27
            (self.length, (self.width - self.penalty_area_width) / 2), #28
            (self.length, (self.width - self.goal_area_width) / 2), #29
            (self.length, (self.width + self.goal_area_width) / 2), #30
            (self.length, (self.width + self.penalty_area_width) / 2), #31
            (self.length, self.width), #32
            # The following two vertices correspond to pitch vertex IDs 14 and 19, not their list positions.
            (self.length / 2 - self.center_circle_radius, self.width / 2),  # 14
            (self.length / 2 + self.center_circle_radius, self.width / 2),  # 19
        ]
