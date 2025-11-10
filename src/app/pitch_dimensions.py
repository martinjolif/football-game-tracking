import math

class PitchDimensions:
    """
    Class representing the dimensions of a standard football pitch.
    https://publications.fifa.com/fr/football-stadiums-guidelines/technical-guideline/stadium-guidelines/pitch-dimensions-and-surrounding-areas/
    """
    def __init__(self):
        self.width = 68  # Width of the pitch in meters
        self.length = 105  # Length of the pitch in meters
        self.penalty_area_length = 16.5  # Length of the penalty area in meters
        self.penalty_area_width = 40.32  # Width of the penalty area in meters
        self.goal_area_length = 5.5  # Length of the goal area in meters
        self.goal_area_width = 18.32  # Width of the goal area in meters
        self.center_circle_radius = 9.15  # Radius of the center circle in meters
        self.penalty_circle_radius = 9.15 # Radius of the penalty circle in meters
        self.penalty_spot_distance = 11  # Distance from the goal line to the penalty spot in meters

    def get_vertices(self):
        """Return the vertices of the pitch as a list of (x, y) tuples."""
        return [
            (0, self.width), #1
            (0, (self.width + self.penalty_area_width) / 2), #2
            (0, (self.width + self.goal_area_width) / 2), #3
            (0, (self.width - self.goal_area_width) / 2), #4
            (0, (self.width - self.penalty_area_width) / 2), #5
            (0, 0), #6
            (self.goal_area_length, (self.width + self.goal_area_width) / 2), #7
            (self.goal_area_length, (self.width - self.goal_area_width) / 2), #8
            (self.penalty_spot_distance, self.width / 2), #9
            (self.penalty_area_length, (self.width + self.penalty_area_width) / 2), #10
            (self.penalty_area_length, self.width / 2 + math.sqrt((self.penalty_circle_radius ** 2) - (self.penalty_area_length - self.penalty_spot_distance) ** 2)), #11
            (self.penalty_area_length, self.width / 2 - math.sqrt((self.penalty_circle_radius ** 2) - (self.penalty_area_length - self.penalty_spot_distance) ** 2)), #12
            (self.penalty_area_length, (self.width - self.penalty_area_width) / 2), #13
            (self.length / 2 - self.center_circle_radius, self.width / 2), #14
            (self.length / 2, self.width), #15
            (self.length / 2, self.width / 2 + self.center_circle_radius),  # 16
            (self.length / 2, self.width / 2 - self.center_circle_radius), #17
            (self.length / 2, 0), #18
            (self.length / 2 + self.center_circle_radius, self.width / 2), #19
            (self.length - self.penalty_area_length, (self.width + self.penalty_area_width) / 2), #20
            (self.length - self.penalty_area_length, self.width / 2 + math.sqrt((self.penalty_circle_radius ** 2) - (self.penalty_area_length - self.penalty_spot_distance) ** 2)), #21
            (self.length - self.penalty_area_length, self.width / 2 - math.sqrt((self.penalty_circle_radius ** 2) - (self.penalty_area_length - self.penalty_spot_distance) ** 2)), #22
            (self.length - self.penalty_area_length, (self.width - self.penalty_area_width) / 2), #23
            (self.length - self.penalty_spot_distance, self.width / 2), #24
            (self.length - self.goal_area_length, (self.width + self.goal_area_width) / 2), #25
            (self.length - self.goal_area_length, (self.width - self.goal_area_width) / 2), #26
            (self.length, self.width), #27
            (self.length, (self.width + self.penalty_area_width) / 2), #28
            (self.length, (self.width + self.goal_area_width) / 2), #29
            (self.length, (self.width - self.goal_area_width) / 2), #30
            (self.length, (self.width - self.penalty_area_width) / 2), #31
            (self.length, 0), #32
        ]

    def get_edges(self):
        """
        Each tuple (a, b) represents an edge connecting vertex number a to vertex number b
        """
        return [
            (1, 2), (2, 3), (3, 4), (3, 7), (7, 8), (8, 4), (4, 5), (5, 6),
            (5, 13), (2, 10), (10, 11), (11, 12), (12, 13),
            (15, 16), (16, 17), (17, 18),
            (20, 21), (21, 22), (22, 23), (25, 26),
            (27, 28), (28, 29), (29, 30), (29, 25), (30, 31), (31, 32),
            (1, 15), (15, 27), (20, 28), (23, 31),
            (6, 18), (18, 32), (26, 30),
        ]

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Arc, Circle
    pitch = PitchDimensions()
    vertices = pitch.get_vertices()
    edges = pitch.get_edges()

    fig, ax = plt.subplots(figsize=(12, 7))

    # plot each edge provided by get_edges (1-based -> 0-based conversion)
    for start_vertex_num, end_vertex_num in edges:
        start_vertex_x, start_vertex_y = vertices[start_vertex_num - 1]
        end_vertex_x, end_vertex_y = vertices[end_vertex_num - 1]
        ax.plot([start_vertex_x, end_vertex_x], [start_vertex_y, end_vertex_y], '-k', linewidth=1)

    # display the vertices and numbers
    x_coordinates = [v[0] for v in vertices]
    y_coordinates = [v[1] for v in vertices]
    ax.scatter(x_coordinates, y_coordinates, c='red', s=30)
    for i, (x, y) in enumerate(vertices, start=1):
        ax.text(x, y, str(i), color='blue', fontsize=8, ha='right', va='bottom')

    # Central circle
    center = (pitch.length / 2, pitch.width / 2)
    ax.add_patch(Circle(center, pitch.center_circle_radius, fill=False, linewidth=1))


    def _angle_deg(cx, cy, px, py):
        """
        Compute the angle in degrees from center (cx, cy) to point (px, py).
        The angle is computed with `atan2(py - cy, px - cx)`, converted to degrees,
        and normalized to the range [0, 360).
        """
        deg = math.degrees(math.atan2(py - cy, px - cx))
        return deg % 360


    def _small_arc_angles(a1, a2):
        """
        Return the start and end angles (in degrees) defining the smaller arc between two angles.

        Both input angles are normalized to the range \[0, 360). The function computes the
        counter-clockwise (CCW) and clockwise (CW) spans between the two normalized angles
        and selects the smaller span. The returned pair (start, end) are degrees suitable
        for plotting routines that expect a start angle and an end angle
        """
        a1 = a1 % 360
        a2 = a2 % 360
        span_ccw = (a1 - a2) % 360  # trigonometric direction (ccw)
        span_cw = (a2 - a1) % 360  # opposite direction
        if span_cw <= span_ccw:
            return a1, a1 + span_cw  # keep the smallest arc
        else:
            return a2, a2 + span_ccw


    # Left penalty arc (between vertices 11 and 12 -> indices 10,11)
    left_center = (pitch.penalty_spot_distance, pitch.width / 2)
    left_arc_upper_vertex = vertices[10]
    left_arc_lower_vertex = vertices[11]
    left_arc_angle_upper = _angle_deg(left_center[0], left_center[1], left_arc_upper_vertex[0], left_arc_upper_vertex[1])
    left_arc_angle_lower = _angle_deg(left_center[0], left_center[1], left_arc_lower_vertex[0], left_arc_lower_vertex[1])
    theta1, theta2 = _small_arc_angles(left_arc_angle_upper, left_arc_angle_lower)
    ax.add_patch(Arc(left_center, 2 * pitch.penalty_circle_radius, 2 * pitch.penalty_circle_radius,
                     angle=0, theta1=theta1, theta2=theta2, linewidth=1))

    # Right penalty arc (between vertices 21 and 22 -> indices 20,21)
    right_center = (pitch.length - pitch.penalty_spot_distance, pitch.width / 2)
    right_arc_upper_vertex = vertices[20]
    right_arc_lower_vertex = vertices[21]
    right_arc_angle_upper = _angle_deg(right_center[0], right_center[1], right_arc_upper_vertex[0], right_arc_upper_vertex[1])
    right_arc_angle_lower = _angle_deg(right_center[0], right_center[1], right_arc_lower_vertex[0], right_arc_lower_vertex[1])
    theta1, theta2 = _small_arc_angles(right_arc_angle_upper, right_arc_angle_lower)
    ax.add_patch(Arc(right_center, 2 * pitch.penalty_circle_radius, 2 * pitch.penalty_circle_radius,
                     angle=0, theta1=theta1, theta2=theta2, linewidth=1))

    ax.set_aspect('equal', 'box')
    ax.set_title("Pitch vertices (edges + arcs)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    plt.tight_layout()
    plt.show()