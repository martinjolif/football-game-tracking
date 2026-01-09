import math
from src.radar.pitch_dimensions import PitchDimensions

def assign_teams(players_xy, cluster_labels):
    players = []
    for i, pos in enumerate(players_xy):
        players.append({
            'team': cluster_labels[i],
            'x': pos[0],
            'y': pos[1]
        })
    return players

def get_teams_barycenter(players):
    team_positions = {}
    for player in players:
        team = player['team']
        if team not in team_positions:
            team_positions[team] = {'x_sum': 0, 'y_sum': 0, 'count': 0}
        team_positions[team]['x_sum'] += player['x']
        team_positions[team]['y_sum'] += player['y']
        team_positions[team]['count'] += 1

    barycenters = {}
    for team, data in team_positions.items():
        barycenters[team] = (data['x_sum'] / data['count'], data['y_sum'] / data['count'])
    return barycenters

def get_left_team(players):
    teams_barycenter = get_teams_barycenter(players)
    sorted_teams = sorted(
        teams_barycenter.items(),
        key=lambda item: item[1][0]  # x coordinate
    )
    left_team = sorted_teams[0][0]
    right_team = sorted_teams[1][0]
    return left_team, right_team, teams_barycenter

def get_ball_possessor(ball_xy, players_xy, radius=100):
    """
    Determines the index of the player closest to the ball within a certain radius.

    ball_xy: list or tuple [(x, y)] representing the ball position
    players_xy: list of tuples [(x, y)] representing player positions
    radius: maximum distance to consider a player as possessing the ball (in centimeters)

    index of the player or None if no player is close enough
    """
    ball_x, ball_y = ball_xy[0]
    min_dist = float('inf')
    possessor_idx = None
    for i, (x, y) in enumerate(players_xy):
        dist = math.sqrt((x - ball_x)**2 + (y - ball_y)**2)
        if dist < min_dist and dist <= radius:
            min_dist = dist
            possessor_idx = i
    return possessor_idx

def get_ball_displacement(current_xy, previous_xy):
    if previous_xy is None:
        return None

    dx = current_xy[0] - previous_xy[0]
    dy = current_xy[1] - previous_xy[1]

    return dx, dy

def get_ball_team_relative_direction(dx, possessing_team, left_team):
    if dx is None:
        return None
    # Forward/backward relative to team attack direction
    if possessing_team == left_team:
        return "forward" if dx > 0 else "backward" if dx < 0 else "static"
    else:
        return "forward" if dx < 0 else "backward" if dx > 0 else "static"


def get_field_zone_3x3(pos, pitch: PitchDimensions):
    """Divide pitch into 3 horizontal × 3 vertical zones (9 zones)."""
    x, y = pos

    # Horizontal thirds
    if x < pitch.length / 3:
        h_zone = "defensive third"
    elif x < 2 * pitch.length / 3:
        h_zone = "middle third"
    else:
        h_zone = "attacking third"

    # Vertical thirds
    if y < pitch.width / 3:
        v_zone = "left wing"
    elif y < 2 * pitch.width / 3:
        v_zone = "center"
    else:
        v_zone = "right wing"

    return f"{v_zone}-{h_zone}"

def generate_event(
        previous_ball_xy,
        ball_xy,
        players_xy,
        cluster_labels,
        left_team,
        right_team,
        teams_barycenter,
        pitch: PitchDimensions
):
    possessor_idx = get_ball_possessor(ball_xy, players_xy)
    if possessor_idx is not None:
        players = assign_teams(players_xy, cluster_labels)
        possessing_team = players[possessor_idx]['team']
        ball_zone = get_field_zone_3x3([ball_xy[0][0], ball_xy[0][1]], pitch)

        ball_relative_possessing_team_x = "ahead of" if ball_xy[0][0] > teams_barycenter[possessing_team][0] else "behind"
        ball_relative_unpossessing_team_x = "ahead of" if ball_xy[0][0] > teams_barycenter[left_team if possessing_team == right_team else right_team][0] else "behind"

        movement = None
        if previous_ball_xy is not None:
            displacement = get_ball_displacement(ball_xy[0], previous_ball_xy[0])
            movement = get_ball_team_relative_direction(displacement[0], possessing_team, left_team)

        event = ""
        event += "- The pitch is shown from a fixed, standard center TV broadcast angle.\n"
        event += f"- Team {left_team} is positioned on the left side of the image and attacks from left to right.\n"
        event += f"- Team {right_team} is positioned on the right side of the image and attacks from right to left.\n\n"

        event += "Pitch spatial structure:\n"
        event += f"- x-axis runs along the pitch length; x=0 is left, increasing to the right until x={pitch.length}.\n"
        event += f"- y-axis runs across the pitch width; y=0 is the bottom touchline and y={pitch.width} is the top touchline.\n"
        event += f"- Team {left_team} barycenter is located around {get_field_zone_3x3(teams_barycenter[left_team], pitch)}.\n"
        event += f"- Team {right_team} barycenter is located around {get_field_zone_3x3(teams_barycenter[right_team], pitch)}.\n\n"

        event += "Live match context:\n"
        event += f"- The ball is located in the {ball_zone}.\n"
        event += f"- Team {possessing_team} is in control of the ball.\n"
        if movement and movement != "static":
            event += f"- The ball is moving {movement} relative to the possessing team.\n"
        else:
            event += f"- The ball is static relative to the possessing team.\n"
        event += f"- The ball is {ball_relative_possessing_team_x} the possessing team’s average position.\n"
        event += f"- The ball is {ball_relative_unpossessing_team_x} the unpossessing team’s average position.\n\n"

        # TV-style commentary prompt embedded
        event += (
            "Commentary task:\n"
            "Provide a short, TV-style football commentary describing the scene exactly as it appears right now.\n"
            "Style and constraints:\n"
            "- Use natural, live broadcast language.\n"
            "- Use present tense only.\n"
            "- Describe only observable facts: ball location and which team has possession.\n"
            "- Do NOT invent passes, shots, movement, pressure, or intent.\n"
            "- Do NOT predict what will happen next.\n"
            "- Do NOT reinterpret pitch orientation or team directions.\n"
            "- Keep the commentary concise (1 sentence is preferred).\n"
        )

        return event
    else:
        return None