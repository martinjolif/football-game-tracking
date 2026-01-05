import numpy as np
from src.radar.pitch_dimensions import PitchDimensions

def assign_teams(players_xy, cluster_labels):
    players = []
    for i, pos in enumerate(players_xy):
        players.append({
            'id': f'player_{i+1}',
            'team': cluster_labels[i],
            'x': pos[0],
            'y': pos[1]
        })
    return players

def get_possession(ball_xy, players):
    min_dist = float('inf')
    possessor = None
    for p in players:
        dist = np.linalg.norm(np.array([p['x'], p['y']]) - ball_xy)
        if dist < min_dist:
            min_dist = dist
            possessor = p
    return possessor

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
        v_zone = "left"
    elif y < 2 * pitch.width / 3:
        v_zone = "center"
    else:
        v_zone = "right"

    return f"{v_zone}-{h_zone}"

def nearby_players(possessor, players, radius=1000):
    """Return teammates and opponents near the possessor within a given radius (in centimeters)."""
    teammates = []
    opponents = []
    for p in players:
        if p['id'] == possessor['id']:
            continue
        dist = np.linalg.norm(np.array([p['x'], p['y']]) - np.array([possessor['x'], possessor['y']]))
        if dist <= radius:
            if p['team'] == possessor['team']:
                teammates.append(p)
            else:
                opponents.append(p)
    return teammates, opponents

def get_other_players_positions(possessor, players, pitch: PitchDimensions):
    """Describe roughly where other players are relative to the pitch zones."""
    other_positions = []
    for p in players:
        if p['id'] == possessor['id']:
            continue
        zone = get_field_zone_3x3((p['x'], p['y']), pitch)
        other_positions.append(f"{p['id']} (Team {p['team']}) in {zone}")
    return ", ".join(other_positions)

def generate_event(ball_xy, players_xy, cluster_labels, pitch: PitchDimensions):
    players = assign_teams(players_xy, cluster_labels)
    ball_possessor = get_possession(ball_xy=ball_xy, players=players)
    teammates, opponents = nearby_players(ball_possessor, players)
    ball_zone = get_field_zone_3x3((ball_possessor['x'], ball_possessor['y']), pitch)
    other_positions = get_other_players_positions(ball_possessor, players, pitch)

    event = f"{ball_possessor['id']} (Team {ball_possessor['team']}) has the ball in the {ball_zone}. \n \
            Nearby: {len(teammates)} teammates, {len(opponents)} opponents. \n\
            Others: {other_positions}."

    return event