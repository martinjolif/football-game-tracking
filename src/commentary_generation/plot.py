from supervision.draw.utils import draw_text
from supervision.geometry.core import Point
from supervision.draw.color import Color

def split_commentary_by_length(commentary: str, max_chars: int = 110) -> list[str]:
    """
    Split commentary into multiple lines, each not exceeding max_chars,
    breaking at word boundaries.

    Args:
        commentary: str - the raw commentary text
        max_chars: int - maximum characters per line

    Returns:
        List of lines (str)
    """
    words = commentary.split()
    lines = []
    current_line = ""

    for word in words:
        # Check if adding this word would exceed max_chars
        if len(current_line) + len(word) + 1 <= max_chars:
            # Add a space if the line is not empty
            if current_line:
                current_line += " "
            current_line += word
        else:
            # Finish the current line and start a new one
            if current_line:
                lines.append(current_line)
            current_line = word

    # Add the last line
    if current_line:
        lines.append(current_line)

    return lines


def draw_commentary(frame, commentary, start_xy=(50, 50), line_spacing=30):
    """
    Draws multiple lines of commentary on the image using sv.draw_text.

    Args:
        frame: np.ndarray – frame on which text will be drawn.
        commentary: str – commentary.
        start_xy: tuple[int,int] – starting position (x,y).
        line_spacing: int – vertical space between lines.
    """
    x0, y0 = start_xy
    commentary_lines = split_commentary_by_length(commentary)
    for idx, line in enumerate(commentary_lines):
        # Anchor point for the text (x, y)
        text_anchor = Point(x=x0, y=y0 + idx * line_spacing)

        # Draw text with background for readability
        frame = draw_text(
            scene=frame,
            text=line,
            text_anchor=text_anchor,
            text_color=Color.WHITE,
            background_color=Color.BLACK,  # background rectangle behind text
            text_scale=0.6,
            text_thickness=1,
            text_padding=8,
        )

    return frame






