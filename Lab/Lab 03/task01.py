# 300% scaling matrix
S = [
    [3, 0],
    [0, 3]
]

def scale_and_center(x, y, cx, cy):
    # Move point relative to screen center
    dx = x - cx
    dy = y - cy

    # Apply scaling
    new_x = 3 * dx
    new_y = 3 * dy

    # Move back to screen coordinates
    return cx + new_x, cy + new_y

# Example screen center
cx, cy = 400, 300

# Example fingerprint point
x, y = 380, 290

print(scale_and_center(x, y, cx, cy))
