import math


def recognize_and_correct(points, closed_hint=False):
    if len(points) < 3:
        return None, points

    shape, corrected = _detect_line(points)
    if shape:
        return shape, corrected

    shape, corrected = _detect_triangle(points)
    if shape:
        return shape, corrected

    shape, corrected = _detect_rectangle(points)
    if shape:
        return shape, corrected

    shape, corrected = _detect_ellipse(points)
    if shape:
        return shape, corrected

    return None, points


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _total_length(points):
    length = 0.0
    for i in range(1, len(points)):
        length += _distance(points[i - 1], points[i])
    return length


def _detect_line(points):
    if len(points) < 2:
        return None, points

    start = points[0]
    end = points[-1]
    direct = _distance(start, end)
    path = _total_length(points)

    if direct < 10:
        return None, points

    ratio = path / direct
    if ratio > 1.20:
        return None, points

    max_dev = _max_deviation_from_line(points, start, end)
    if max_dev > direct * 0.15 + 20:
        return None, points

    return "LINE", [start, end]


def _max_deviation_from_line(points, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1:
        return 0

    max_d = 0
    for p in points:
        t = max(0, min(1, ((p[0] - p1[0]) * dx + (p[1] - p1[1]) * dy) / length_sq))
        proj_x = p1[0] + t * dx
        proj_y = p1[1] + t * dy
        d = _distance(p, (proj_x, proj_y))
        if d > max_d:
            max_d = d
    return max_d


def _detect_triangle(points):
    if len(points) < 8:
        return None, points

    start = points[0]
    end = points[-1]
    closed_dist = _distance(start, end)
    path_len = _total_length(points)

    if path_len < 40:
        return None, points

    if closed_dist > path_len * 0.35:
        return None, points

    simplified = _ramer_douglas_peucker(points, epsilon=max(5, path_len * 0.05))

    if len(simplified) >= 3 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]

    if len(simplified) != 3:
        return None, points

    tri_points = [simplified[0], simplified[1], simplified[2], simplified[0]]
    return "TRIANGLE", tri_points


def _detect_ellipse(points):
    if len(points) < 8:
        return None, points

    start = points[0]
    end = points[-1]
    closed_dist = _distance(start, end)
    path_len = _total_length(points)

    if path_len < 40:
        return None, points

    if closed_dist > path_len * 0.35:
        return None, points

    simplified_check = _ramer_douglas_peucker(points, epsilon=max(5, path_len * 0.05))
    if len(simplified_check) >= 4 and simplified_check[0] == simplified_check[-1]:
        simplified_check = simplified_check[:-1]
    if 3 <= len(simplified_check) <= 5:
        return None, points

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)

    distances = [_distance(p, (cx, cy)) for p in points]
    mean_r = sum(distances) / len(distances)

    if mean_r < 10:
        return None, points

    variance = sum((d - mean_r) ** 2 for d in distances) / len(distances)
    std_r = math.sqrt(variance)

    if std_r / mean_r > 0.35:
        return None, points

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    rx = (max_x - min_x) / 2
    ry = (max_y - min_y) / 2

    if rx < 8 or ry < 8:
        return None, points

    ellipse_points = _generate_ellipse(cx, cy, rx, ry)
    return "ELLIPSE", ellipse_points


def _generate_ellipse(cx, cy, rx, ry, num_points=72):
    points = []
    for i in range(num_points + 1):
        angle = 2 * math.pi * i / num_points
        x = cx + rx * math.cos(angle)
        y = cy + ry * math.sin(angle)
        points.append((x, y))
    return points


def _detect_rectangle(points):
    if len(points) < 8:
        return None, points

    start = points[0]
    end = points[-1]
    closed_dist = _distance(start, end)
    path_len = _total_length(points)

    if path_len < 40:
        return None, points

    if closed_dist > path_len * 0.35:
        return None, points

    simplified = _ramer_douglas_peucker(points, epsilon=max(5, path_len * 0.05))

    if len(simplified) >= 4 and simplified[0] == simplified[-1]:
        simplified = simplified[:-1]

    if len(simplified) < 4:
        return None, points

    if len(simplified) > 6:
        return None, points

    angles = []
    for i in range(len(simplified)):
        p_prev = simplified[(i - 1) % len(simplified)]
        p_curr = simplified[i]
        p_next = simplified[(i + 1) % len(simplified)]

        v1 = (p_prev[0] - p_curr[0], p_prev[1] - p_curr[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        len1 = math.hypot(*v1)
        len2 = math.hypot(*v2)
        if len1 < 1 or len2 < 1:
            continue

        cos_a = (v1[0] * v2[0] + v1[1] * v2[1]) / (len1 * len2)
        cos_a = max(-1, min(1, cos_a))
        angle = math.degrees(math.acos(cos_a))
        angles.append(angle)

    if len(angles) != 4:
        return None, points

    right_angle_count = sum(1 for a in angles if 50 <= a <= 130)
    if right_angle_count < 3:
        return None, points

    corners = simplified[:4] if len(simplified) == 4 else _snap_to_rectangle(simplified)

    rect_points = []
    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        rect_points.append(p1)
    rect_points.append(corners[0])

    return "RECTANGLE", rect_points


def _snap_to_rectangle(corners):
    if len(corners) != 4:
        return corners

    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)

    return [
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
    ]


def _ramer_douglas_peucker(points, epsilon):
    if len(points) <= 2:
        return points[:]

    dmax = 0
    index = 0
    end = len(points) - 1

    for i in range(1, end):
        d = _perpendicular_distance(points[i], points[0], points[end])
        if d > dmax:
            dmax = d
            index = i

    if dmax > epsilon:
        left = _ramer_douglas_peucker(points[:index + 1], epsilon)
        right = _ramer_douglas_peucker(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [points[0], points[end]]


def _perpendicular_distance(point, line_start, line_end):
    dx = line_end[0] - line_start[0]
    dy = line_end[1] - line_start[1]
    length_sq = dx * dx + dy * dy
    if length_sq < 1:
        return _distance(point, line_start)

    t = max(0, min(1, ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy) / length_sq))
    proj_x = line_start[0] + t * dx
    proj_y = line_start[1] + t * dy
    return _distance(point, (proj_x, proj_y))


def _smooth_points(points, passes=2):
    if len(points) < 4:
        return points[:]

    result = [points[0]]
    for p in points[1:-1]:
        result.append(p)
    result.append(points[-1])

    for _ in range(passes):
        new_result = [result[0]]
        for i in range(1, len(result) - 1):
            new_x = result[i - 1][0] * 0.25 + result[i][0] * 0.5 + result[i + 1][0] * 0.25
            new_y = result[i - 1][1] * 0.25 + result[i][1] * 0.5 + result[i + 1][1] * 0.25
            new_result.append((new_x, new_y))
        new_result.append(result[-1])
        result = new_result

    return result
