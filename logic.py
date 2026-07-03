"""Model-backed move-selection logic for the Battlesnake.

The served policy uses a linear ranking model, scores each legal move,
and returns the highest-scoring direction. A compact heuristic remains as a
fallback so gameplay still returns a legal move if model scoring fails.

Board coordinates: ``(0, 0)`` is the bottom-left corner.
  up    -> y + 1
  down  -> y - 1
  left  -> x - 1
  right -> x + 1

Game-state schema reference: https://docs.battlesnake.com/api
"""

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

Point = Tuple[int, int]

DIRECTIONS: Dict[str, Point] = {
    "up": (0, 1),
    "down": (0, -1),
    "left": (-1, 0),
    "right": (1, 0),
}

# Penalty applied to a move that could lose a head-to-head collision.
HEAD_TO_HEAD_PENALTY = 10_000
# Below this health we start actively steering toward food.
HUNGRY_THRESHOLD = 50


def get_info() -> Dict[str, str]:
    """Appearance + metadata returned from ``GET /``."""
    return {
        "apiversion": "1",
        "author": "hackathon",
        "color": "#6434eb",
        "head": "smart-caterpillar",
        "tail": "weight",
        "version": "0.1.0",
    }


def choose_move(game_state: Dict) -> str:
    """Return the next move: time-boxed search, then model, then heuristic."""
    try:
        move = choose_move_search(game_state)
    except Exception:  # noqa: BLE001 - search must never break gameplay
        move = None
    if move is not None:
        return move
    try:
        move = choose_move_model(game_state)
    except Exception:  # noqa: BLE001 - a model issue must never break gameplay
        move = None
    if move is not None:
        return move
    return choose_move_heuristic(game_state)


def choose_move_heuristic(game_state: Dict) -> str:
    """Return the next move for the current turn."""
    board = game_state["board"]
    you = game_state["you"]
    width: int = board["width"]
    height: int = board["height"]

    head: Point = (you["head"]["x"], you["head"]["y"])
    my_length: int = you["length"]
    health: int = you["health"]

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]

    best_move = None
    best_score = float("-inf")

    for move, (dx, dy) in DIRECTIONS.items():
        nxt = (head[0] + dx, head[1] + dy)

        if not _in_bounds(nxt, width, height):
            continue
        if nxt in occupied:
            continue

        # Reachable open space from this cell. If we can't fit our own body in
        # the space we'd be moving into, we're about to trap ourselves.
        space = _flood_fill(nxt, occupied, width, height, limit=my_length + 1)
        score = float(space)

        if nxt in danger:
            score -= HEAD_TO_HEAD_PENALTY

        # When hungry, nudge toward the closest food.
        if foods and health < HUNGRY_THRESHOLD:
            nearest = min(_manhattan(nxt, f) for f in foods)
            score += (width + height - nearest) * 2

        if score > best_score:
            best_score = score
            best_move = move

    # No safe move found -> we're cornered. Move up and hope for the best.
    return best_move or "up"


def _occupied_cells(snakes: List[Dict]) -> Set[Point]:
    """All cells currently filled by any snake's body.

    We keep tails occupied too; they only free up *next* turn and treating them
    as solid is the conservative, safe choice for a base bot.
    """
    occupied: Set[Point] = set()
    for snake in snakes:
        for seg in snake["body"]:
            occupied.add((seg["x"], seg["y"]))
    return occupied


def _head_to_head_cells(snakes: List[Dict], my_id: str, my_length: int) -> Set[Point]:
    """Cells adjacent to enemy heads that are >= our length.

    Moving onto one of these risks a head-to-head collision we would lose or
    tie, so they are heavily penalized (but not forbidden — sometimes it's the
    only move).
    """
    danger: Set[Point] = set()
    for snake in snakes:
        if snake["id"] == my_id:
            continue
        if snake["length"] < my_length:
            continue
        ehead = (snake["head"]["x"], snake["head"]["y"])
        for dx, dy in DIRECTIONS.values():
            danger.add((ehead[0] + dx, ehead[1] + dy))
    return danger


def _flood_fill(start: Point, occupied: Set[Point], width: int, height: int, limit: int) -> int:
    """Count open cells reachable from ``start`` (capped at ``limit``).

    Used to avoid moves that would seal us into a small pocket.
    """
    seen: Set[Point] = {start}
    stack: List[Point] = [start]
    count = 0
    while stack:
        x, y = stack.pop()
        count += 1
        if count >= limit:
            break
        for dx, dy in DIRECTIONS.values():
            nbr = (x + dx, y + dy)
            if nbr in seen:
                continue
            if not _in_bounds(nbr, width, height):
                continue
            if nbr in occupied:
                continue
            seen.add(nbr)
            stack.append(nbr)
    return count


def _in_bounds(p: Point, width: int, height: int) -> bool:
    return 0 <= p[0] < width and 0 <= p[1] < height


def _manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# --- Paranoid minimax search with alpha-beta + iterative deepening ----------
#
# Board state is copy-on-write: bodies are tuples (head first), so building a
# child state after a turn is just constructing new tuples, no deep copies.
# Opponents outside ``_ACTIVE_RADIUS`` of our head are not branched over (that
# would blow up the branching factor with 3+ snakes); they're assigned a
# single greedy survival move instead. Nearby opponents are treated as fully
# adversarial (paranoid) so alpha-beta pruning stays valid.

_ACTIVE_RADIUS = 5
_MAX_SEARCH_DEPTH = 12
_TIME_SAFETY_MARGIN_MS = 120


class _SearchTimeout(Exception):
    pass


def _build_sim(game_state: Dict) -> Dict:
    board = game_state["board"]
    food = frozenset((f["x"], f["y"]) for f in board["food"])
    snakes = {
        s["id"]: {"body": tuple((seg["x"], seg["y"]) for seg in s["body"]), "health": s["health"]}
        for s in board["snakes"]
    }
    return {
        "width": board["width"],
        "height": board["height"],
        "food": food,
        "snakes": snakes,
        "my_id": game_state["you"]["id"],
    }


def _legal_moves_sim(body: Tuple[Point, ...], width: int, height: int) -> List[str]:
    """Directions that stay on the board and don't reverse into our own neck."""
    head = body[0]
    neck = body[1] if len(body) > 1 else None
    moves = [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) != neck
    ]
    if moves:
        return moves
    # Fully boxed in (or neck check left nothing) -> any in-bounds move, else all.
    moves = [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
    ]
    return moves or list(DIRECTIONS.keys())


def _order_moves(sim: Dict, sid: str, legal_moves: List[str]) -> List[str]:
    """Cheap best-first ordering (more resulting open space first) to help pruning."""
    if len(legal_moves) <= 1:
        return legal_moves
    occupied: Set[Point] = set()
    for s in sim["snakes"].values():
        occupied.update(s["body"])
    head = sim["snakes"][sid]["body"][0]
    scored = []
    for move in legal_moves:
        dx, dy = DIRECTIONS[move]
        nxt = (head[0] + dx, head[1] + dy)
        space = _flood_fill(nxt, occupied, sim["width"], sim["height"], limit=8)
        scored.append((space, move))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [move for _, move in scored]


def _active_opponents(sim: Dict, my_id: str) -> List[str]:
    my_head = sim["snakes"][my_id]["body"][0]
    return [
        sid
        for sid, s in sim["snakes"].items()
        if sid != my_id and _manhattan(my_head, s["body"][0]) <= _ACTIVE_RADIUS
    ]


def _apply_moves(sim: Dict, moves: Dict[str, str]) -> Dict:
    """Resolve one simultaneous turn for every snake that has a chosen move."""
    width, height = sim["width"], sim["height"]
    snakes = sim["snakes"]

    new_heads: Dict[str, Point] = {}
    new_bodies: Dict[str, Tuple[Point, ...]] = {}
    ate: Dict[str, bool] = {}
    for sid, move in moves.items():
        body = snakes[sid]["body"]
        dx, dy = DIRECTIONS[move]
        new_head = (body[0][0] + dx, body[0][1] + dy)
        new_heads[sid] = new_head
        will_eat = new_head in sim["food"]
        ate[sid] = will_eat
        new_bodies[sid] = (new_head,) + (body if will_eat else body[:-1])

    ids = list(new_bodies.keys())
    blocking: Set[Point] = set()
    for sid in ids:
        blocking.update(new_bodies[sid][1:])

    dead: Set[str] = set()
    for sid in ids:
        head = new_heads[sid]
        if not _in_bounds(head, width, height) or head in blocking:
            dead.add(sid)
    for sid in ids:
        if sid in dead:
            continue
        new_health = 100 if ate[sid] else snakes[sid]["health"] - 1
        if new_health <= 0:
            dead.add(sid)

    head_groups: Dict[Point, List[str]] = {}
    for sid in ids:
        head_groups.setdefault(new_heads[sid], []).append(sid)
    for head, group in head_groups.items():
        if len(group) < 2:
            continue
        lengths = {sid: len(new_bodies[sid]) for sid in group}
        max_len = max(lengths.values())
        survivors = [sid for sid in group if lengths[sid] == max_len]
        losers = group if len(survivors) != 1 else [sid for sid in group if sid != survivors[0]]
        dead.update(losers)

    new_food = set(sim["food"])
    for sid in ids:
        if ate[sid]:
            new_food.discard(new_heads[sid])

    new_snakes = {}
    for sid in ids:
        if sid in dead:
            continue
        new_health = 100 if ate[sid] else snakes[sid]["health"] - 1
        new_snakes[sid] = {"body": new_bodies[sid], "health": new_health}

    return {
        "width": width,
        "height": height,
        "food": frozenset(new_food),
        "snakes": new_snakes,
        "my_id": sim["my_id"],
    }


def _evaluate(sim: Dict, my_id: str) -> float:
    snakes = sim["snakes"]
    if my_id not in snakes:
        return -1_000_000.0

    width, height = sim["width"], sim["height"]
    me = snakes[my_id]
    my_head = me["body"][0]
    others = [sid for sid in snakes if sid != my_id]

    if not others:
        # Solo mode, or every opponent has been eliminated: still need to
        # avoid trapping ourselves, so score by self-reachable space.
        space = _flood_fill(my_head, set(me["body"]), width, height, limit=width * height)
        score = 1_000_000.0 + space + me["health"] * 0.1
        if sim["food"] and me["health"] < HUNGRY_THRESHOLD:
            nearest_food = min(_manhattan(my_head, f) for f in sim["food"])
            score -= nearest_food * 3.0
        return score

    occupied: Set[Point] = set()
    for s in snakes.values():
        occupied.update(s["body"])

    enemy_heads = [snakes[sid]["body"][0] for sid in others]
    my_dist = _bfs_dist([my_head], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height)
    voronoi = sum(1 for cell, d in my_dist.items() if d < enemy_dist.get(cell, _BIG))

    length_adv = len(me["body"]) - max(len(snakes[sid]["body"]) for sid in others)

    score = voronoi * 4.0 + length_adv * 20.0 + me["health"] * 0.5
    if sim["food"]:
        nearest_food = min(_manhattan(my_head, f) for f in sim["food"])
        score -= nearest_food * (3.0 if me["health"] < HUNGRY_THRESHOLD else 0.1)
    return score


def _search(
    sim: Dict,
    depth: int,
    alpha: float,
    beta: float,
    my_id: str,
    deadline: float,
    want_move: bool = False,
):
    if time.monotonic() > deadline:
        raise _SearchTimeout()
    if my_id not in sim["snakes"]:
        return (None, -1_000_000.0) if want_move else -1_000_000.0
    if depth == 0:
        value = _evaluate(sim, my_id)
        return (None, value) if want_move else value

    # Opponents remaining on the board (empty in solo mode or once we've won
    # by elimination) - search still continues so we keep avoiding self-traps.
    others = [sid for sid in sim["snakes"] if sid != my_id]
    active = _active_opponents(sim, my_id)
    order = [my_id] + active

    def rec(i: int, moves_acc: Dict[str, str], alpha: float, beta: float) -> float:
        if time.monotonic() > deadline:
            raise _SearchTimeout()
        if i == len(order):
            full_moves = dict(moves_acc)
            for sid in others:
                if sid in full_moves:
                    continue
                passive_legal = _legal_moves_sim(sim["snakes"][sid]["body"], sim["width"], sim["height"])
                full_moves[sid] = _order_moves(sim, sid, passive_legal)[0]
            new_sim = _apply_moves(sim, full_moves)
            return _search(new_sim, depth - 1, alpha, beta, my_id, deadline)

        sid = order[i]
        legal = _order_moves(sim, sid, _legal_moves_sim(sim["snakes"][sid]["body"], sim["width"], sim["height"]))
        if sid == my_id:
            value = float("-inf")
            for move in legal:
                moves_acc[sid] = move
                value = max(value, rec(i + 1, moves_acc, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = float("inf")
            for move in legal:
                moves_acc[sid] = move
                value = min(value, rec(i + 1, moves_acc, alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

    if not want_move:
        return rec(0, {}, alpha, beta)

    legal = _order_moves(sim, my_id, _legal_moves_sim(sim["snakes"][my_id]["body"], sim["width"], sim["height"]))
    best_move, best_value = legal[0], float("-inf")
    for move in legal:
        value = rec(1, {my_id: move}, alpha, beta)
        if value > best_value:
            best_value, best_move = value, move
        alpha = max(alpha, best_value)
    return best_move, best_value


def choose_move_search(game_state: Dict) -> Optional[str]:
    """Iterative-deepening paranoid minimax, time-boxed to the move timeout."""
    start = time.monotonic()
    timeout_ms = game_state.get("game", {}).get("timeout", 500)
    budget = max(0.05, (timeout_ms - _TIME_SAFETY_MARGIN_MS) / 1000.0)
    deadline = start + budget

    sim = _build_sim(game_state)
    my_id = sim["my_id"]
    if my_id not in sim["snakes"]:
        return None

    legal = _legal_moves_sim(sim["snakes"][my_id]["body"], sim["width"], sim["height"])
    if not legal:
        return None
    if len(legal) == 1:
        return legal[0]

    best_move: Optional[str] = None
    depth = 1
    try:
        while depth <= _MAX_SEARCH_DEPTH:
            if time.monotonic() > deadline:
                break
            move, _value = _search(sim, depth, float("-inf"), float("inf"), my_id, deadline, want_move=True)
            best_move = move
            depth += 1
    except _SearchTimeout:
        pass

    return best_move


# --- Embedded model features -------------------------------------------------

_BIG = 10_000
_NEIGHBORS = ((0, 1), (0, -1), (-1, 0), (1, 0))


def _bfs_dist(sources, blocked, width, height):
    """Shortest free-cell distances from seed cells."""
    dist = {}
    dq = deque()
    for source in sources:
        if source not in dist:
            dist[source] = 0
            dq.append(source)
    while dq:
        x, y = dq.popleft()
        d = dist[(x, y)]
        for dx, dy in _NEIGHBORS:
            nb = (x + dx, y + dy)
            if 0 <= nb[0] < width and 0 <= nb[1] < height and nb not in blocked and nb not in dist:
                dist[nb] = d + 1
                dq.append(nb)
    return dist


def _candidate_features(state: Dict, move: str) -> Dict[str, float]:
    """Feature vector for playing ``move`` from ``state``. Assumes ``move`` is legal."""
    board = state["board"]
    you = state["you"]
    width, height = board["width"], board["height"]
    head = (you["head"]["x"], you["head"]["y"])
    my_length = you["length"]
    health = you["health"]

    dx, dy = DIRECTIONS[move]
    nxt = (head[0] + dx, head[1] + dy)

    occupied = _occupied_cells(board["snakes"])
    danger = _head_to_head_cells(board["snakes"], you["id"], my_length)
    foods = [(f["x"], f["y"]) for f in board["food"]]
    enemies = [s for s in board["snakes"] if s["id"] != you["id"]]
    enemy_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies]
    bigger_heads = [(s["head"]["x"], s["head"]["y"]) for s in enemies if s["length"] >= my_length]

    # Voronoi control: cells we reach strictly before any enemy.
    my_dist = _bfs_dist([nxt], occupied, width, height)
    enemy_dist = _bfs_dist(enemy_heads, occupied, width, height) if enemy_heads else {}
    voronoi = sum(1 for cell, md in my_dist.items() if md < enemy_dist.get(cell, _BIG))

    # Tail reachability is a useful anti-self-trap signal.
    my_tail = (you["body"][-1]["x"], you["body"][-1]["y"])
    reach = _bfs_dist([nxt], occupied - {my_tail}, width, height)
    reaches_tail = 1.0 if my_tail in reach else 0.0

    escape = sum(
        1
        for ddx, ddy in _NEIGHBORS
        if _in_bounds((nxt[0] + ddx, nxt[1] + ddy), width, height)
        and (nxt[0] + ddx, nxt[1] + ddy) not in occupied
    )

    nearest_now = min((_manhattan(head, f) for f in foods), default=_BIG)
    nearest_next = min((_manhattan(nxt, f) for f in foods), default=_BIG)
    hungry = health < HUNGRY_THRESHOLD

    return {
        "space_capped": float(_flood_fill(nxt, occupied, width, height, limit=my_length + 1)),
        "open_space": float(_flood_fill(nxt, occupied, width, height, limit=width * height)),
        "voronoi": float(voronoi),
        "reaches_tail": reaches_tail,
        "escape": float(escape),
        "h2h_danger": 1.0 if nxt in danger else 0.0,
        "near_bigger_head": float(min((_manhattan(nxt, h) for h in bigger_heads), default=width + height)),
        "near_enemy_head": float(min((_manhattan(nxt, h) for h in enemy_heads), default=width + height)),
        "wall_dist": float(min(nxt[0], width - 1 - nxt[0], nxt[1], height - 1 - nxt[1])),
        "food_score": float((width + height - nearest_next) * 2) if hungry and foods else 0.0,
        "food_delta": float(nearest_now - nearest_next) if foods else 0.0,
        "is_food": 1.0 if nxt in foods else 0.0,
        "dist_to_center": abs(nxt[0] - (width - 1) / 2) + abs(nxt[1] - (height - 1) / 2),
    }


# --- Model -----------------------------------------------------
# Embedded standardized linear model.

_MODEL: Dict = {
    "feature_names": [
        "space_capped",
        "open_space",
        "voronoi",
        "reaches_tail",
        "escape",
        "h2h_danger",
        "near_bigger_head",
        "near_enemy_head",
        "wall_dist",
        "food_score",
        "food_delta",
        "is_food",
        "dist_to_center",
    ],
    "mean": [
        7.357954545454546,
        100.9034090909091,
        48.26988636363637,
        0.9943181818181818,
        2.4431818181818183,
        0.04261363636363636,
        9.673295454545455,
        4.676136363636363,
        1.625,
        0.8920454545454546,
        0.14772727272727273,
        0.036931818181818184,
        5.056818181818182,
    ],
    "std": [
        3.5995966185276513,
        22.80542174802676,
        31.41119158524981,
        0.07516338951888041,
        0.6235520417417705,
        0.20198444088469822,
        7.9675173248507924,
        2.2532045017839604,
        1.3552297691803878,
        5.861056404757769,
        0.9449599886584031,
        0.18859442989548575,
        2.34451950177747,
    ],
    "coef": [
        0.00010539398521136327,
        -1.6778512168946185,
        80.89420182766183,
        9.793855564450467,
        0.7884630868036275,
        -11.025170822665032,
        -0.7981723553489,
        0.5410534990053248,
        1.5629078731518526,
        7.582325762611304,
        0.12463070008097832,
        0.21036618806863483,
        1.836259515524985,
    ],
    "intercept": 0.0,
    "top1_accuracy": 0.9928571428571429,
}


def choose_move_model(game_state: Dict) -> Optional[str]:
    """Score each legal move with the trained model; return the best.

    Returns ``None`` (so the caller falls back to the heuristic) if the model
    isn't available or the snake is trapped with no legal move.
    """
    legal = _legal_moves(game_state)
    if not legal:
        return None

    names = _MODEL["feature_names"]
    mean = _MODEL["mean"]
    std = _MODEL["std"]
    coef = _MODEL["coef"]
    intercept = _MODEL["intercept"]

    best_move, best_score = None, float("-inf")
    for move in legal:
        feats = _candidate_features(game_state, move)
        score = intercept
        for i, name in enumerate(names):
            z = (feats.get(name, 0.0) - mean[i]) / std[i] if std[i] else 0.0
            score += coef[i] * z
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def _legal_moves(game_state: Dict) -> List[str]:
    board = game_state["board"]
    width, height = board["width"], board["height"]
    head = (game_state["you"]["head"]["x"], game_state["you"]["head"]["y"])
    occupied = _occupied_cells(board["snakes"])
    return [
        move
        for move, (dx, dy) in DIRECTIONS.items()
        if _in_bounds((head[0] + dx, head[1] + dy), width, height)
        and (head[0] + dx, head[1] + dy) not in occupied
    ]
