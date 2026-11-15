"""Bake an underdamped spring and smoothstep easing into ordinary tracks."""
import json
import math

FPS = 30
FRAMES = 60


def spring(t, mass=1.0, stiffness=120.0, damping=12.0):
    """Unit step response, initially at rest; this example requires underdamping."""
    omega = math.sqrt(stiffness / mass)
    decay = damping / (2 * mass)
    frequency = math.sqrt(omega * omega - decay * decay)
    return 1 - math.exp(-decay * t) * (
        math.cos(frequency * t) + decay / frequency * math.sin(frequency * t))


def scene():
    def track(name, values):
        return {"path": ["nodes", name, "translation"], "interpolation": "LINEAR",
                "keys": [[f"{frame}/{FPS}", [40 + 440 * value, 0, 0]]
                         for frame, value in enumerate(values)]}
    return {"version": 2, "composition": {"space": "2d", "viewBox": [0, 0, 640, 360]},
            "render": {"resolution": [640, 360], "fps": FPS, "duration": f"{FRAMES}/{FPS}",
                       "background": "#142136"},
            "nodes": [{"name": name, "type": "path", "d": f"M0 {y}h100v70H0Z", "fill": color}
                      for name, y, color in [("spring", 80, "#fbbf24"), ("ease", 210, "#38bdf8")]],
            "animation": [{"tracks": [
                track("spring", [spring(f / FPS) for f in range(FRAMES + 1)]),
                track("ease", [(u := min(f / FPS, 1)) ** 2 * (3 - 2 * u)
                               for f in range(FRAMES + 1)])]}]}


if __name__ == "__main__":
    print(json.dumps(scene(), indent=2))
