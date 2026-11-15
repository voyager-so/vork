"""Two 60-frame cards with a 12-frame dissolve, at 30000/1001 FPS."""
import json

FPS = "30000/1001"
LENGTH = 60
OVERLAP = 12


def seconds(frame):
    return f"{frame * 1001}/30000"


def scene():
    if LENGTH <= 0 or not 0 <= OVERLAP <= LENGTH:
        raise ValueError("require LENGTH > 0 and 0 <= OVERLAP <= LENGTH")
    cards = []
    for index, color in enumerate(["#2563eb", "#db2777"]):
        start = index * (LENGTH - OVERLAP)
        card = {"name": f"card-{index}", "active": [seconds(start), seconds(start + LENGTH)],
                "children": [{"type": "path", "d": "M0 0H640V360H0Z", "fill": color},
                             {"type": "path", "d": f"M{120 + index * 200} 110h200v140H{120 + index * 200}Z",
                              "fill": "#ffffff"}]}
        if index and OVERLAP:
            # The window starts the local clock at zero. Fade the incoming picture
            # over the opaque outgoing picture, reaching one when the overlap ends.
            card["animation"] = [{"tracks": [{"path": ["opacity"],
                "keys": [[seconds(0), 0], [seconds(OVERLAP), 1]], "interpolation": "LINEAR"}]}]
        cards.append(card)
    return {"version": 2, "composition": {"space": "2d", "viewBox": [0, 0, 640, 360]},
            "render": {"resolution": [640, 360], "fps": FPS,
                       "duration": seconds(2 * LENGTH - OVERLAP)}, "nodes": cards}


if __name__ == "__main__":
    print(json.dumps(scene(), indent=2))
