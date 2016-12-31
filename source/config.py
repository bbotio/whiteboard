""" Configuration for sloth hand labeling tool
"""

LABELS = [
    {
        "text": "board",
        "inserter": "sloth.items.PolygonItemInserter",
        "item": "sloth.items.PolygonItem",
        "attributes": {
            "type": "poly",
            "class": "whiteboard",
            "id": ["black", "white"]
        }
    },
    {
        "text": "text",
        "inserter": "sloth.items.RectItemInserter",
        "item": "sloth.items.RectItem",
        "attributes": {
            "type": "rect",
            "class": "text",
            "id": ["text"]
        }
    },
    {
        "text": "figure",
        "inserter": "sloth.items.RectItemInserter",
        "item": "sloth.items.RectItem",
        "attributes": {
            "type": "rect",
            "class": "figure",
            "id": ["figure"]
        }
    }
]
