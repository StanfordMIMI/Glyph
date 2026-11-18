"""Optional, explicitly named concept neighborhoods for a fixed layout."""
import math
from collections import Counter


def pack_regions(regions, nodes):
    lookup = {str(n.get('concept_id', n['name'])): n for n in nodes}
    names = Counter(n['name'] for n in nodes)
    for n in nodes:
        if names[n['name']] == 1:
            lookup.setdefault(str(n['name']), n)
    result = []
    for region in regions or []:
        label = str(region['label']).strip()
        members = list(dict.fromkeys(str(value) for value in region['concepts']))
        if not label or not members:
            raise ValueError('Regions require a label and at least one concept')
        unknown = set(members).difference(lookup)
        if unknown:
            raise ValueError(f'Region {label!r} references unknown concepts: {sorted(unknown)}')
        selected = [lookup[key] for key in members]
        # A small fixed margin also gives single-point and collinear regions area.
        points = sorted(set(
            (n['x'] + .024 * math.cos(i * math.pi / 4),
             n['y'] + .024 * math.sin(i * math.pi / 4))
            for n in selected for i in range(8)
        ))
        def cross(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        def half(values):
            hull = []
            for point in values:
                while len(hull) >= 2 and cross(hull[-2], hull[-1], point) <= 0:
                    hull.pop()
                hull.append(point)
            return hull[:-1]
        hull = half(points) + half(reversed(points))
        result.append(dict(label=label, members=[n['i'] for n in selected],
                           points=hull,
                           x=sum(n['x'] for n in selected)/len(selected),
                           y=min(p[1] for p in hull)-.012))
    return result
