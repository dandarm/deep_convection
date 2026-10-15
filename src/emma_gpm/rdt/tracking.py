"""Compact-mask overlap and deterministic primary lineage, frame by frame."""
import numpy as np


def associate(previous, current, labels, area, config):
    parents = {c: [] for c in current}
    children = {p: [] for p in previous}
    edges = {}
    height, width = labels.shape
    number_labels = int(labels.max()) + 1
    for p, before in previous.items():
        dy, dx = np.rint(before.get("motion", np.zeros(2))).astype(int)
        ys, xs = before["bbox"]
        y0, x0 = ys.start+dy, xs.start+dx
        y1, x1 = ys.stop+dy, xs.stop+dx
        cy0, cx0, cy1, cx1 = max(0, y0), max(0, x0), min(height, y1), min(width, x1)
        if cy0 >= cy1 or cx0 >= cx1:
            continue
        mask = before["mask"][cy0-y0:cy1-y0, cx0-x0:cx1-x0].copy()
        destination = labels[cy0:cy1, cx0:cx1]
        local_area = area[cy0:cy1, cx0:cx1]
        mask &= np.isfinite(local_area)
        translated_area = float(local_area[mask].sum())
        intersection = np.bincount(destination[mask], weights=local_area[mask], minlength=number_labels)
        for c in np.flatnonzero(intersection[1:])+1:
            denominator = min(translated_area, current[int(c)]["area_km2"])
            aor = float(intersection[c]/denominator) if denominator else 0.
            if aor >= config.min_aor:
                parents[int(c)].append(p)
                children[p].append(int(c))
                edges[p, int(c)] = dict(aor=aor, intersection_km2=float(intersection[c]))
    return parents, children, edges


def primary_lineage(previous, current, parents, children):
    primary, events, visited = {}, {}, set()
    for c in current:
        if c in visited:
            continue
        ps, cs, todo = set(), {c}, [c]
        while todo:
            cc = todo.pop()
            for p in parents[cc]:
                if p not in ps:
                    ps.add(p)
                    for child in children[p]:
                        if child not in cs:
                            cs.add(child)
                            todo.append(child)
        visited.update(cs)
        if not ps:
            events[c] = "new"
        elif len(cs) == len(ps) == 1:
            primary[c] = next(iter(ps))
            events[c] = "continuation"
        elif len(cs) == 1:
            primary[c] = max(ps, key=lambda p: (previous[p]["area_km2"], -p))
            events[c] = "merge_main"
        elif len(ps) == 1:
            p = next(iter(ps))
            main = max(cs, key=lambda cc: (current[cc]["area_km2"], -cc))
            primary[main] = p
            for cc in cs:
                events[cc] = "split_main" if cc == main else "split_new"
        else:
            for cc in cs:
                events[cc] = "complex"
    return primary, events


def update_motion(before, now, config):
    displacement = now["centroid"]-before["centroid"]
    if np.linalg.norm(displacement) <= config.max_motion_pixels:
        return .5*np.asarray(before.get("motion", np.zeros(2)))+.5*displacement
    return np.zeros(2)
