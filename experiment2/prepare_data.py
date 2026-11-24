"""Read LiTraj XYZ frames for training and offline scoring."""

import itertools
import math
import re


META = re.compile(rb'([\w.-]+)=(?:"([^"]*)"|([^\s"]+))')


def raw_frames(archive, member):
    with archive.open(member) as source:
        number, line = 0, 1
        while count_line := source.readline():
            n_atoms = int(count_line)
            header = source.readline()
            body = list(itertools.islice(source, n_atoms))
            if n_atoms <= 0 or len(body) != n_atoms or not header:
                raise ValueError(f"Incomplete frame at {member}:{line}")
            number += 1
            yield number, line, count_line, header, body
            line += n_atoms + 2


def metadata(header):
    return {key: quoted or bare for key, quoted, bare in META.findall(header)}


def labeled_frame(meta, body):
    props = meta[b"Properties"].split(b":")
    offset = 0
    force_start = None
    for key, kind, width in zip(props[::3], props[1::3], props[2::3]):
        width = int(width)
        if key == b"forces" and kind == b"R" and width == 3:
            force_start = offset
        offset += width
    if force_start is None:
        raise ValueError("Missing Cartesian force column")
    energy = float(meta[b"energy"])
    stress = [float(x) for x in meta[b"stress"].split()]
    if len(stress) != 9 or not all(map(math.isfinite, [energy, *stress])):
        raise ValueError("Invalid energy or stress")
    max_force = 0.0
    for atom in body:
        fields = atom.split()
        force = tuple(map(float, fields[force_start:force_start + 3]))
        if len(fields) != offset or not all(map(math.isfinite, force)):
            raise ValueError("Invalid atomic record or force")
        max_force = max(max_force, abs(force[0]), abs(force[1]), abs(force[2]))
    return energy, max_force
