import logging

for name in ("matplotlib", "shapely"):
    logging.getLogger(name).setLevel(logging.WARNING)
