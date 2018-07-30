import logging

for name in ("matplotlib", "shapely", "PIL.PngImagePlugin"):
    logging.getLogger(name).setLevel(logging.WARNING)
