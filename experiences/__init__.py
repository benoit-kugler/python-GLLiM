import logging

for name in ("matplotlib", "shapely", "PIL.PngImagePlugin", "matplotlib.font_manager"):
    logging.getLogger(name).setLevel(logging.WARNING)
