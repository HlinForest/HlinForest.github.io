"""Pure scene descriptions, independently loadable without OpenGL."""
def mat(symbol, values, role, focus=None, blocks=None, **metadata):
    return dict(symbol=symbol, values=values, role=role, focus=focus, blocks=blocks, **metadata)

def frame(key, title, tex, items, caption):
    return dict(key=key, title=title, tex=tex, items=items, caption=caption)
