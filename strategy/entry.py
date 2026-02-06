from config.settings import MIN_EDGE, MIN_VOLUME

def should_enter(edge, volume):
    return edge >= MIN_EDGE and volume >= MIN_VOLUME