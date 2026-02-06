from config.settings import BANKROLL, MAX_TRADE_SIZE

def get_position_size():
    return min(MAX_TRADE_SIZE, BANKROLL * 0.1)