def calculate_edge(p_true: float, kalshi_price: float, fee_rate: float = 0.0) -> float:
    """
    Edge = p_true minus Kalshi break-even probability.
    Kalshi fee is charged on profit: fee = fee_rate * (1 - price).
    Break-even probability = price + fee_rate * (1 - price).
    """
    breakeven = kalshi_price + fee_rate * (1 - kalshi_price)
    return p_true - breakeven
