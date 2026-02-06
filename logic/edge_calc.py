def calculate_edge(book_prob: float, kalshi_price: float, fee_rate: float = 0.0) -> float:
    """
    Edge = bookmaker implied win prob minus Kalshi break-even prob.
    Kalshi fee is charged on profit: fee = fee_rate * (1 - price).
    Break-even probability = price + fee_rate * (1 - price).
    """
    breakeven = kalshi_price + fee_rate * (1 - kalshi_price)
    return book_prob - breakeven
