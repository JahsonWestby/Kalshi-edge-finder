def alert_edge(team, kalshi_price, book_prob, edge):
    print(
        f"[EDGE FOUND] {team} | "
        f"Kalshi YES: {kalshi_price:.2f} | "
        f"Book: {book_prob:.2f} | "
        f"Edge: {edge*100:.1f}%"
    )