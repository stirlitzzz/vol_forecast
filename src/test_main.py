# =================== Example pipeline script ==================
if __name__ == "__main__":
    # Lazy demo of end‑to‑end run
    realized = load_realized_vol()
    h1d, h1w, h1m = build_har_features(realized)

    # Naive forecast: simple linear combo (placeholder)
    forecast = 0.5 * h1d + 0.3 * h1w + 0.2 * h1m

    earnings = load_earnings()
    implied = load_implied_vol("../data/iv_raw.csv", earnings)

    signal = compute_signal_matrix(forecast, implied)
    positions = size_positions(signal, implied, method="risk", threshold=0.02)

    future_real = build_future_realized(realized, n=5)
    pnl = (future_real - implied) * positions  # simple PnL proxy

    print("Mean PnL:", np.nanmean(pnl))
