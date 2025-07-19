import pandas as pd
import numpy as np
import py_vollib.black_scholes
import py_vollib_vectorized


def simulate_ou_vol_matrix(
    dates, tickers, mean=0.2, vol=0.05, theta=0.1, dt=1 / 252, seed=None
):
    """
    Simulate a mean-reverting log-vol process across tickers and dates.
    Returns a DataFrame (dates × tickers).
    """
    if seed is not None:
        np.random.seed(seed)

    n_dates, n_tickers = len(dates), len(tickers)
    log_vols = np.zeros((n_dates, n_tickers))
    log_vols[0, :] = np.log(mean)

    for t in range(1, n_dates):
        dW = np.random.normal(0, 1, size=n_tickers)
        log_vols[t] = (
            log_vols[t - 1]
            + theta * (np.log(mean) - log_vols[t - 1]) * dt
            + vol * np.sqrt(dt) * dW
        )

    vols = np.exp(log_vols)
    return pd.DataFrame(vols, index=dates, columns=tickers)


def compute_realized_vol(price_df, window=5):
    """Compute realized volatility from price data"""
    price_df = price_df.astype(float)
    log_returns = np.log(price_df / price_df.shift(1))
    realized_vol = log_returns.rolling(window).std() * np.sqrt(252)  # annualized
    return realized_vol


def compute_straddle_price(spot, strike, ttm, r, q, sigma):
    """Compute straddle option prices using Black-Scholes"""
    # Flatten inputs for vectorized calculation
    spot_flat = spot.values.flatten()
    strike_flat = strike.values.flatten()
    sigma_flat = sigma.values.flatten()

    # Compute call and put prices
    call_prices = py_vollib.black_scholes.black_scholes(
        "c", spot_flat, strike_flat, ttm, r, sigma_flat, return_as="array"
    )
    put_prices = py_vollib.black_scholes.black_scholes(
        "p", spot_flat, strike_flat, ttm, r, sigma_flat, return_as="array"
    )

    # Return straddle price (call + put)
    return call_prices + put_prices


def compute_volatility_trading_pnl(
    trades,
    spot_prices,
    implied_vols,
    realized_vols,
    holding_period=5,
    r=0.0,
    q=0.0,
    slippage=0.002,
):
    """
    Compute PnL for volatility trading using straddles.

    Parameters:
    -----------
    trades : DataFrame
        Trade signals (-1: short straddle, 0: no position, 1: long straddle)
    spot_prices : DataFrame
        Spot prices for each date and ticker
    implied_vols : DataFrame
        Implied volatilities at trade entry
    realized_vols : DataFrame
        Realized volatilities at trade exit
    holding_period : int
        Number of days to hold positions
    r : float
        Risk-free rate
    q : float
        Dividend yield
    slippage : float
        Transaction cost as fraction of notional

    Returns:
    --------
    DataFrame with PnL for each date and ticker
    """
    # Ensure all inputs are aligned
    assert (
        trades.shape == spot_prices.shape == implied_vols.shape == realized_vols.shape
    )

    # Calculate option prices at entry and exit
    ttm = holding_period / 252  # Convert to years

    # Entry prices (using implied vol)
    entry_prices = compute_straddle_price(
        spot_prices, spot_prices, ttm, r, q, implied_vols
    ).reshape(spot_prices.shape)
    entry_prices = pd.DataFrame(
        entry_prices, index=spot_prices.index, columns=spot_prices.columns
    )

    # Exit prices (using realized vol, shifted forward by holding period)
    exit_spot = spot_prices.shift(-holding_period)
    exit_prices = compute_straddle_price(
        exit_spot, spot_prices, ttm, r, q, realized_vols.shift(-holding_period)
    ).reshape(spot_prices.shape)
    exit_prices = pd.DataFrame(
        exit_prices, index=spot_prices.index, columns=spot_prices.columns
    )

    # Calculate raw PnL: position * (exit_price - entry_price)
    raw_pnl = trades * (exit_prices - entry_prices)

    # Calculate transaction costs
    position_changes = trades.diff().abs().fillna(trades.abs())
    transaction_costs = position_changes * entry_prices * slippage

    # Net PnL
    net_pnl = raw_pnl - transaction_costs

    # Remove rows where we don't have exit prices (end of sample)
    return net_pnl.iloc[:-holding_period]


def compute_simple_vol_pnl(
    trades, implied_vols, realized_vols, holding_period=5, slippage=0.0
):
    """
    Simple PnL calculation using volatility spread (for comparison).
    This assumes you can trade volatility directly.
    """
    entry_vol = implied_vols.astype(float)
    exit_vol = realized_vols.shift(-holding_period).astype(float)

    raw_pnl = trades * (exit_vol - entry_vol)
    turnover = trades.diff().abs()
    cost = turnover * slippage
    net_pnl = raw_pnl - cost

    return net_pnl.iloc[:-holding_period]


def generate_sample_data(n_dates=100, n_tickers=10, seed=42):
    """Generate sample data for testing"""
    np.random.seed(seed)

    dates = pd.date_range("2023-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]

    # Generate price data
    price_matrix = pd.DataFrame(index=dates, columns=tickers)
    log_returns = np.random.normal(0, 1, size=(n_dates, n_tickers))
    stock_vols = np.random.uniform(0.15, 0.45, size=n_tickers)
    stock_vols_matrix = np.tile(stock_vols, (n_dates, 1))
    log_returns = log_returns * stock_vols_matrix / np.sqrt(252)
    initial_price = np.random.uniform(10, 100, size=n_tickers)
    initial_price_matrix = np.tile(initial_price, (n_dates, 1))
    price_matrix[:] = initial_price_matrix * np.exp(np.cumsum(log_returns, axis=0))

    # Generate volatility data
    realized_vols = compute_realized_vol(price_matrix, window=5)
    implied_vols = simulate_ou_vol_matrix(
        dates, tickers, mean=0.25, vol=0.05, seed=seed
    )

    # Generate trade signals
    trades = pd.DataFrame(index=dates, columns=tickers)
    trades[:] = np.random.randint(0, 3, size=(n_dates, n_tickers)) - 1

    return price_matrix, realized_vols, implied_vols, trades


if __name__ == "__main__":
    # Example usage
    price_matrix, realized_vols, implied_vols, trades = generate_sample_data()

    # Calculate PnL using both methods
    simple_pnl = compute_simple_vol_pnl(
        trades, implied_vols, realized_vols, holding_period=5
    )
    straddle_pnl = compute_volatility_trading_pnl(
        trades, price_matrix, implied_vols, realized_vols, holding_period=5
    )

    print("Simple vol PnL (first 5 rows):")
    print(simple_pnl.head())
    print("\nStraddle PnL (first 5 rows):")
    print(straddle_pnl.head())

    # Summary statistics
    print(
        f"\nSimple vol PnL - Total: {simple_pnl.sum().sum():.4f}, Mean: {simple_pnl.mean().mean():.4f}"
    )
    print(
        f"Straddle PnL - Total: {straddle_pnl.sum().sum():.4f}, Mean: {straddle_pnl.mean().mean():.4f}"
    )
