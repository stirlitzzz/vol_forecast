"""
Project Scaffold: Quant Vol Forecast Pipeline
Directory structure (relative):

src/
└── __init__.py
└── load_data.py      # data I/O & alignment helpers
└── features.py       # HAR features & realized vol prep
└── signals.py        # signal construction & trade sizing

notebooks/
└── 01_vol_forecast.ipynb      # exploration / diagnostics
└── 02_implied_alignment.ipynb # IV cleaning / alignment
└── 03_signal_backtest.ipynb   # strategy evaluation

output/
└── forecast_matrix.parquet
└── iv_matrix.parquet
└── realized_future.parquet
"""