import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.io as pio
import logging
import sklearn as sk
import hmmlearn as hmm

class InsufficientCash(Exception):
    """Raised when the backtest runs out of cash."""
    pass

class Backtester:
    """
        A class to prepare data, generate positions, run a simple backtest,
        compute performance metrics, and plot results (static & interactive).
    """

    def __init__(self, df: pd.DataFrame, start_date: str, end_date: str, init_cash: float = 10_000, num_trades: int = 20):
        """
            df:           intraday-filtered DataFrame with a DatetimeIndex and columns
            start_date:   'YYYY-MM-DD' or datetime
            end_date:     'YYYY-MM-DD' or datetime
            init_cash:    starting cash balance
            num_trades:   number of non-zero trades per strategy
        """
        self.raw_df     = df.copy()
        self.start_date = pd.to_datetime(start_date)
        self.end_date   = pd.to_datetime(end_date)
        self.init_cash  = init_cash
        self.num_trades = num_trades

        # Strategies we're testing
        self.strats = df.columns[1:] # Exclude Index

        # Containers to be filled
        self.bt_df    = None
        self.metrics  = {}

        # For interactive plotting
        pio.renderers.default = 'notebook'

    @staticmethod
    def _generate_random_positions(size: int, num_trades: int) -> np.ndarray:
        """
            This function generates random positions. I have also provided a simple moving average (SMA) strategy in _generate_sma_positions() below, which the program is currently using.
        """
        if num_trades > size:
            raise ValueError("num_trades cannot exceed size")
        positions = np.zeros(size, dtype=int)
        idx = np.random.choice(size, size=num_trades, replace=False)
        positions[idx] = np.random.choice([-2, -1, 1, 2], size=num_trades)
        return positions
    
    def _generate_sma_positions(self, price: pd.Series) -> np.ndarray:
        """
            Simple 14/28 SMA-crossover strategy on a single price series.

            1. Compute moving averages:
                sma_short[t] = mean(price[t-50:t])
                sma_long[t]  = mean(price[t-200:t])

            2. Warm-up period:
            Until t = long_window-1 (here 27), we have no signal → pos=0.

            3. Signal rule for t >= long_window:
            if sma_short[t] > sma_long[t]:
                pos[t] = +1   (go long)
            elif sma_short[t] < sma_long[t]:
                pos[t] = -1   (go short)
            else:  # exact equality
                pos[t] = pos[t-1]

            4. We return a position array of +1/-1 (or 0 in warm-up).
            The backtester will then diff this to figure out trade sizes:
                trade_size[t] = pos[t] - pos[t-1]
        """
        short_w = 50
        long_w  = 200

        # 1. Compute rolling means
        sma_s = price.rolling(short_w, min_periods=1).mean()
        sma_l = price.rolling(long_w,  min_periods=1).mean()

        n = len(price)
        pos = np.zeros(n, dtype=int)

        # 2. No signal until both SMAs have "filled"
        #    i.e. t < long_w → pos[t]=0
        for t in range(long_w, n):
            if sma_s.iat[t] > sma_l.iat[t]:
                pos[t] =  1
            elif sma_s.iat[t] < sma_l.iat[t]:
                pos[t] = -1
            else:
                # flat on exact tie: carry forward prior position
                pos[t] = pos[t-1]

        return pos
    
##############################################
### DO NOT CHANGE ANYTHING ABOVE THIS LINE ### 
##############################################
    
    def _generate_positions(self, df: pd.DataFrame) -> np.ndarray:

        """
            TODO: 
            - Code your logic here. Note that you have access to price of a specific stock, which is a pandas series. 
            - You may want to look into previous returns, rolling average, price/return volatility and the like.
            - You may want to analyze using the original .csv file for any statistically significant patterns of each price time series, 
              and apply it in this function
        """            

        def buy_and_hold():
            n = len(df)
            pos_math = np.zeros(n, dtype=int)
            pos_stat = np.zeros(n, dtype=int)

            for i in range(n):
                pos_math[i] = 1
                pos_stat[i] = 1

            pos_math[n-1] = 0
            pos_stat[n-1] = 0

            symbols = {'MATH': pos_math, 'STAT': pos_stat}

            for ticker, position in symbols.items():
                df[f'Position_{ticker}'] = position

        def ewmac():
            """
            EWMAC (Exponential Weighted Moving Average Crossover) strategy using intraday prices.
            - Uses 200-period and 1000-period EMAs on 5-minute data
            - Fast EMA (200) vs Slow EMA (1000) crossover strategy
            """
            n = len(df)
            pos_stat = np.zeros(n, dtype=int)
            
            # Calculate EMAs on intraday 5-minute data
            ema_200 = df['STAT'].ewm(span=200).mean()   # Fast EMA (200 periods = ~15 hours)
            ema_1000 = df['STAT'].ewm(span=1000).mean() # Slow EMA (1000 periods = ~3.6 days)

            for i in range(n):
                if i < 1000:  # Wait for slow EMA to be meaningful
                    pos_stat[i] = 0  # Hold neutral during warmup
                else:
                    # Compare EMAs at this time step
                    if ema_200.iloc[i] > ema_1000.iloc[i]:
                        pos_stat[i] = 1   # Long signal (fast EMA above slow EMA)
                    elif ema_200.iloc[i] < ema_1000.iloc[i]:
                        pos_stat[i] = -1  # Short signal (fast EMA below slow EMA)
                    else:
                        pos_stat[i] = pos_stat[i-1] if i > 0 else 0  # Carry forward previous position

            symbols = {"STAT": pos_stat}

            for ticker, position in symbols.items():
                df[f"Position_{ticker}"] = position
            

        def trend_line_breakout():
            #72 hour lookback


            def check_trend_line(support: bool, pivot: int, slope: float, y: np.array):
                # compute sum of differences between line and prices, 
                # return negative val if invalid 
                # Find the intercept of the line going through pivot point with given slope
                intercept = -slope * pivot + y[pivot]
                line_vals = slope * np.arange(len(y)) + intercept
                diffs = line_vals - y
    
                # Check to see if the line is valid, return -1 if it is not valid.
                if support and diffs.max() > 1e-5:
                    return -1.0
                elif not support and diffs.min() < -1e-5:
                    return -1.0

                # Squared sum of diffs between data and line 
                err = (diffs ** 2.0).sum()
                return err
        



        def hidden_markov_model():
            """
            HMM-based regime signals for every strategy symbol in self.strats.

            Behaviour (simple and easy to read):
            - Uses a rolling window of past returns (window_size) to train a
              3-state Gaussian HMM. Training is performed once per calendar month
              (when we first encounter a new month in the index) using the
              most-recent `window_size` returns.
            - For each timestamp we compute the HMM posterior for the latest
              return. If the most-positive-state probability > most-negative-state
              probability -> +1, and vice-versa -> -1; otherwise 0.
            - If hmmlearn is unavailable or the model fails, we fall back to a
              simple tercile rule computed from the same rolling window.

            Outputs integer signals {-1, 0, 1} in columns named `Position_{symbol}`.
            """

            window_size = 150  # number of most-recent returns used for training
            n = len(df)

            # loop symbols and produce a position series per symbol
            for s in self.strats:
                price = df[s].ffill().bfill()
                rets = price.pct_change().fillna(0)

                pos = np.zeros(n, dtype=int)

                # keep a model reference and the month it was last trained on
                model = None
                last_model_month = None

                # We will compute training-state mean mapping on each retrain
                state_mean_map = None

                for i in range(n):
                    # current timestamp and return
                    ts = df.index[i]
                    cur_ret = rets.iat[i]
                    cur_month = ts.month

                    # retrain at month boundary once we have enough history
                    if i >= window_size and cur_month != last_model_month:
                        X = rets.iloc[i - window_size:i].values.reshape(-1, 1)
                        try:
                            # import locally so top-level import isn't required
                            from hmmlearn.hmm import GaussianHMM

                            model = GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=100)
                            model.fit(X)

                            # determine which HMM state is low/mid/high by mean return
                            states = model.predict(X)
                            state_means = {st: X[states == st].mean() for st in np.unique(states)}
                            # sort states by mean return: lowest -> highest
                            sorted_states = sorted(state_means, key=lambda k: state_means[k])
                            if len(sorted_states) == 3:
                                low_state, mid_state, high_state = sorted_states
                                state_mean_map = {'low': low_state, 'mid': mid_state, 'high': high_state}
                            else:
                                # unexpected: fallback to None to force tercile fallback later
                                model = None
                                state_mean_map = None

                        except Exception:
                            # any error using HMM -> disable model and fall back
                            model = None
                            state_mean_map = None

                        last_model_month = cur_month

                    # decide position using trained model (if available) or terciles
                    if model is not None:
                        try:
                            probs = model.predict_proba(np.array([[cur_ret]])) .flatten()
                            # safety: ensure we have a mapping from state index -> mean rank
                            if state_mean_map is not None:
                                # index of the high/low states
                                high_idx = int(state_mean_map['high'])
                                low_idx = int(state_mean_map['low'])

                                if probs[high_idx] > probs[low_idx]:
                                    pos[i] = 1
                                elif probs[low_idx] > probs[high_idx]:
                                    pos[i] = -1
                                else:
                                    pos[i] = 0
                            else:
                                # unexpected: fallback to 0
                                pos[i] = 0
                        except Exception:
                            # model prediction failed -> fallback to tercile rule below
                            model = None

                    if model is None:
                        # compute terciles on the same rolling window (or available history)
                        if i >= window_size:
                            window_vals = rets.iloc[i - window_size:i].values
                        else:
                            window_vals = rets.iloc[:i + 1].values

                        lo = np.nanpercentile(window_vals, 33) if len(window_vals) > 0 else 0.0
                        hi = np.nanpercentile(window_vals, 66) if len(window_vals) > 0 else 0.0

                        if cur_ret > hi:
                            pos[i] = 1
                        elif cur_ret < lo:
                            pos[i] = -1
                        else:
                            # carry forward prior non-na position where possible
                            pos[i] = pos[i - 1] if i > 0 else 0

                # write the integer positions back into the dataframe
                df[f'Position_{s}'] = pos.astype(int)   
                
                       
        def math_stat_soci_arbitrage():
            n = len(df)
            pos_soth = np.zeros(n, dtype=int)
            pos_math = np.zeros(n, dtype=int)
            pos_soci = np.zeros(n, dtype=int)

            df['MATH_Returns'] = df['MATH'].pct_change().fillna(0.0)
            df['SOCI_Returns'] = df['SOCI'].pct_change().fillna(0.0)
            df['SOTH_Returns'] = df['SOTH'].pct_change().fillna(0.0)
            math_return = df['MATH_Returns'].to_numpy()
            soci_return = df['SOCI_Returns'].to_numpy()
            soth_return = df['SOTH_Returns'].to_numpy()

            for i in range(n):
                avg_math_soci_return = (soci_return[i] + math_return[i]) / 2
                positive_diff = soth_return[i] - avg_math_soci_return

            #if the SOTH index is overpriced, short SOTH and long MATH and SOCI
            if positive_diff > (0.05 / 100):
                pos_math[i] = 0.5
                pos_soci[i] = 0.5
                pos_soth[i] = -1
            #vice versa, if SOTH is underpriced, long SOTH and short MATH and SOCI
            elif positive_diff < (-0.05 / 100):
                pos_math[i] = -0.5
                pos_soci[i] = -0.5
                pos_soth[i] = 1
            #if they converge and have a mis-pricing of less than +/-0.5 on either side
            else:
                pos_math[i] = 0
                pos_soci[i] = 0
                pos_soth[i] = 0
                
            symbols = {"MATH": pos_math, "SOCI": pos_soci, "SOTH": pos_soth}

            for ticker, position in symbols.items():
                df[f"Position_{ticker}"] = position
        def mean_reversion(price):
            # use RSI and Bollinger Bands
            # RSI period = 6 DAYS, Bollinger Bands period = 20 DAYS
            # I noticed that each day has 68 entry times
            # So:
                # bollinger period = 20 * 68
                # rsi period = 6 * 68

            n = len(df)
            pos_math = np.zeros(n, dtype=int)

            symbols = {'MATH': pos_math}
            for ticker, position in symbols.items():
                df[f'Position_{ticker}'] = position

            # Bollinger bands
            bollinger_period = 6
            sma_20 = price.rolling(window = bollinger_period).mean()
            stdev_20 = price.rolling(window = bollinger_period).std()

            upper_bollinger = sma_20 + 2 * stdev_20
            lower_bollinger = sma_20 - 2 * stdev_20

            # RSI Indicators
            rsi_period = 2
            returns = price.pct_change()
            average_gain = returns.clip(lower = 0).ewm(rsi_period).mean()
            average_loss = (returns.clip(upper = 0) * -1).ewm(rsi_period).mean()

            rsi_index =  100 - (100 / (1 + average_gain / average_loss))

            # Implementation: If price > upper bollinger AND RSI > 70, short. 
            # If price < lower bollinger AND RSI < 30, long.

            n = len(price)
            pos = np.zeros(n, dtype=int)

            for i in range(bollinger_period, n):
                if price.iat[i] > upper_bollinger.iat[i] and rsi_index.iat[i] > 70:
                    pos[i] = -1
                elif price.iat[i] < lower_bollinger.iat[i] and rsi_index.iat[i] < 30:
                    pos[i] = 1
                else:
                    pos[i] = pos[i-1] #carry forward previous position

            return pos
        
            symbols = {"MATH": pos_math, "SOCI": pos_soci, "SOTH": pos_soth}

            for ticker, position in symbols.items():
                df[f"Position_{ticker}"] = position

        #Call function in main
        ewmac()

    def prepare_data(self):
        """
        Truncates raw_df to [start_date, end_date], initializes position columns,
        generates random positions and computes their diffs.
        """
        # Filter date range
        df = self.raw_df[
            (self.raw_df.index >= self.start_date) &
            (self.raw_df.index <= self.end_date)
        ].copy()

        # Init position & PnL columns
        for s in self.strats:
            df[f'Position_{s}'] = 0
        df['Cash'] = 0.0
        df['PnL']  = 0.0

        # Seed for reproducibility
        np.random.seed(42)

        # Call the function to generate positions

        self._generate_positions(df)

        # # TODO: Uncomment above line to activate your trading positions for backtesting
        
##############################################
### DO NOT CHANGE ANYTHING BELOW THIS LINE ### 
##############################################

        # Compute diffs for transaction sizing
        for s in self.strats:
            pos_col  = f'Position_{s}'
            diff_col = f'{pos_col}_diff'
            df[diff_col] = df[pos_col].diff().fillna(df[pos_col].iloc[0])

        self.bt_df = df

    def run(self):
        """
        Loops through self.bt_df, applies trades, updates cash, stock value and PnL.
        """
        if self.bt_df is None:
            raise RuntimeError("Data not prepared. Call prepare_data() first.")

        df = self.bt_df
        cash = self.init_cash

        # price↔position diff & price↔position mappings
        txn_map = {s: (s, f'Position_{s}_diff') for s in self.strats}
        pos_map = {s: (s, f'Position_{s}') for s in self.strats}

        for ts, row in df.iterrows():
            # cash needed to execute today's trades
            cash_needed = sum(row[p] * row[d] for p, d in txn_map.values())
            # total market value of current holdings
            stock_value = sum(row[p] * row[pos] for p, pos in pos_map.values())

            if cash < cash_needed:
                print(f"[{ts}] Insufficient cash. Stopping simulation.")
                raise InsufficientCash(
                    f"[{ts}] Insufficient cash: have {cash:.2f}, "
                    f"need {cash_needed:.2f}"
                )

            # execute trades
            cash -= cash_needed
            df.at[ts, 'Cash'] = cash
            df.at[ts, 'Total_Stock_Value'] = stock_value
            df.at[ts, 'PnL'] = cash + stock_value - self.init_cash

        self.bt_df = df

    def evaluate_metrics(self):
        """
            Computes Sharpe, max drawdown, mean daily return, # trades, stores in self.metrics.
        """
        df = self.bt_df.copy()
        # cumulative return curve
        cumret = (df['PnL'] / self.init_cash + 1).fillna(1)
        run_max = cumret.cummax()
        try:
            drawdown = (cumret - run_max) / run_max
        except:
            drawdown = np.nan
        # instant returns for sharpe
        inst_ret = cumret.diff().fillna(0)

        # Sharpe (annualized: 252 days * ~78 5-min bars/day)
        sr = inst_ret.mean() / inst_ret.std() * np.sqrt(252 * 78) if inst_ret.std() != 0 else np.nan
        mdd = drawdown.min()

        # daily returns
        inst_ret.name = 'intraday_ret'
        tmp = inst_ret.to_frame()
        tmp['date'] = tmp.index.date
        daily = tmp.groupby('date')['intraday_ret'].sum()
        mean_daily = daily.mean() * 100

        # num trades
        pos_diffs = df[[f'Position_{s}_diff' for s in self.strats]].abs()
        ntrades = pos_diffs.sum().sum()

        self.metrics = {
            'Sharpe Ratio': sr,
            'Max Drawdown': mdd,
            'Mean Daily Return (%)': mean_daily,
            'Number of Trades': int(ntrades)
        }

    def plot_summary(self):
        """
            2x2 Matplotlib summary:
            • cumulative return
            • drawdown
            • positions
            • daily-return histogram
        """
        df = self.bt_df.copy()
        m  = self.metrics

        # Strategy cumulative return (in %)
        strat_cumret = (df['PnL'] / self.init_cash + 1).fillna(1)
        strat_pct    = strat_cumret * 100 - 100

        # Drawdown
        run_max = strat_cumret.cummax()
        try: 
            drawdown = (strat_cumret - run_max) / run_max
        except:
            drawdown = np.nan

        # Daily returns histogram
        inst_ret = strat_cumret.diff().fillna(0)
        inst_ret.name = 'intraday_ret'
        tmp = inst_ret.to_frame()
        tmp['date'] = tmp.index.date
        daily = tmp.groupby('date')['intraday_ret'].sum() * 100
        mean_daily = m['Mean Daily Return (%)']

        fig, axs = plt.subplots(2,2, figsize=(16,10))
        fig.suptitle("Backtest Summary", fontsize=16)

        # 1) Cumulative Return
        axs[0,0].plot(df.index, strat_pct, color='green', label='Strategy')
        axs[0,0].set_title("Cumulative Return (%)")
        axs[0,0].set_ylabel("% Return")
        axs[0,0].grid(True)
        axs[0,0].legend()

        # 2) Drawdown
        axs[0,1].plot(df.index, drawdown*100, color='red')
        axs[0,1].set_title("Drawdown (%)")
        axs[0,1].set_ylabel("% Drawdown")
        axs[0,1].grid(True)

        # 3) Positions
        for s in self.strats:
            axs[1,0].plot(df.index, df[f'Position_{s}'], label=s)
        axs[1,0].set_title("Position Management")
        axs[1,0].legend(fontsize='small')
        axs[1,0].grid(True)

        # 4) Daily‐return histogram
        axs[1,1].hist(daily, bins=30, color='steelblue', edgecolor='black')
        axs[1,1].axvline(mean_daily, color='orange', linestyle='--',
                         label=f'Mean = {mean_daily:.2f}%')
        axs[1,1].set_title("Distribution of Daily Returns (%)")
        axs[1,1].legend()
        axs[1,1].grid(True)

        for ax in axs.flat:
            for lbl in ax.get_xticklabels():
                lbl.set_rotation(45)

        plt.tight_layout(rect=[0,0,1,0.96])
        plt.show()

        # Print key metrics
        print(f"Sharpe Ratio:            {m['Sharpe Ratio']:.2f}")
        print(f"Max Drawdown:            {m['Max Drawdown']:.2%}")
        print(f"Mean Daily Return:       {m['Mean Daily Return (%)']:.2f}%")
        print(f"Number of Trades:        {m['Number of Trades']}")

    def plot_trades(self):
        """
            Interactive Plotly price+buy/sell markers per strategy.
        """
        df = self.bt_df
        for s in self.strats:
            price_col = s
            diff_col  = f'Position_{s}_diff'
            buys  = df[df[diff_col] > 0]
            sells = df[df[diff_col] < 0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df[price_col], mode='lines', name=f'{s} Price'
            ))
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys[price_col], mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=15),
                name='Buy'
            ))
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells[price_col], mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=15),
                name='Sell'
            ))

            fig.update_layout(
                title=f"{s}: Price with Buy/Sell Signals",
                xaxis_title="Time", yaxis_title="Price",
                legend=dict(orientation='h', y=1.1),
                template='plotly_white', hovermode='x unified',
                height=500
            )
            fig.show()

    def run_all(self):
        """
            Convenience: prepare → simulate → evaluate → plot summary.
            If InsufficientCash is raised, we log and exit early (no plots).
        """
        try:
            self.prepare_data()
            self.run()
            self.evaluate_metrics()
            self.plot_summary()
        except InsufficientCash as e:
            logging.getLogger('Backtester').error(
                "Backtest aborted: %s. No further evaluation or plotting.", e
            )
            return
        except Exception:
            logging.getLogger('Backtester').exception(
                "Unhandled exception in run_all()"
            )
            raise