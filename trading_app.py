import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.base import clone
from mlxtend.evaluate import bias_variance_decomp
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings

# Suprimir avisos para uma saída mais limpa no Streamlit
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="Análise de Estratégia de Trading ML")

st.title("📈 Análise de Estratégia de Trading com Machine Learning")
st.markdown("""
Esta aplicação permite que você explore uma estratégia de trading baseada em Machine Learning,
utilizando a decomposição Bias-Variância para seleção de modelos e a estacionarização de séries temporais.
""")

# --- Funções do Pipeline (Mantidas as mesmas, adaptadas para o Streamlit) ---

@st.cache_data(show_spinner="A baixar dados financeiros...")
def download_data(ticker: str, period: str) -> pd.DataFrame:
    """Baixa dados históricos do Yahoo Finance."""
    df = yf.download(ticker, period=period, multi_level_index=False)
    df.reset_index(inplace=True)
    return df

@st.cache_data(show_spinner="A criar indicadores técnicos...")
def create_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula uma série de indicadores técnicos como features."""
    price = df['Close']
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']
    indicators = pd.DataFrame(index=df.index)

    indicators['sma_5'] = price.rolling(window=5).mean()
    indicators['sma_10'] = price.rolling(window=10).mean()
    indicators['ema_5'] = price.ewm(span=5).mean()
    indicators['ema_10'] = price.ewm(span=10).mean()
    indicators['momentum_5'] = price - price.shift(5)
    indicators['momentum_10'] = price - price.shift(10)
    indicators['roc_5'] = price.pct_change(5)
    indicators['roc_10'] = price.pct_change(10)
    indicators['std_5'] = price.rolling(window=5).std()
    indicators['std_10'] = price.rolling(window=10).std()

    delta = price.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_gain = up.rolling(window=14).mean()
    avg_loss = down.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    indicators['rsi_14'] = 100 - (100 / (1 + rs))

    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    indicators['vwap'] = vwap

    # OBV - Usando operações vetorizadas do Pandas
    obv = pd.Series(0, index=close.index)
    price_change = close.diff()
    obv[price_change > 0] = volume[price_change > 0]
    obv[price_change < 0] = -volume[price_change < 0]
    indicators['obv'] = obv.cumsum().fillna(0)

    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([
        high - low,
        abs(high - close.shift()),
        abs(low - close.shift())
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (-minus_dm.ewm(alpha=1/14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    indicators['adx_14'] = dx.ewm(alpha=1/14).mean()

    indicators['atr_14'] = atr
    sma20 = price.rolling(20).mean()
    std20 = price.rolling(20).std()
    indicators['bollinger_upper'] = sma20 + 2 * std20
    indicators['bollinger_lower'] = sma20 - 2 * std20
    ema12 = price.ewm(span=12).mean()
    ema26 = price.ewm(span=26).mean()
    indicators['macd'] = ema12 - ema26
    tp = (high + low + close) / 3
    cci = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
    indicators['cci_20'] = cci
    highest_high = high.rolling(14).max()
    lowest_low = low.rolling(14).min()
    indicators['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    indicators['stochastic_k'] = 100 * (close - low14) / (high14 - low14 + 1e-10)

    return indicators.dropna().reset_index(drop=True)

@st.cache_data
def compute_target(df: pd.DataFrame) -> pd.Series:
    """Calcula a variável alvo (retorno percentual de 5 dias)."""
    return df['Close'].pct_change(periods=5).shift(-5)

@st.cache_data(show_spinner="A executar otimização walk-forward e extração de features...")
def walk_forward_with_pca_vif(data, model, window_size=250, step_size=1, n_components=4):
    """Executa a otimização walk-forward com PCA e VIF."""
    predictions = []
    prediction_indices = []
    last_scaler = None
    last_pca = None
    last_vif_scores = None
    last_trained_model = None
    last_selected_pc_indices = None
    
    for i in range(window_size, len(data) - 5, step_size):
        train_data = data.iloc[i - window_size:i]
        test_row = data.iloc[i]

        X_raw = train_data.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'], errors='ignore')
        y_train = train_data['Target']

        if X_raw.empty or y_train.empty:
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)

        vif_data = pd.DataFrame(X_pca, columns=[f'PC{j+1}' for j in range(n_components)])
        
        selected_pc_indices = np.array([True] * vif_data.shape[1]) # Default to all if VIF not applicable
        if vif_data.shape[1] > 1:
            vif_scores = [variance_inflation_factor(vif_data.values, j) for j in range(vif_data.shape[1])]
            selected_pc_indices = np.array(vif_scores) < 3
        
        selected_pcs = vif_data.iloc[:, selected_pc_indices]

        test_raw = test_row.drop(labels=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'], errors='ignore').values.reshape(1, -1)
        test_scaled = scaler.transform(test_raw)
        test_pca = pca.transform(test_scaled)
        test_vif_df = pd.DataFrame(test_pca, columns=[f'PC{j+1}' for j in range(n_components)])
        test_selected = test_vif_df.iloc[:, selected_pc_indices]

        if selected_pcs.empty or test_selected.empty:
            continue

        model_clone = clone(model)
        model_clone.fit(selected_pcs, y_train)
        prediction = model_clone.predict(test_selected)[0]
        predictions.append(prediction)
        prediction_indices.append(i)

        last_scaler = scaler
        last_pca = pca
        last_vif_scores = vif_scores if 'vif_scores' in locals() else None
        last_trained_model = model_clone
        last_selected_pc_indices = selected_pc_indices 

    pred_series = pd.Series(data=np.nan, index=data.index)
    if prediction_indices:
        pred_series.iloc[prediction_indices] = predictions
    
    return pred_series, last_scaler, last_pca, last_vif_scores, last_trained_model, last_selected_pc_indices

@st.cache_data(show_spinner="A executar backtesting da estratégia...")
def backtest_strategy(final_data, initial_capital=10000, capital_deployed=0.2):
    """Simula a estratégia de trading."""
    trade_log = []
    for i in range(len(final_data) - 5):
        prediction = final_data.iloc[i]['Predictions']
        if pd.isna(prediction) or prediction == 0:
            continue

        direction = 'Long' if prediction > 0 else 'Short'
        entry_price = final_data.iloc[i + 1]['Open']
        exit_price = final_data.iloc[i + 5]['Close']

        if direction == 'Long':
            trade_return = (exit_price - entry_price) / entry_price
        else:
            trade_return = (entry_price - exit_price) / entry_price

        capital_change = capital_deployed * initial_capital * trade_return

        trade_log.append({
            'Signal_Day': final_data.iloc[i]['Date'],
            'Entry_Day': final_data.iloc[i + 1]['Date'],
            'Exit_Day': final_data.iloc[i + 5]['Date'],
            'Direction': direction,
            'Entry_Price': entry_price,
            'Exit_Price': exit_price,
            'Return': trade_return,
            'Capital_Change': capital_change
        })
    return pd.DataFrame(trade_log)

def sharpe_ratio(equity_curve, risk_free_rate=0.0):
    """Calcula o Sharpe Ratio anualizado."""
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess_returns = daily_returns - (risk_free_rate / 252)
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def max_drawdown(equity_curve):
    """Calcula o Maximum Drawdown."""
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min(), drawdown

def get_models():
    """Retorna os modelos de regressão para avaliação."""
    return {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'DecisionTree': DecisionTreeRegressor(),
        'Bagging': BaggingRegressor(n_estimators=100, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

@st.cache_data(show_spinner="A executar decomposição Bias-Variância...")
def evaluate_bias_variance_all(_models, X, y, test_size=0.2, num_rounds=100): # Changed 'models' to '_models'
    """Executa a decomposição Bias-Variância para múltiplos modelos."""
    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    results = {}
    for name, model in _models.items(): # Use _models here
        if len(y_train) == 0 or len(y_test) == 0:
            results[name] = {'Total Error': np.nan, 'Bias': np.nan, 'Variance': np.nan, 'Irreducible Error': np.nan}
            continue

        try:
            avg_loss, bias, var = bias_variance_decomp(
                model,
                X_train.values, y_train.values,
                X_test.values, y_test.values,
                loss='mse',
                num_rounds=num_rounds,
                random_seed=42
            )
            results[name] = {
                'Total Error': avg_loss,
                'Bias': bias,
                'Variance': var,
                'Irreducible Error': avg_loss - bias - var
            }
        except Exception as e:
            results[name] = {'Total Error': np.nan, 'Bias': np.nan, 'Variance': np.nan, 'Irreducible Error': np.nan}
    return pd.DataFrame(results).T

@st.cache_data(show_spinner="A verificar a ordem de integração dos preditores...")
def find_integration_order(series):
   """Determina a ordem de integração (d) usando o teste Augmented Dickey-Fuller."""
   d = 0
   current_series = series.copy()
   while True:
       if len(current_series.dropna()) < 20:
           return d
       result = adfuller(current_series.dropna(), autolag='AIC')
       if result[1] <= 0.05:
           return d
       current_series = current_series.diff().dropna()
       d += 1
       if d >= 2:
           return d

# --- Interface do Streamlit ---

with st.sidebar:
    st.header("Parâmetros de Entrada")
    ticker_input = st.text_input("Ticker do Ativo (ex: PETR4.SA)", value="PETR4.SA")
    period_input = st.selectbox("Período dos Dados", ["1y", "2y", "5y", "10y", "max"], index=2) # Default 5y

    st.markdown("---")
    st.subheader("Configurações do Modelo")
    window_size_input = st.slider("Tamanho da Janela (Walk-Forward)", 100, 500, 250, 10)
    n_components_pca_input = st.slider("Componentes PCA (n_components)", 1, 10, 4, 1)
    
    st.markdown("---")
    if st.button("Executar Análise Completa"):
        st.session_state.run_analysis = True
    else:
        st.session_state.run_analysis = False

# Initialize session_state.run_analysis if it doesn't exist
if 'run_analysis' not in st.session_state:
    st.session_state.run_analysis = False

if st.session_state.run_analysis:
    st.info(f"A iniciar análise para {ticker_input} ({period_input}). Isso pode demorar um pouco...")

    # --- Execução do Pipeline ---
    try:
        # 1. Download de Dados
        st.subheader("1. Download de Dados")
        data = download_data(ticker_input, period_input)
        if data.empty:
            st.error(f"Não foram encontrados dados para o ticker {ticker_input} no período {period_input}. Por favor, verifique o ticker.")
            st.stop()
        st.write("Dados brutos baixados (primeiras 5 linhas):")
        st.dataframe(data.head())

        if len(data) < 30:
            st.warning(f"AVISO: Poucos dados ({len(data)} linhas) para análise completa. Tente um período maior.")
            st.stop()

        # 2. Criação de Indicadores Técnicos
        st.subheader("2. Criação de Indicadores Técnicos")
        indicators = create_technical_indicators(data)
        st.write("Indicadores técnicos criados (informações):")
        # Use st.text to display info() output as preformatted text
        buffer = pd.io.common.StringIO()
        indicators.info(verbose=True, buf=buffer)
        st.text(buffer.getvalue())
        st.write("Primeiras 5 linhas dos indicadores:")
        st.dataframe(indicators.head())

        # 3. Alinhamento de Dados e Variável Alvo
        st.subheader("3. Alinhamento de Dados e Variável Alvo")
        data_aligned_for_indicators = data.iloc[-len(indicators):].reset_index(drop=True)
        data_merged = pd.concat([data_aligned_for_indicators, indicators], axis=1)
        data_merged['Target'] = compute_target(data_merged)
        data_merged.dropna(inplace=True)
        st.write("Dados alinhados e alvo calculado (informações):")
        buffer = pd.io.common.StringIO()
        data_merged.info(verbose=True, buf=buffer)
        st.text(buffer.getvalue())
        st.write("Últimas 5 linhas dos dados mesclados com alvo:")
        st.dataframe(data_merged.tail())

        if data_merged.empty:
            st.error("Dados insuficientes após o pré-processamento para continuar a análise.")
            st.stop()

        # 4. Decomposição Bias-Variância (sem estacionarização)
        st.subheader("4. Decomposição Bias-Variância (sem estacionarização)")
        X_bv = data_merged.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'], errors='ignore')
        y_bv = data_merged['Target']
        X_bv = X_bv.dropna()
        y_bv = y_bv.loc[X_bv.index]

        if not X_bv.empty and not y_bv.empty:
            models = get_models()
            bv_results = evaluate_bias_variance_all(_models=models, X=X_bv, y=y_bv, num_rounds=50) # Pass _models
            st.write("Resultados da Decomposição Bias-Variância (Sem Estacionarização):")
            st.dataframe(bv_results)
        else:
            st.warning("Dados insuficientes para a decomposição Bias-Variância (sem estacionarização).")

        # 5. Estacionarização dos Inputs
        st.subheader("5. Estacionarização dos Inputs")
        integration_orders = {}
        for col in indicators.columns:
            integration_orders[col] = find_integration_order(indicators[col].copy())
        integration_orders_df = pd.DataFrame.from_dict(integration_orders, orient='index', columns=['Integration Order'])
        integration_orders_df.index.name = 'Indicator'
        st.write("Ordens de integração dos indicadores:")
        st.dataframe(integration_orders_df["Integration Order"].value_counts().to_frame(name='Contagem'))

        differenced_indicators = indicators.copy()
        for indicator in integration_orders_df.index:
            order = integration_orders_df.loc[indicator, 'Integration Order']
            if order > 0:
                temp_series = indicators[indicator]
                for d_order in range(order):
                    temp_series = temp_series.diff().dropna()
                differenced_indicators[indicator] = temp_series
        differenced_indicators.dropna(inplace=True)
        st.write("Preditores estacionarizados (informações):")
        buffer = pd.io.common.StringIO()
        differenced_indicators.info(verbose=True, buf=buffer)
        st.text(buffer.getvalue())
        st.write("Primeiras 5 linhas dos preditores estacionarizados:")
        st.dataframe(differenced_indicators.head())

        # Verificação da estacionarização
        integration_orders_2 = {}
        for col in differenced_indicators.columns:
            integration_orders_2[col] = find_integration_order(differenced_indicators[col].copy())
        integration_orders_2_df = pd.DataFrame.from_dict(integration_orders_2, orient='index', columns=['Integration Order'])
        st.write("Ordens de integração após estacionarização (verificação):")
        st.dataframe(integration_orders_2_df["Integration Order"].value_counts().to_frame(name='Contagem'))

        # 6. Alinhamento de Dados com Indicadores Estacionarizados
        st.subheader("6. Alinhamento de Dados com Indicadores Estacionarizados")
        data_aligned_for_differenced_indicators = data.iloc[-len(differenced_indicators):].reset_index(drop=True)
        data_merged_differenced = pd.concat([data_aligned_for_differenced_indicators, differenced_indicators], axis=1)
        data_merged_differenced['Target'] = compute_target(data_merged_differenced)
        data_merged_differenced.dropna(inplace=True)
        st.write("Dados alinhados com indicadores estacionarizados (informações):")
        buffer = pd.io.common.StringIO()
        data_merged_differenced.info(verbose=True, buf=buffer)
        st.text(buffer.getvalue())
        st.write("Últimas 5 linhas dos dados mesclados e estacionarizados com alvo:")
        st.dataframe(data_merged_differenced.tail())

        if data_merged_differenced.empty:
            st.error("Dados insuficientes após estacionarização e pré-processamento para continuar a análise.")
            st.stop()

        # 7. Decomposição Bias-Variância (com estacionarização)
        st.subheader("7. Decomposição Bias-Variância (com estacionarização)")
        X_bv_differenced = data_merged_differenced.drop(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'], errors='ignore')
        y_bv_differenced = data_merged_differenced['Target']
        X_bv_differenced = X_bv_differenced.dropna()
        y_bv_differenced = y_bv_differenced.loc[X_bv_differenced.index]

        if not X_bv_differenced.empty and not y_bv_differenced.empty:
            bv_results_differenced = evaluate_bias_variance_all(_models=models, X=X_bv_differenced, y=y_bv_differenced, num_rounds=50) # Pass _models
            st.write("Resultados da Decomposição Bias-Variância (Com Preditores Estacionarizados):")
            st.dataframe(bv_results_differenced)
            st.info("O modelo GradientBoosting geralmente oferece um bom equilíbrio e menor erro total.")
        else:
            st.warning("Dados insuficientes para a decomposição Bias-Variância (com estacionarização).")

        # 8. Rodar Previsão Walk-Forward
        st.subheader("8. Previsão Walk-Forward e Backtesting")
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        pred_series_differenced, last_scaler, last_pca, last_vif_scores, last_trained_model, last_selected_pc_indices = \
            walk_forward_with_pca_vif(data_merged_differenced, model, window_size=window_size_input, n_components=n_components_pca_input)

        data_merged_differenced['Predictions'] = pred_series_differenced
        data_merged_differenced.dropna(subset=['Predictions'], inplace=True)

        final_data_differenced = data_merged_differenced[['Date', 'Open', 'Close', 'Target', 'Predictions']].copy()
        final_data_differenced['Date'] = pd.to_datetime(final_data_differenced['Date'])
        trades_df_differenced = backtest_strategy(final_data_differenced)

        st.subheader("Log de Trades")
        if not trades_df_differenced.empty:
            cols_to_round_to_two = ['Entry_Price', 'Exit_Price']
            trades_df_differenced[cols_to_round_to_two] = trades_df_differenced[cols_to_round_to_two].round(2)
            cols_to_round_to_four = ['Return', 'Capital_Change']
            trades_df_differenced[cols_to_round_to_four] = trades_df_differenced[cols_to_round_to_four].round(4)
            st.dataframe(trades_df_differenced.head(10))
            st.write(f"Total de trades gerados: {len(trades_df_differenced)}")
        else:
            st.warning("Nenhum trade foi gerado durante o backtesting. Isso pode ocorrer se houver dados insuficientes ou se o modelo não gerar sinais de trading válidos.")

        # 9. Curvas de Capital e Métricas de Desempenho
        st.subheader("9. Curvas de Capital e Métricas de Desempenho")
        initial_capital = 10000
        if not trades_df_differenced.empty:
            trades_df_differenced['Cumulative_Capital_Strategy'] = initial_capital + trades_df_differenced['Capital_Change'].cumsum()

        first_trade_date = trades_df_differenced['Entry_Day'].min() if not trades_df_differenced.empty else None
        last_trade_date = trades_df_differenced['Exit_Day'].max() if not trades_df_differenced.empty else None

        bh_data = pd.DataFrame()
        if first_trade_date and last_trade_date:
            bh_data = data_merged_differenced[(data_merged_differenced['Date'] >= first_trade_date) &
                                            (data_merged_differenced['Date'] <= last_trade_date)].copy()

        if not bh_data.empty:
            bh_initial_price = bh_data.iloc[0]['Close']
            bh_data['Cumulative_Capital_BH'] = initial_capital * (bh_data['Close'] / bh_initial_price)
        else:
            st.warning("Não há dados suficientes para calcular o Buy and Hold no período dos trades.")

        st.markdown("#### Curva de Capital")
        fig_equity, ax_equity = plt.subplots(figsize=(12, 6))
        if not trades_df_differenced.empty and 'Cumulative_Capital_Strategy' in trades_df_differenced.columns:
            ax_equity.plot(trades_df_differenced['Exit_Day'], trades_df_differenced['Cumulative_Capital_Strategy'], label='Estratégia de Trading', color='blue')
        if not bh_data.empty and 'Cumulative_Capital_BH' in bh_data.columns:
            ax_equity.plot(bh_data['Date'], bh_data['Cumulative_Capital_BH'], label='Buy and Hold', color='orange')
        ax_equity.set_title('Curva de Capital da Estratégia vs. Buy and Hold')
        ax_equity.set_xlabel('Data')
        ax_equity.set_ylabel('Capital (R$)')
        ax_equity.legend()
        ax_equity.grid(True)
        st.pyplot(fig_equity)

        st.markdown("#### Métricas de Desempenho")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Estratégia de Trading:**")
            if not trades_df_differenced.empty:
                final_capital_strategy = trades_df_differenced['Cumulative_Capital_Strategy'].iloc[-1]
                total_return_strategy = (final_capital_strategy - initial_capital) / initial_capital
                trading_days = (trades_df_differenced['Exit_Day'].iloc[-1] - trades_df_differenced['Entry_Day'].iloc[0]).days
                num_years = trading_days / 365.25
                cagr_strategy = ((final_capital_strategy / initial_capital) ** (1 / num_years)) - 1 if num_years > 0 else 0
                sharpe_strategy = sharpe_ratio(trades_df_differenced['Cumulative_Capital_Strategy'])
                max_dd_strategy, _ = max_drawdown(trades_df_differenced['Cumulative_Capital_Strategy'])
                profitable_trades = trades_df_differenced[trades_df_differenced['Return'] > 0]
                hit_ratio_strategy = len(profitable_trades) / len(trades_df_differenced) if len(trades_df_differenced) > 0 else 0
                avg_return_per_trade_strategy = trades_df_differenced['Return'].mean()

                st.write(f"Capital Final: R$ {final_capital_strategy:.2f}")
                st.write(f"Retorno Total: {total_return_strategy:.2%}")
                st.write(f"CAGR: {cagr_strategy:.2%}")
                st.write(f"Sharpe Ratio: {sharpe_strategy:.2f}")
                st.write(f"Max Drawdown: {max_dd_strategy:.2%}")
                st.write(f"Hit Ratio: {hit_ratio_strategy:.2%}")
                st.write(f"Retorno Médio por Trade: {avg_return_per_trade_strategy:.2%}")
            else:
                st.write("Não há trades para calcular as métricas.")

        with col2:
            st.markdown("**Buy and Hold:**")
            if not bh_data.empty:
                final_capital_bh = bh_data['Cumulative_Capital_BH'].iloc[-1]
                total_return_bh = (final_capital_bh - initial_capital) / initial_capital
                cagr_bh = ((final_capital_bh / initial_capital) ** (1 / num_years)) - 1 if num_years > 0 else 0
                sharpe_bh = sharpe_ratio(bh_data['Cumulative_Capital_BH'])
                max_dd_bh, _ = max_drawdown(bh_data['Cumulative_Capital_BH'])

                st.write(f"Capital Final: R$ {final_capital_bh:.2f}")
                st.write(f"Retorno Total: {total_return_bh:.2%}")
                st.write(f"CAGR: {cagr_bh:.2%}")
                st.write(f"Sharpe Ratio: {sharpe_bh:.2f}")
                st.write(f"Max Drawdown: {max_dd_bh:.2%}")
            else:
                st.write("Não há dados para o Buy and Hold para calcular as métricas.")

        if not trades_df_differenced.empty:
            st.markdown("#### Distribuição dos Retornos por Trade")
            fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
            ax_hist.hist(trades_df_differenced['Return'], bins=50, edgecolor='black')
            ax_hist.set_title('Distribuição dos Retornos por Trade')
            ax_hist.set_xlabel('Retorno')
            ax_hist.set_ylabel('Frequência')
            ax_hist.grid(True)
            st.pyplot(fig_hist)

        # 10. Previsão Futura
        st.subheader("10. Previsão Futura Baseada nos Dados Mais Recentes")
        if last_trained_model is not None and last_scaler is not None and last_pca is not None and last_selected_pc_indices is not None:
            if len(data) >= 30:
                temp_df_for_latest_indicators = data.iloc[-30:].copy()
            else:
                temp_df_for_latest_indicators = data.copy()

            latest_indicators_full = create_technical_indicators(temp_df_for_latest_indicators)

            if not latest_indicators_full.empty:
                latest_full_indicators_row = latest_indicators_full.iloc[-1]
                latest_differenced_indicators_for_prediction = latest_full_indicators_row.copy()
                
                for indicator_name, order in integration_orders.items():
                    if order > 0 and indicator_name in indicators.columns:
                        temp_series = indicators[indicator_name]
                        for d_order in range(order):
                            temp_series = temp_series.diff().dropna()
                        if not temp_series.empty:
                            latest_differenced_indicators_for_prediction[indicator_name] = temp_series.iloc[-1]
                        else:
                            latest_differenced_indicators_for_prediction.drop(indicator_name, inplace=True, errors='ignore')
                    elif indicator_name in latest_full_indicators_row.index:
                        latest_differenced_indicators_for_prediction[indicator_name] = latest_full_indicators_row[indicator_name]
                    else:
                        latest_differenced_indicators_for_prediction.drop(indicator_name, inplace=True, errors='ignore')
                
                X_latest_for_prediction = pd.DataFrame([latest_differenced_indicators_for_prediction]).dropna(axis=1)

                training_feature_columns = data_merged_differenced.drop(
                    columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'Predictions'], 
                    errors='ignore'
                ).columns

                missing_cols = set(training_feature_columns) - set(X_latest_for_prediction.columns)
                for c in missing_cols:
                    X_latest_for_prediction[c] = 0

                extra_cols = set(X_latest_for_prediction.columns) - set(training_feature_columns)
                X_latest_for_prediction = X_latest_for_prediction.drop(columns=list(extra_cols))
                
                X_latest_for_prediction = X_latest_for_prediction[training_feature_columns]

                X_latest_raw_for_transform = X_latest_for_prediction.values.reshape(1, -1)

                X_latest_scaled = last_scaler.transform(X_latest_raw_for_transform)
                X_latest_pca = last_pca.transform(X_latest_scaled)

                X_latest_selected_pcs = pd.DataFrame(X_latest_pca, columns=[f'PC{j+1}' for j in range(last_pca.n_components_)])
                X_latest_selected_for_prediction = X_latest_selected_pcs.iloc[:, last_selected_pc_indices]

                future_prediction = last_trained_model.predict(X_latest_selected_for_prediction)[0]

                latest_date_available = data['Date'].iloc[-1].strftime('%Y-%m-%d')
                st.write(f"A última data de dados disponível para análise (baixada pelo yfinance) é: **{latest_date_available}**")
                st.write(f"Previsão de retorno percentual para os próximos **5 dias de negociação** a partir desta data: **{future_prediction:.4%}**")
                
                if future_prediction > 0:
                    st.success("O modelo sugere um movimento de **ALTA** nos próximos 5 dias de negociação.")
                elif future_prediction < 0:
                    st.error("O modelo sugere um movimento de **BAIXA** nos próximos 5 dias de negociação.")
                else:
                    st.info("O modelo sugere um movimento **NEUTRO** nos próximos 5 dias de negociação.")
            else:
                st.warning("Não foi possível gerar indicadores para a previsão futura (dados insuficientes ou erro na criação).")
        else:
            st.warning("Não foi possível gerar a previsão futura. Verifique se o pipeline de dados foi executado corretamente e gerou modelos/transformadores válidos.")

        st.success("Análise Completa!")

    except Exception as e:
        st.error(f"Ocorreu um erro durante a execução da análise: {e}")
        st.exception(e)

st.markdown("---")
st.markdown("""
### Considerações Realistas
É crucial lembrar que este backtest é baseado em dados históricos e **não garante retornos futuros**. Aspectos como **custos de transação (corretagem, taxas), slippage (diferença entre preço esperado e preço de execução), impostos** e a **capacidade de execução em tempo real** não foram incluídos. Além disso, a escolha dos parâmetros (`window_size`, `n_components` do PCA, `num_rounds` da decomposição de bias-variância) são fundamentais para um modelo mais robusto na prática.

Este aplicativo é uma base excelente para explorar a decomposição de bias-variância na construção de estratégias de trading. Sinta-se à vontade para experimentar diferentes modelos, parâmetros e períodos de dados!
""")
