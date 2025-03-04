import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return
from pypfopt.discrete_allocation import DiscreteAllocation

# Judul aplikasi
st.title("📈 Sustainable Investment Portfolio Optimizer")

# Pilihan ETF berbasis ESG
default_tickers = ["ESGU", "SUSA", "ESG", "VSGX"]
tickers = st.multiselect("Pilih ESG Funds:", default_tickers, default=default_tickers)

start_date = st.date_input("Mulai:", pd.to_datetime("2020-01-01"))
end_date = st.date_input("Akhir:", pd.to_datetime("2025-01-01"))

if st.button("Ambil Data & Optimasi"):
    # Mengambil data harga penutupan saham
    data = yf.download(tickers, start=start_date, end=end_date)["Close"]
    st.write("📊 **Harga Saham (5 data pertama)**")
    st.dataframe(data.head())

    # Perhitungan optimasi portofolio
    mu = mean_historical_return(data)
    S = CovarianceShrinkage(data).ledoit_wolf()

    ef = EfficientFrontier(mu, S)
    raw_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    st.write("⚖️ **Optimized Portfolio Weights:**")
    st.json(cleaned_weights)

    expected_return, volatility, sharpe_ratio = ef.portfolio_performance()
    st.write(f"📈 **Expected Annual Return:** {expected_return:.2%}")
    st.write(f"📉 **Annual Volatility:** {volatility:.2%}")
    st.write(f"📊 **Sharpe Ratio:** {sharpe_ratio:.2f}")

    # Alokasi portofolio
    portfolio_value = st.number_input("Total Investasi (USD):", min_value=1000, value=10000, step=1000)
    latest_prices = data.iloc[-1]
    da = DiscreteAllocation(cleaned_weights, latest_prices, total_portfolio_value=portfolio_value)
    allocation, leftover = da.lp_portfolio()

    st.write("💰 **Discrete Allocation (Jumlah Saham per ETF):**")
    st.json(allocation)
    st.write(f"💵 **Dana Sisa:** ${leftover:.2f}")
