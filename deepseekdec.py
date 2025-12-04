# import streamlit as st
# import pandas as pd
# import numpy as np
# import yfinance as yf
# from datetime import datetime, timedelta
# import plotly.graph_objects as go
# import plotly.express as px
# import warnings
# import time
#
# warnings.filterwarnings('ignore')
#
# # Page configuration
# st.set_page_config(
#     page_title="Stock Portfolio Tracker",
#     page_icon="📈",
#     layout="wide"
# )
#
# # Title
# st.title("📊 Stock Portfolio Tracker")
#
# # Sidebar with auto-refresh
# with st.sidebar:
#     st.header("🔄 Controls")
#
#     # Auto-refresh settings
#     auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
#     if auto_refresh:
#         refresh_interval = st.selectbox("Refresh Interval",
#                                         ["30 seconds", "1 minute", "5 minutes", "10 minutes"])
#         # Convert to seconds
#         interval_map = {
#             "30 seconds": 30,
#             "1 minute": 60,
#             "5 minutes": 300,
#             "10 minutes": 600
#         }
#         refresh_seconds = interval_map[refresh_interval]
#
#     # Manual refresh button
#     if st.button("🔄 Refresh Now", type="primary"):
#         st.rerun()
#
#     show_graphs = st.checkbox("Show Graphs", value=True)
#     show_pie = st.checkbox("Show MCap Pie Chart", value=True)
#
#     st.markdown("---")
#     st.info("""
#     **Your CSV Format:**
#     - Open: Position = 'Open', no EXIT DATE
#     - Close: Position = 'Close', has EXIT DATE
#     - MCap only for open positions
#     - Closed price for exited positions
#     """)
#
#
# # Formatting functions - NO DECIMALS for currency values
# def format_currency_no_decimals(value):
#     """Format currency WITHOUT decimals for main metrics"""
#     if pd.isna(value) or value is None:
#         return "₹0"
#     try:
#         # Force to integer by removing decimals
#         value_int = int(round(float(value)))
#         return f"₹{value_int:,}"
#     except:
#         return "₹0"
#
#
# def format_currency_with_decimals(value):
#     """Format currency WITH decimals (for prices only)"""
#     if pd.isna(value) or value is None:
#         return ""
#     try:
#         return f"₹{float(value):,.2f}"
#     except:
#         return ""
#
#
# def format_percentage(value):
#     if pd.isna(value) or value is None:
#         return ""
#     try:
#         return f"{float(value):.2f}%"
#     except:
#         return ""
#
#
# # Fetch live price
# @st.cache_data(ttl=60)  # Shorter cache for auto-refresh
# def get_live_price(ticker):
#     try:
#         if not ticker.endswith(('.NS', '.BO', '.NSE')):
#             ticker = f"{ticker}.NS"
#         stock = yf.Ticker(ticker)
#         hist = stock.history(period="1d")
#         if not hist.empty:
#             return hist['Close'].iloc[-1]
#         return None
#     except:
#         return None
#
#
# # Fetch yesterday's close price
# @st.cache_data(ttl=300)
# def get_yesterday_close(ticker):
#     try:
#         if not ticker.endswith(('.NS', '.BO', '.NSE')):
#             ticker = f"{ticker}.NS"
#         stock = yf.Ticker(ticker)
#         hist = stock.history(period="3d")
#         if len(hist) >= 2:
#             return hist['Close'].iloc[-2]
#         elif len(hist) == 1:
#             return hist['Close'].iloc[-1]
#         return None
#     except:
#         return None
#
#
# # Fetch historical price
# @st.cache_data(ttl=3600)
# def get_historical_price(ticker, date):
#     try:
#         if pd.isna(date):
#             return None
#         if not ticker.endswith(('.NS', '.BO', '.NSE')):
#             ticker = f"{ticker}.NS"
#         stock = yf.Ticker(ticker)
#         start_date = date - timedelta(days=7)
#         end_date = date + timedelta(days=1)
#         hist = stock.history(start=start_date, end=end_date)
#         if not hist.empty:
#             hist.index = pd.to_datetime(hist.index).normalize()
#             target_date = pd.Timestamp(date).normalize()
#             if target_date in hist.index:
#                 return hist.loc[target_date, 'Close']
#             else:
#                 dates = hist.index
#                 time_diff = abs(dates - target_date)
#                 nearest_idx = time_diff.argmin()
#                 return hist.iloc[nearest_idx]['Close']
#         return None
#     except:
#         return None
#
#
# # Calculate days held
# def calculate_days_held(entry_date, exit_date):
#     if pd.isna(entry_date):
#         return 0
#     if pd.isna(exit_date):
#         return (datetime.now().date() - entry_date.date()).days
#     else:
#         return (exit_date.date() - entry_date.date()).days
#
#
# # Calculate MTM for open positions
# def calculate_daily_mtm(open_positions_df):
#     if open_positions_df.empty:
#         return pd.DataFrame(), 0, 0, 0
#
#     mtm_data = []
#     total_mtm = 0
#     total_yesterday_value = 0
#     total_today_value = 0
#
#     for idx, row in open_positions_df.iterrows():
#         ticker = row['Ticker']
#         qty = row['Quantity']
#         today_price = row['Current Price']
#         yesterday_price = get_yesterday_close(ticker)
#
#         if today_price and yesterday_price and qty > 0:
#             today_value = qty * today_price
#             yesterday_value = qty * yesterday_price
#             daily_mtm = today_value - yesterday_value
#             daily_mtm_percent = (daily_mtm / yesterday_value * 100) if yesterday_value > 0 else 0
#
#             mtm_data.append({
#                 'Ticker': ticker,
#                 'Quantity': qty,
#                 'Yesterday Close': yesterday_price,
#                 'Today Price': today_price,
#                 'Yesterday Value': yesterday_value,
#                 'Today Value': today_value,
#                 'Daily MTM': daily_mtm,
#                 'Daily MTM %': daily_mtm_percent
#             })
#
#             total_mtm += daily_mtm
#             total_yesterday_value += yesterday_value
#             total_today_value += today_value
#
#     mtm_df = pd.DataFrame(mtm_data)
#     portfolio_mtm_percent = (total_mtm / total_yesterday_value * 100) if total_yesterday_value > 0 else 0
#
#     return mtm_df, total_mtm, portfolio_mtm_percent, total_yesterday_value
#
#
# # Main app
# uploaded_file = st.file_uploader("📁 Upload your portfolio CSV file", type=['csv'])
#
# if uploaded_file is not None:
#     # Read and process file
#     df = pd.read_csv(uploaded_file)
#     df.columns = df.columns.str.strip()
#
#     # Show file info
#     st.info(f"📄 File: {uploaded_file.name} | 📊 Rows: {len(df)}")
#
#     # Check for footer row
#     footer_row = df.iloc[-1] if len(df) > 0 else None
#     has_footer = False
#     realized_pl_from_footer = 0
#
#     if footer_row is not None and isinstance(footer_row.get('Closed price'), str) and 'realized' in str(footer_row.get('Closed price', '')).lower():
#         has_footer = True
#         realized_pl_from_footer = footer_row.get('closed position Profit/loss', 0)
#         df = df.iloc[:-1]
#
#     # Clean data
#     df = df.replace('', np.nan)
#
#
#     def parse_date(date_str):
#         if pd.isna(date_str):
#             return pd.NaT
#         try:
#             return datetime.strptime(str(date_str), '%d-%m-%Y')
#         except:
#             try:
#                 return datetime.strptime(str(date_str), '%Y-%m-%d')
#             except:
#                 return pd.NaT
#
#
#     if 'ENTRY DATE' in df.columns:
#         df['ENTRY DATE'] = df['ENTRY DATE'].apply(parse_date)
#     if 'EXIT DATE' in df.columns:
#         df['EXIT DATE'] = df['EXIT DATE'].apply(parse_date)
#
#     numeric_cols = ['QTY', 'ENTRY PRICE', 'Closed price', 'closed position Profit/loss']
#     for col in numeric_cols:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#
#     text_cols = ['SCRIP', 'Mcap', 'Position']
#     for col in text_cols:
#         if col in df.columns:
#             df[col] = df[col].astype(str).str.strip()
#
#     df = df[df['SCRIP'].notna() & (df['SCRIP'] != 'nan')]
#
#     # Process positions
#     results = []
#     total_investment = 0
#     realized_pl = 0
#
#     with st.spinner("🔄 Processing positions and fetching live prices..."):
#         for idx, row in df.iterrows():
#             ticker = str(row['SCRIP']).strip()
#
#             if pd.isna(row['ENTRY DATE']) or pd.isna(row['QTY']) or pd.isna(row['ENTRY PRICE']):
#                 continue
#
#             position_type = str(row.get('Position', '')).upper()
#             is_open = position_type == 'OPEN'
#             is_closed = position_type == 'CLOSE'
#
#             if pd.isna(position_type) or position_type == 'NAN':
#                 is_open = pd.isna(row.get('EXIT DATE'))
#                 is_closed = not pd.isna(row.get('EXIT DATE'))
#
#             days_held = calculate_days_held(row['ENTRY DATE'], row.get('EXIT DATE'))
#
#             if is_open:
#                 current_price = get_live_price(ticker)
#                 entry_value = row['QTY'] * row['ENTRY PRICE']
#
#                 if current_price:
#                     current_position_value = row['QTY'] * current_price
#                     unrealized_pl = current_position_value - entry_value
#                 else:
#                     current_position_value = entry_value
#                     unrealized_pl = 0
#
#                 pnl_percent = (unrealized_pl / entry_value * 100) if entry_value != 0 else 0
#
#                 results.append({
#                     'Ticker': ticker,
#                     'Status': 'Open',
#                     'Entry Date': row['ENTRY DATE'],
#                     'Exit Date': None,
#                     'Quantity': row['QTY'],
#                     'Entry Price': row['ENTRY PRICE'],
#                     'Current Price': current_price,
#                     'Entry Value': entry_value,
#                     'Current Value': current_position_value,
#                     'P&L': unrealized_pl,
#                     'P&L %': pnl_percent,
#                     'Mcap': row.get('Mcap', 'Unknown'),
#                     'Days Held': days_held,
#                     'Type': 'Unrealized'
#                 })
#
#                 total_investment += entry_value
#
#             elif is_closed:
#                 exit_price = row.get('Closed price')
#
#                 if pd.isna(exit_price) and not pd.isna(row.get('EXIT DATE')):
#                     exit_price = get_historical_price(ticker, row['EXIT DATE'])
#
#                 if exit_price and not pd.isna(exit_price):
#                     entry_value = row['QTY'] * row['ENTRY PRICE']
#                     exit_value = row['QTY'] * exit_price
#                     position_pl = exit_value - entry_value
#                     pnl_percent = (position_pl / entry_value * 100) if entry_value != 0 else 0
#
#                     provided_pl = row.get('closed position Profit/loss')
#                     if not pd.isna(provided_pl):
#                         position_pl = provided_pl
#
#                     results.append({
#                         'Ticker': ticker,
#                         'Status': 'Closed',
#                         'Entry Date': row['ENTRY DATE'],
#                         'Exit Date': row['EXIT DATE'],
#                         'Quantity': row['QTY'],
#                         'Entry Price': row['ENTRY PRICE'],
#                         'Exit Price': exit_price,
#                         'Entry Value': entry_value,
#                         'Exit Value': exit_value,
#                         'Current Value': exit_value,
#                         'P&L': position_pl,
#                         'P&L %': pnl_percent,
#                         'Mcap': row.get('Mcap', ''),
#                         'Days Held': days_held,
#                         'Type': 'Booked/Realized'
#                     })
#
#                     realized_pl += position_pl
#                     total_investment += entry_value
#
#     # Create results dataframe
#     if results:
#         results_df = pd.DataFrame(results)
#
#         # Fill NaN values
#         numeric_cols_results = ['Entry Price', 'Exit Price', 'Current Price', 'Entry Value',
#                                 'Exit Value', 'Current Value', 'P&L', 'P&L %', 'Days Held']
#         for col in numeric_cols_results:
#             if col in results_df.columns:
#                 results_df[col] = results_df[col].fillna(0)
#
#         # Convert dates
#         for col in ['Entry Date', 'Exit Date']:
#             if col in results_df.columns:
#                 results_df[col] = pd.to_datetime(results_df[col], errors='coerce')
#
#         # Separate open and closed
#         open_positions = results_df[results_df['Status'] == 'Open']
#         closed_positions = results_df[results_df['Status'] == 'Closed']
#
#         # Calculate metrics
#         total_pl = results_df['P&L'].sum()
#         unrealized_pl = open_positions['P&L'].sum() if not open_positions.empty else 0
#
#         # Use footer P&L if provided
#         if has_footer and abs(realized_pl_from_footer) > 0:
#             realized_pl = realized_pl_from_footer
#
#         realized_pl = closed_positions['P&L'].sum() if not closed_positions.empty else 0
#         total_pl = unrealized_pl + realized_pl
#
#         # PORTFOLIO VALUE - ONLY OPEN POSITIONS (NO DECIMALS)
#         portfolio_value_raw = open_positions['Current Value'].sum() if not open_positions.empty else 0
#         portfolio_value = int(round(portfolio_value_raw))
#
#         # Profitable/loss counts
#         profitable_all = results_df[results_df['P&L'] > 0]
#         loss_all = results_df[results_df['P&L'] < 0]
#
#         profitable_open = open_positions[open_positions['P&L'] > 0] if not open_positions.empty else pd.DataFrame()
#         loss_open = open_positions[open_positions['P&L'] < 0] if not open_positions.empty else pd.DataFrame()
#
#         profitable_closed = closed_positions[closed_positions['P&L'] > 0] if not closed_positions.empty else pd.DataFrame()
#         loss_closed = closed_positions[closed_positions['P&L'] < 0] if not closed_positions.empty else pd.DataFrame()
#
#         # CORRECT TURNOVER CALCULATION - FIXED
#         # Turnover = (QTY * Entry Price for ALL positions) + (QTY * Exit Price for closed positions)
#         # This matches: buy value for all + sell value for closed
#
#         # 1. Buy value for all positions (entry value for all)
#         all_buy_value = results_df['Entry Value'].sum()
#
#         # 2. Sell value for closed positions (exit value for closed)
#         closed_sell_value = closed_positions['Exit Value'].sum() if not closed_positions.empty else 0
#
#         # Total turnover
#         turnover_raw = all_buy_value + closed_sell_value
#         turnover = int(round(turnover_raw))
#
#         # Alternative: If you want only trading volume (buy + sell for closed, buy for open)
#         # turnover = closed_positions['Entry Value'].sum() + closed_positions['Exit Value'].sum() + open_positions['Entry Value'].sum()
#
#         # Show turnover breakdown
#         with st.expander("📊 Turnover Breakdown"):
#             st.write(f"All Positions Buy Value: {format_currency_no_decimals(all_buy_value)}")
#             st.write(f"Closed Positions Sell Value: {format_currency_no_decimals(closed_sell_value)}")
#             st.write(f"**Total Turnover: {format_currency_no_decimals(turnover)}**")
#
#         # Average days held
#         avg_days_held = results_df['Days Held'].mean() if not results_df.empty else 0
#
#         # Annualized return - based on portfolio value (open positions)
#         if portfolio_value > 0 and avg_days_held > 0:
#             total_return_pct = (total_pl / portfolio_value) * 100
#             annualized_return = ((1 + total_return_pct / 100) ** (365 / avg_days_held) - 1) * 100
#         else:
#             annualized_return = 0
#
#         # Win rate
#         win_rate = (len(profitable_all) / len(results_df) * 100) if len(results_df) > 0 else 0
#
#         # Calculate Daily MTM
#         with st.spinner("📈 Calculating daily MTM..."):
#             mtm_df, total_mtm, portfolio_mtm_percent, total_yesterday_value = calculate_daily_mtm(open_positions)
#
#         # Display Portfolio Summary - ALL METRICS WITHOUT DECIMALS
#         st.subheader("📊 Portfolio Summary")
#
#         col1, col2, col3, col4 = st.columns(4)
#
#         with col1:
#             # Format without decimals
#             total_pl_int = int(round(total_pl))
#             portfolio_value_int = int(round(portfolio_value))
#             st.metric("💰 Total P&L", format_currency_no_decimals(total_pl_int),
#                       f"{total_pl / portfolio_value * 100:.2f}%" if portfolio_value > 0 else "0%")
#             st.metric("📊 Portfolio Value", format_currency_no_decimals(portfolio_value_int))
#
#         with col2:
#             # Format without decimals
#             realized_pl_int = int(round(realized_pl))
#             unrealized_pl_int = int(round(unrealized_pl))
#             st.metric("📈 Realized P&L", format_currency_no_decimals(realized_pl_int))
#             st.metric("📉 Unrealized P&L", format_currency_no_decimals(unrealized_pl_int))
#
#         with col3:
#             st.metric("🎯 Annualized Return", f"{annualized_return:.2f}%")
#             st.metric("📅 Avg Days Held", f"{avg_days_held:.1f}")
#
#         with col4:
#             # Format without decimals
#             turnover_int = int(round(turnover))
#             st.metric("📊 Turnover", format_currency_no_decimals(turnover_int))
#             st.metric("📈 Win Rate", f"{win_rate:.1f}%")
#
#         # Position Statistics
#         st.subheader("📊 Position Statistics")
#
#         col1, col2, col3, col4 = st.columns(4)
#
#         with col1:
#             st.metric("💹 All Profitable", len(profitable_all))
#             st.metric("🔻 All Loss", len(loss_all))
#
#         with col2:
#             st.metric("✅ Open Positions", len(open_positions))
#             st.metric("🏁 Closed Positions", len(closed_positions))
#
#         with col3:
#             if not open_positions.empty:
#                 st.metric("📈 Open Profitable", len(profitable_open))
#                 st.metric("📉 Open Loss", len(loss_open))
#
#         with col4:
#             if not closed_positions.empty:
#                 st.metric("💰 Closed Profitable", len(profitable_closed))
#                 st.metric("🔻 Closed Loss", len(loss_closed))
#
#         # Top 5 & Bottom 5 Performers
#         st.subheader("🏆 Top 5 & Bottom 5 Performers")
#
#         if not results_df.empty:
#             top_5 = results_df.nlargest(5, 'P&L')[['Ticker', 'P&L', 'P&L %', 'Status']].copy()
#             bottom_5 = results_df.nsmallest(5, 'P&L')[['Ticker', 'P&L', 'P&L %', 'Status']].copy()
#
#             col1, col2 = st.columns(2)
#
#             with col1:
#                 st.write("**🥇 Top 5 Performers**")
#                 if not top_5.empty:
#                     for idx, row in top_5.iterrows():
#                         st.success(f"**{row['Ticker']}**: {format_currency_no_decimals(row['P&L'])} ({format_percentage(row['P&L %'])})")
#
#             with col2:
#                 st.write("**🔻 Bottom 5 Losers**")
#                 if not bottom_5.empty:
#                     for idx, row in bottom_5.iterrows():
#                         st.error(f"**{row['Ticker']}**: {format_currency_no_decimals(row['P&L'])} ({format_percentage(row['P&L %'])})")
#
#         # Portfolio Analysis Tabs
#         st.subheader("📋 Portfolio Analysis")
#
#
#         def format_for_display(df):
#             if df.empty:
#                 return df
#
#             formatted = df.copy()
#
#             # Format dates
#             for col in ['Entry Date', 'Exit Date']:
#                 if col in formatted.columns:
#                     formatted[col] = pd.to_datetime(formatted[col]).dt.strftime('%Y-%m-%d')
#
#             # Format currency - DIFFERENT FOR PRICES VS VALUES
#             price_cols = ['Entry Price', 'Exit Price', 'Current Price']
#             value_cols = ['Entry Value', 'Exit Value', 'Current Value', 'P&L']
#
#             for col in price_cols:
#                 if col in formatted.columns:
#                     formatted[col] = formatted[col].apply(format_currency_with_decimals)
#
#             for col in value_cols:
#                 if col in formatted.columns:
#                     formatted[col] = formatted[col].apply(format_currency_no_decimals)
#
#             # Format percentages
#             if 'P&L %' in formatted.columns:
#                 formatted['P&L %'] = formatted['P&L %'].apply(format_percentage)
#
#             formatted = formatted.fillna('')
#             return formatted
#
#
#         tab1, tab2, tab3, tab4 = st.tabs([
#             f"All Positions ({len(results_df)})",
#             f"Open Positions ({len(open_positions)})",
#             f"Closed Positions ({len(closed_positions)})",
#             f"Daily MTM ({datetime.now().strftime('%d %b')})"
#         ])
#
#         with tab1:
#             display_all = format_for_display(results_df)
#             st.dataframe(display_all, use_container_width=True)
#
#         with tab2:
#             if not open_positions.empty:
#                 display_open = format_for_display(open_positions)
#                 st.dataframe(display_open, use_container_width=True)
#             else:
#                 st.info("No open positions")
#
#         with tab3:
#             if not closed_positions.empty:
#                 display_closed = format_for_display(closed_positions)
#                 st.dataframe(display_closed, use_container_width=True)
#             else:
#                 st.info("No closed positions")
#
#         with tab4:
#             st.subheader(f"📈 Daily Mark-to-Market ({datetime.now().strftime('%d %b %Y')})")
#
#             if not mtm_df.empty:
#                 col1, col2, col3, col4 = st.columns(4)
#
#                 with col1:
#                     st.metric("📊 Today's MTM", format_currency_no_decimals(total_mtm),
#                               f"{portfolio_mtm_percent:.2f}%")
#
#                 with col2:
#                     st.metric("💰 Yesterday's Value", format_currency_no_decimals(total_yesterday_value))
#
#                 with col3:
#                     gainers = len(mtm_df[mtm_df['Daily MTM'] > 0])
#                     st.metric("📈 Gainers", gainers)
#
#                 with col4:
#                     losers = len(mtm_df[mtm_df['Daily MTM'] < 0])
#                     st.metric("📉 Losers", losers)
#
#                 # Format MTM table
#                 display_mtm = mtm_df.copy()
#                 display_mtm['Yesterday Close'] = display_mtm['Yesterday Close'].apply(format_currency_with_decimals)
#                 display_mtm['Today Price'] = display_mtm['Today Price'].apply(format_currency_with_decimals)
#                 display_mtm['Yesterday Value'] = display_mtm['Yesterday Value'].apply(format_currency_no_decimals)
#                 display_mtm['Today Value'] = display_mtm['Today Value'].apply(format_currency_no_decimals)
#                 display_mtm['Daily MTM'] = display_mtm['Daily MTM'].apply(format_currency_no_decimals)
#                 display_mtm['Daily MTM %'] = display_mtm['Daily MTM %'].apply(lambda x: f"{x:.2f}%")
#
#                 st.dataframe(display_mtm, use_container_width=True)
#             else:
#                 if open_positions.empty:
#                     st.info("No open positions for MTM calculation")
#                 else:
#                     st.warning("Could not fetch yesterday's prices")
#
#         # MCap Exposure
#         st.subheader("🏢 Market Cap Exposure (Open Positions Only)")
#
#         if not open_positions.empty and 'Mcap' in open_positions.columns:
#             open_positions['Mcap'] = open_positions['Mcap'].astype(str).str.strip()
#             open_positions['Mcap'] = open_positions['Mcap'].replace(['', 'nan', 'NaN', 'None'], 'Unknown')
#
#             mcap_exposure = open_positions.groupby('Mcap').agg({
#                 'Current Value': 'sum',
#                 'P&L': 'sum',
#                 'Ticker': 'count'
#             }).rename(columns={'Ticker': 'Count'}).reset_index()
#
#             if not mcap_exposure.empty:
#                 if show_pie:
#                     fig = px.pie(mcap_exposure,
#                                  values='Current Value',
#                                  names='Mcap',
#                                  title='MCap Distribution - Open Positions',
#                                  hole=0.3)
#                     fig.update_traces(textposition='inside', textinfo='percent+label')
#                     st.plotly_chart(fig, use_container_width=True)
#
#                 # Display table
#                 open_total = open_positions['Current Value'].sum()
#                 mcap_exposure['% of Portfolio'] = (mcap_exposure['Current Value'] / open_total * 100).round(2)
#
#                 display_table = mcap_exposure.copy()
#                 display_table['Current Value'] = display_table['Current Value'].apply(format_currency_no_decimals)
#                 display_table['P&L'] = display_table['P&L'].apply(format_currency_no_decimals)
#                 display_table['% of Portfolio'] = display_table['% of Portfolio'].apply(lambda x: f"{x:.1f}%")
#
#                 st.dataframe(display_table[['Mcap', 'Count', 'Current Value', 'P&L', '% of Portfolio']])
#
#         # Export
#         st.subheader("💾 Export Results")
#
#
#         @st.cache_data
#         def convert_df(df):
#             return df.to_csv(index=False).encode('utf-8')
#
#
#         col1, col2, col3, col4 = st.columns(4)
#
#         with col1:
#             csv_all = convert_df(results_df)
#             st.download_button(
#                 label="📥 All Positions",
#                 data=csv_all,
#                 file_name="portfolio_all.csv",
#                 mime="text/csv"
#             )
#
#         with col2:
#             if not open_positions.empty:
#                 csv_open = convert_df(open_positions)
#                 st.download_button(
#                     label="📥 Open Positions",
#                     data=csv_open,
#                     file_name="portfolio_open.csv",
#                     mime="text/csv"
#                 )
#
#         with col3:
#             if not closed_positions.empty:
#                 csv_closed = convert_df(closed_positions)
#                 st.download_button(
#                     label="📥 Closed Positions",
#                     data=csv_closed,
#                     file_name="portfolio_closed.csv",
#                     mime="text/csv"
#                 )
#
#         with col4:
#             if not mtm_df.empty:
#                 csv_mtm = convert_df(mtm_df)
#                 st.download_button(
#                     label="📥 Daily MTM",
#                     data=csv_mtm,
#                     file_name=f"mtm_{datetime.now().strftime('%Y%m%d')}.csv",
#                     mime="text/csv"
#                 )
#
#         # Show footer note
#         if has_footer:
#             st.info(f"📝 Note: Used realized P&L from file footer: {format_currency_no_decimals(realized_pl_from_footer)}")
#
#     else:
#         st.error("No positions processed. Check your file format.")
#
# else:
#     # Instructions
#     st.info("👈 Upload your portfolio CSV file to begin")
#
#     st.subheader("📝 Expected Format")
#
#     sample_data = {
#         'ENTRY DATE': ['02-07-2025', '02-07-2025'],
#         'EXIT DATE': ['', '11-08-2025'],
#         'SCRIP': ['TORNTPHARM', 'DCAL'],
#         'Mcap': ['Large Cap', ''],
#         'Position': ['Open', 'Close'],
#         'QTY': [7, 98],
#         'ENTRY PRICE': [3431, 257],
#         'Closed price': ['', 235],
#         'closed position Profit/loss': ['', -2156]
#     }
#
#     sample_df = pd.DataFrame(sample_data)
#
#     with st.expander("View Sample Format"):
#         st.dataframe(sample_df)
#
# # Auto-refresh JavaScript (if enabled)
# if 'auto_refresh' in locals() and auto_refresh:
#     # Show refresh info in sidebar
#     current_time = datetime.now().strftime("%H:%M:%S")
#     next_refresh = (datetime.now() + timedelta(seconds=refresh_seconds)).strftime("%H:%M:%S")
#
#     st.sidebar.markdown("---")
#     st.sidebar.info(f"**Auto Refresh**\n\n🕒 Current: {current_time}\n🔄 Next: {next_refresh}")
#
#     # JavaScript for auto-refresh
#     st.markdown(f"""
#     <script>
#         setTimeout(function() {{
#             window.location.reload();
#         }}, {refresh_seconds * 1000});
#     </script>
#     """, unsafe_allow_html=True)
#
# # Footer
# st.markdown("---")
# st.markdown("""
# <div style="text-align: center">
#     <p>📊 <b>Portfolio Tracker</b> | {'Auto-refresh: Enabled' if 'auto_refresh' in locals() and auto_refresh else 'Auto-refresh: Disabled'}</p>
# </div>
# """, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings
import time

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Portfolio Tracker",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📊 Stock Portfolio Tracker")

# Sidebar with auto-refresh
with st.sidebar:
    st.header("🔄 Controls")

    # Auto-refresh settings
    auto_refresh = st.checkbox("Enable Auto Refresh", value=False)
    if auto_refresh:
        refresh_interval = st.selectbox("Refresh Interval",
                                        ["30 seconds", "1 minute", "5 minutes", "10 minutes"])
        # Convert to seconds
        interval_map = {
            "30 seconds": 30,
            "1 minute": 60,
            "5 minutes": 300,
            "10 minutes": 600
        }
        refresh_seconds = interval_map[refresh_interval]

    # Manual refresh button
    if st.button("🔄 Refresh Now", type="primary"):
        st.rerun()

    show_graphs = st.checkbox("Show Graphs", value=True)
    show_pie = st.checkbox("Show MCap Pie Chart", value=True)

    st.markdown("---")
    st.info("""
    **Your CSV Format:**
    - Open: Position = 'Open', no EXIT DATE
    - Close: Position = 'Close', has EXIT DATE
    - MCap only for open positions
    - Closed price for exited positions
    - Footer with total realized profit/loss
    """)


# Formatting functions - NO DECIMALS for currency values
def format_currency_no_decimals(value):
    """Format currency WITHOUT decimals for main metrics"""
    if pd.isna(value) or value is None:
        return "₹0"
    try:
        # Force to integer by removing decimals
        value_int = int(round(float(value)))
        return f"₹{value_int:,}"
    except:
        return "₹0"


def format_currency_with_decimals(value):
    """Format currency WITH decimals (for prices only)"""
    if pd.isna(value) or value is None:
        return ""
    try:
        return f"₹{float(value):,.2f}"
    except:
        return ""


def format_percentage(value):
    if pd.isna(value) or value is None:
        return ""
    try:
        return f"{float(value):.2f}%"
    except:
        return ""


# Fetch live price
@st.cache_data(ttl=60)  # Shorter cache for auto-refresh
def get_live_price(ticker):
    try:
        if not ticker.endswith(('.NS', '.BO', '.NSE')):
            ticker = f"{ticker}.NS"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
        return None
    except:
        return None


# Fetch yesterday's close price
@st.cache_data(ttl=300)
def get_yesterday_close(ticker):
    try:
        if not ticker.endswith(('.NS', '.BO', '.NSE')):
            ticker = f"{ticker}.NS"
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3d")
        if len(hist) >= 2:
            return hist['Close'].iloc[-2]
        elif len(hist) == 1:
            return hist['Close'].iloc[-1]
        return None
    except:
        return None


# Fetch historical price
@st.cache_data(ttl=3600)
def get_historical_price(ticker, date):
    try:
        if pd.isna(date):
            return None
        if not ticker.endswith(('.NS', '.BO', '.NSE')):
            ticker = f"{ticker}.NS"
        stock = yf.Ticker(ticker)
        start_date = date - timedelta(days=7)
        end_date = date + timedelta(days=1)
        hist = stock.history(start=start_date, end=end_date)
        if not hist.empty:
            hist.index = pd.to_datetime(hist.index).normalize()
            target_date = pd.Timestamp(date).normalize()
            if target_date in hist.index:
                return hist.loc[target_date, 'Close']
            else:
                dates = hist.index
                time_diff = abs(dates - target_date)
                nearest_idx = time_diff.argmin()
                return hist.iloc[nearest_idx]['Close']
        return None
    except:
        return None


# Calculate days held
def calculate_days_held(entry_date, exit_date):
    if pd.isna(entry_date):
        return 0
    if pd.isna(exit_date):
        return (datetime.now().date() - entry_date.date()).days
    else:
        return (exit_date.date() - entry_date.date()).days


# Calculate MTM for open positions
def calculate_daily_mtm(open_positions_df):
    if open_positions_df.empty:
        return pd.DataFrame(), 0, 0, 0

    mtm_data = []
    total_mtm = 0
    total_yesterday_value = 0
    total_today_value = 0

    for idx, row in open_positions_df.iterrows():
        ticker = row['Ticker']
        qty = row['Quantity']
        today_price = row['Current Price']
        yesterday_price = get_yesterday_close(ticker)

        if today_price and yesterday_price and qty > 0:
            today_value = qty * today_price
            yesterday_value = qty * yesterday_price
            daily_mtm = today_value - yesterday_value
            daily_mtm_percent = (daily_mtm / yesterday_value * 100) if yesterday_value > 0 else 0

            mtm_data.append({
                'Ticker': ticker,
                'Quantity': qty,
                'Yesterday Close': yesterday_price,
                'Today Price': today_price,
                'Yesterday Value': yesterday_value,
                'Today Value': today_value,
                'Daily MTM': daily_mtm,
                'Daily MTM %': daily_mtm_percent
            })

            total_mtm += daily_mtm
            total_yesterday_value += yesterday_value
            total_today_value += today_value

    mtm_df = pd.DataFrame(mtm_data)
    portfolio_mtm_percent = (total_mtm / total_yesterday_value * 100) if total_yesterday_value > 0 else 0

    return mtm_df, total_mtm, portfolio_mtm_percent, total_yesterday_value


# Main app
uploaded_file = st.file_uploader("📁 Upload your portfolio CSV file", type=['csv'])

if uploaded_file is not None:
    # Read and process file
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # Show file info
    st.info(f"📄 File: {uploaded_file.name} | 📊 Rows: {len(df)}")

    # Check for footer rows (realized profit/loss)
    has_footer = False
    realized_pl_from_footer = 0
    c_f_profit = 0

    # Check the last row for "realized profit/loss"
    if len(df) > 0:
        last_row = df.iloc[-1]

        # Check if last row contains "realized profit/loss" in any column
        for col in df.columns:
            if isinstance(last_row[col], str) and 'realized profit/loss' in last_row[col].lower():
                has_footer = True
                # Try to get the value from 'closed position Profit/loss' column
                if 'closed position Profit/loss' in df.columns:
                    realized_pl_from_footer = last_row['closed position Profit/loss']
                break

        # Also check for C/F Profit in second last row
        if len(df) > 1:
            second_last_row = df.iloc[-2]
            if 'closed position Profit/loss' in df.columns and pd.notna(second_last_row['closed position Profit/loss']):
                # Check if it's the C/F profit row (usually has no ticker)
                if pd.isna(second_last_row.get('SCRIP')) or second_last_row.get('SCRIP', '') == '':
                    c_f_profit = second_last_row['closed position Profit/loss']

    # Remove footer rows from main data
    if has_footer:
        df = df.iloc[:-1]  # Remove last row (realized profit/loss)

    # Also remove the C/F profit row if it exists
    if len(df) > 0 and pd.isna(df.iloc[-1].get('SCRIP')) or (len(df) > 0 and df.iloc[-1].get('SCRIP', '') == ''):
        df = df.iloc[:-1]

    # Remove empty rows
    df = df.dropna(subset=['SCRIP'], how='all')
    df = df[df['SCRIP'].notna() & (df['SCRIP'] != '')]

    # Clean data
    df = df.replace('', np.nan)


    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT
        try:
            return datetime.strptime(str(date_str), '%d-%m-%Y')
        except:
            try:
                return datetime.strptime(str(date_str), '%Y-%m-%d')
            except:
                return pd.NaT


    if 'ENTRY DATE' in df.columns:
        df['ENTRY DATE'] = df['ENTRY DATE'].apply(parse_date)
    if 'EXIT DATE' in df.columns:
        df['EXIT DATE'] = df['EXIT DATE'].apply(parse_date)

    numeric_cols = ['QTY', 'ENTRY PRICE', 'Closed price', 'closed position Profit/loss']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    text_cols = ['SCRIP', 'Mcap', 'Position']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    df = df[df['SCRIP'].notna() & (df['SCRIP'] != 'nan')]

    # Process positions
    results = []
    total_investment = 0
    unrealized_pl = 0

    with st.spinner("🔄 Processing positions and fetching live prices..."):
        for idx, row in df.iterrows():
            ticker = str(row['SCRIP']).strip()

            if pd.isna(row['ENTRY DATE']) or pd.isna(row['QTY']) or pd.isna(row['ENTRY PRICE']):
                continue

            position_type = str(row.get('Position', '')).upper()
            is_open = position_type == 'OPEN'
            is_closed = position_type == 'CLOSE'

            if pd.isna(position_type) or position_type == 'NAN':
                is_open = pd.isna(row.get('EXIT DATE'))
                is_closed = not pd.isna(row.get('EXIT DATE'))

            days_held = calculate_days_held(row['ENTRY DATE'], row.get('EXIT DATE'))

            if is_open:
                current_price = get_live_price(ticker)
                entry_value = row['QTY'] * row['ENTRY PRICE']

                if current_price:
                    current_position_value = row['QTY'] * current_price
                    position_unrealized_pl = current_position_value - entry_value
                else:
                    current_position_value = entry_value
                    position_unrealized_pl = 0

                pnl_percent = (position_unrealized_pl / entry_value * 100) if entry_value != 0 else 0

                results.append({
                    'Ticker': ticker,
                    'Status': 'Open',
                    'Entry Date': row['ENTRY DATE'],
                    'Exit Date': None,
                    'Quantity': row['QTY'],
                    'Entry Price': row['ENTRY PRICE'],
                    'Current Price': current_price,
                    'Entry Value': entry_value,
                    'Current Value': current_position_value,
                    'P&L': position_unrealized_pl,
                    'P&L %': pnl_percent,
                    'Mcap': row.get('Mcap', 'Unknown'),
                    'Days Held': days_held,
                    'Type': 'Unrealized'
                })

                total_investment += entry_value
                unrealized_pl += position_unrealized_pl

            elif is_closed:
                exit_price = row.get('Closed price')

                if pd.isna(exit_price) and not pd.isna(row.get('EXIT DATE')):
                    exit_price = get_historical_price(ticker, row['EXIT DATE'])

                if exit_price and not pd.isna(exit_price):
                    entry_value = row['QTY'] * row['ENTRY PRICE']
                    exit_value = row['QTY'] * exit_price
                    position_pl = exit_value - entry_value
                    pnl_percent = (position_pl / entry_value * 100) if entry_value != 0 else 0

                    provided_pl = row.get('closed position Profit/loss')
                    if not pd.isna(provided_pl):
                        position_pl = provided_pl

                    results.append({
                        'Ticker': ticker,
                        'Status': 'Closed',
                        'Entry Date': row['ENTRY DATE'],
                        'Exit Date': row['EXIT DATE'],
                        'Quantity': row['QTY'],
                        'Entry Price': row['ENTRY PRICE'],
                        'Exit Price': exit_price,
                        'Entry Value': entry_value,
                        'Exit Value': exit_value,
                        'Current Value': exit_value,
                        'P&L': position_pl,
                        'P&L %': pnl_percent,
                        'Mcap': row.get('Mcap', ''),
                        'Days Held': days_held,
                        'Type': 'Booked/Realized'
                    })

                    total_investment += entry_value

    # Create results dataframe
    if results:
        results_df = pd.DataFrame(results)

        # Fill NaN values
        numeric_cols_results = ['Entry Price', 'Exit Price', 'Current Price', 'Entry Value',
                                'Exit Value', 'Current Value', 'P&L', 'P&L %', 'Days Held']
        for col in numeric_cols_results:
            if col in results_df.columns:
                results_df[col] = results_df[col].fillna(0)

        # Convert dates
        for col in ['Entry Date', 'Exit Date']:
            if col in results_df.columns:
                results_df[col] = pd.to_datetime(results_df[col], errors='coerce')

        # Separate open and closed
        open_positions = results_df[results_df['Status'] == 'Open']
        closed_positions = results_df[results_df['Status'] == 'Closed']

        # Calculate realized P&L - USE FOOTER VALUE IF AVAILABLE
        if has_footer and abs(realized_pl_from_footer) > 0:
            # Use footer value for realized P&L (includes C/F profit)
            realized_pl = realized_pl_from_footer
        else:
            # Calculate from closed positions
            realized_pl = closed_positions['P&L'].sum() if not closed_positions.empty else 0

        # Unrealized P&L is already calculated during processing
        total_pl = unrealized_pl + realized_pl

        # PORTFOLIO VALUE - Only open positions
        portfolio_value_raw = open_positions['Current Value'].sum() if not open_positions.empty else 0
        portfolio_value = int(round(portfolio_value_raw))

        # Profitable/loss counts
        profitable_all = results_df[results_df['P&L'] > 0]
        loss_all = results_df[results_df['P&L'] < 0]

        profitable_open = open_positions[open_positions['P&L'] > 0] if not open_positions.empty else pd.DataFrame()
        loss_open = open_positions[open_positions['P&L'] < 0] if not open_positions.empty else pd.DataFrame()

        profitable_closed = closed_positions[closed_positions['P&L'] > 0] if not closed_positions.empty else pd.DataFrame()
        loss_closed = closed_positions[closed_positions['P&L'] < 0] if not closed_positions.empty else pd.DataFrame()

        # TURNOVER CALCULATION
        # For open positions: Entry value
        # For closed positions: Entry value + Exit value
        open_turnover = open_positions['Entry Value'].sum() if not open_positions.empty else 0
        closed_entry_turnover = closed_positions['Entry Value'].sum() if not closed_positions.empty else 0
        closed_exit_turnover = closed_positions['Exit Value'].sum() if not closed_positions.empty else 0
        turnover_raw = open_turnover + closed_entry_turnover + closed_exit_turnover
        turnover = int(round(turnover_raw))

        # Show turnover breakdown
        with st.expander("📊 Turnover Breakdown"):
            st.write(f"Open Positions Entry Value: {format_currency_no_decimals(open_turnover)}")
            st.write(f"Closed Positions Entry Value: {format_currency_no_decimals(closed_entry_turnover)}")
            st.write(f"Closed Positions Exit Value: {format_currency_no_decimals(closed_exit_turnover)}")
            st.write(f"**Total Turnover: {format_currency_no_decimals(turnover)}**")

        # Average days held
        avg_days_held = results_df['Days Held'].mean() if not results_df.empty else 0

        # Annualized return - based on portfolio value
        if portfolio_value > 0 and avg_days_held > 0:
            total_return_pct = (total_pl / portfolio_value) * 100
            annualized_return = ((1 + total_return_pct / 100) ** (365 / avg_days_held) - 1) * 100
        else:
            annualized_return = 0

        # Win rate
        win_rate = (len(profitable_all) / len(results_df) * 100) if len(results_df) > 0 else 0

        # Calculate Daily MTM
        with st.spinner("📈 Calculating daily MTM..."):
            mtm_df, total_mtm, portfolio_mtm_percent, total_yesterday_value = calculate_daily_mtm(open_positions)

        # Display Portfolio Summary
        st.subheader("📊 Portfolio Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_pl_int = int(round(total_pl))
            portfolio_value_int = int(round(portfolio_value))
            st.metric("💰 Total P&L", format_currency_no_decimals(total_pl_int),
                      f"{(total_pl / portfolio_value * 100):.2f}%" if portfolio_value > 0 else "0%")
            st.metric("📊 Portfolio Value", format_currency_no_decimals(portfolio_value_int))

        with col2:
            realized_pl_int = int(round(realized_pl))
            unrealized_pl_int = int(round(unrealized_pl))
            st.metric("📈 Realized P&L", format_currency_no_decimals(realized_pl_int))
            st.metric("📉 Unrealized P&L", format_currency_no_decimals(unrealized_pl_int))

        with col3:
            st.metric("🎯 Annualized Return", f"{annualized_return:.2f}%")
            st.metric("📅 Avg Days Held", f"{avg_days_held:.1f}")

        with col4:
            turnover_int = int(round(turnover))
            st.metric("📊 Turnover", format_currency_no_decimals(turnover_int))
            st.metric("📈 Win Rate", f"{win_rate:.1f}%")

        # Position Statistics
        st.subheader("📊 Position Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("💹 All Profitable", len(profitable_all))
            st.metric("🔻 All Loss", len(loss_all))

        with col2:
            st.metric("✅ Open Positions", len(open_positions))
            st.metric("🏁 Closed Positions", len(closed_positions))

        with col3:
            if not open_positions.empty:
                st.metric("📈 Open Profitable", len(profitable_open))
                st.metric("📉 Open Loss", len(loss_open))

        with col4:
            if not closed_positions.empty:
                st.metric("💰 Closed Profitable", len(profitable_closed))
                st.metric("🔻 Closed Loss", len(loss_closed))

        # Top & Bottom Performers - Sorted by percentage
        st.subheader("🏆 Top & Bottom Performers")

        if not results_df.empty:
            # Get top 5 by percentage gain (excluding positions with 0% or NaN)
            valid_pnl = results_df[results_df['P&L %'].notna() & (results_df['P&L %'] != 0)]

            if not valid_pnl.empty:
                top_5_pct = valid_pnl.nlargest(5, 'P&L %')[['Ticker', 'P&L', 'P&L %', 'Status']].copy()
                bottom_5_pct = valid_pnl.nsmallest(5, 'P&L %')[['Ticker', 'P&L', 'P&L %', 'Status']].copy()

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**🥇 Top 5 by % Gain**")
                    if not top_5_pct.empty:
                        for idx, row in top_5_pct.iterrows():
                            st.success(f"**{row['Ticker']}**: {format_currency_no_decimals(row['P&L'])} ({format_percentage(row['P&L %'])})")
                    else:
                        st.info("No profitable positions")

                with col2:
                    st.write("**🔻 Bottom 5 by % Loss**")
                    if not bottom_5_pct.empty:
                        for idx, row in bottom_5_pct.iterrows():
                            st.error(f"**{row['Ticker']}**: {format_currency_no_decimals(row['P&L'])} ({format_percentage(row['P&L %'])})")
                    else:
                        st.info("No loss positions")

        # Portfolio Analysis Tabs
        st.subheader("📋 Portfolio Analysis")


        def format_for_display(df):
            if df.empty:
                return df

            formatted = df.copy()

            # Format dates
            for col in ['Entry Date', 'Exit Date']:
                if col in formatted.columns:
                    formatted[col] = pd.to_datetime(formatted[col]).dt.strftime('%Y-%m-%d')

            # Format currency - DIFFERENT FOR PRICES VS VALUES
            price_cols = ['Entry Price', 'Exit Price', 'Current Price']
            value_cols = ['Entry Value', 'Exit Value', 'Current Value', 'P&L']

            for col in price_cols:
                if col in formatted.columns:
                    formatted[col] = formatted[col].apply(format_currency_with_decimals)

            for col in value_cols:
                if col in formatted.columns:
                    formatted[col] = formatted[col].apply(format_currency_no_decimals)

            # Format percentages
            if 'P&L %' in formatted.columns:
                formatted['P&L %'] = formatted['P&L %'].apply(format_percentage)

            formatted = formatted.fillna('')
            return formatted


        tab1, tab2, tab3, tab4 = st.tabs([
            f"All Positions ({len(results_df)})",
            f"Open Positions ({len(open_positions)})",
            f"Closed Positions ({len(closed_positions)})",
            f"Daily MTM ({datetime.now().strftime('%d %b')})"
        ])

        with tab1:
            display_all = format_for_display(results_df)
            st.dataframe(display_all, use_container_width=True)

        with tab2:
            if not open_positions.empty:
                display_open = format_for_display(open_positions)
                st.dataframe(display_open, use_container_width=True)
            else:
                st.info("No open positions")

        with tab3:
            if not closed_positions.empty:
                display_closed = format_for_display(closed_positions)
                st.dataframe(display_closed, use_container_width=True)
            else:
                st.info("No closed positions")

        with tab4:
            st.subheader(f"📈 Daily Mark-to-Market ({datetime.now().strftime('%d %b %Y')})")

            if not mtm_df.empty:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("📊 Today's MTM", format_currency_no_decimals(total_mtm),
                              f"{portfolio_mtm_percent:.2f}%")

                with col2:
                    st.metric("💰 Yesterday's Value", format_currency_no_decimals(total_yesterday_value))

                with col3:
                    gainers = len(mtm_df[mtm_df['Daily MTM'] > 0])
                    st.metric("📈 Gainers", gainers)

                with col4:
                    losers = len(mtm_df[mtm_df['Daily MTM'] < 0])
                    st.metric("📉 Losers", losers)

                # Format MTM table
                display_mtm = mtm_df.copy()
                display_mtm['Yesterday Close'] = display_mtm['Yesterday Close'].apply(format_currency_with_decimals)
                display_mtm['Today Price'] = display_mtm['Today Price'].apply(format_currency_with_decimals)
                display_mtm['Yesterday Value'] = display_mtm['Yesterday Value'].apply(format_currency_no_decimals)
                display_mtm['Today Value'] = display_mtm['Today Value'].apply(format_currency_no_decimals)
                display_mtm['Daily MTM'] = display_mtm['Daily MTM'].apply(format_currency_no_decimals)
                display_mtm['Daily MTM %'] = display_mtm['Daily MTM %'].apply(lambda x: f"{x:.2f}%")

                st.dataframe(display_mtm, use_container_width=True)
            else:
                if open_positions.empty:
                    st.info("No open positions for MTM calculation")
                else:
                    st.warning("Could not fetch yesterday's prices")

        # MCap Exposure
        st.subheader("🏢 Market Cap Exposure (Open Positions Only)")

        if not open_positions.empty and 'Mcap' in open_positions.columns:
            open_positions['Mcap'] = open_positions['Mcap'].astype(str).str.strip()
            open_positions['Mcap'] = open_positions['Mcap'].replace(['', 'nan', 'NaN', 'None'], 'Unknown')

            mcap_exposure = open_positions.groupby('Mcap').agg({
                'Current Value': 'sum',
                'P&L': 'sum',
                'Ticker': 'count'
            }).rename(columns={'Ticker': 'Count'}).reset_index()

            if not mcap_exposure.empty:
                if show_pie:
                    fig = px.pie(mcap_exposure,
                                 values='Current Value',
                                 names='Mcap',
                                 title='MCap Distribution - Open Positions',
                                 hole=0.3)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)

                # Display table
                open_total = open_positions['Current Value'].sum()
                mcap_exposure['% of Portfolio'] = (mcap_exposure['Current Value'] / open_total * 100).round(2)

                display_table = mcap_exposure.copy()
                display_table['Current Value'] = display_table['Current Value'].apply(format_currency_no_decimals)
                display_table['P&L'] = display_table['P&L'].apply(format_currency_no_decimals)
                display_table['% of Portfolio'] = display_table['% of Portfolio'].apply(lambda x: f"{x:.1f}%")

                st.dataframe(display_table[['Mcap', 'Count', 'Current Value', 'P&L', '% of Portfolio']])

        # Export
        st.subheader("💾 Export Results")


        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            csv_all = convert_df(results_df)
            st.download_button(
                label="📥 All Positions",
                data=csv_all,
                file_name="portfolio_all.csv",
                mime="text/csv"
            )

        with col2:
            if not open_positions.empty:
                csv_open = convert_df(open_positions)
                st.download_button(
                    label="📥 Open Positions",
                    data=csv_open,
                    file_name="portfolio_open.csv",
                    mime="text/csv"
                )

        with col3:
            if not closed_positions.empty:
                csv_closed = convert_df(closed_positions)
                st.download_button(
                    label="📥 Closed Positions",
                    data=csv_closed,
                    file_name="portfolio_closed.csv",
                    mime="text/csv"
                )

        with col4:
            if not mtm_df.empty:
                csv_mtm = convert_df(mtm_df)
                st.download_button(
                    label="📥 Daily MTM",
                    data=csv_mtm,
                    file_name=f"mtm_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

        # Show footer note
        if has_footer:
            st.success(f"✅ Using realized P&L from file footer: {format_currency_no_decimals(realized_pl_from_footer)}")
            if c_f_profit != 0:
                st.info(f"📝 C/F Profit from previous period: {format_currency_no_decimals(c_f_profit)}")

    else:
        st.error("No positions processed. Check your file format.")

else:
    # Instructions
    st.info("👈 Upload your portfolio CSV file to begin")

    st.subheader("📝 Expected Format")

    sample_data = {
        'ENTRY DATE': ['02-07-2025', '02-07-2025'],
        'EXIT DATE': ['', '11-08-2025'],
        'SCRIP': ['TORNTPHARM', 'DCAL'],
        'Mcap': ['Large Cap', ''],
        'Position': ['Open', 'Close'],
        'QTY': [7, 98],
        'ENTRY PRICE': [3431, 257],
        'Closed price': ['', 235],
        'closed position Profit/loss': ['', -2156]
    }

    sample_df = pd.DataFrame(sample_data)

    with st.expander("View Sample Format"):
        st.dataframe(sample_df)

# Auto-refresh JavaScript (if enabled)
if 'auto_refresh' in locals() and auto_refresh:
    # Show refresh info in sidebar
    current_time = datetime.now().strftime("%H:%M:%S")
    next_refresh = (datetime.now() + timedelta(seconds=refresh_seconds)).strftime("%H:%M:%S")

    st.sidebar.markdown("---")
    st.sidebar.info(f"**Auto Refresh**\n\n🕒 Current: {current_time}\n🔄 Next: {next_refresh}")

    # JavaScript for auto-refresh
    st.markdown(f"""
    <script>
        setTimeout(function() {{
            window.location.reload();
        }}, {refresh_seconds * 1000});
    </script>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>📊 <b>Portfolio Tracker</b> | {'Auto-refresh: Enabled' if 'auto_refresh' in locals() and auto_refresh else 'Auto-refresh: Disabled'}</p>
</div>
""", unsafe_allow_html=True)


