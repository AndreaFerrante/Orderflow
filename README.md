-----------------------------------------------------------
This is a short Python module to analyze tick-by-tick data.
-----------------------------------------------------------

The structure of the data is the following:

- <b>Date</b>: date of trade execution (YYYY-MM-DD)
- <b>Time</b>: time of trade execution (%H:%M:%S.ffffff)
- <b>Price</b>: price of execution trade
- <b>Volume</b>: volume executed
- <b>TradeType</b>: 1 if trade has been executed on the BID, 2 if the trade has been executed on the ASK
- <b>AskPrice</b>: price on the ask at the time of the execution
- <b>BidPrice</b>: price on the bid at the time of the execution
- <b>AskSize</b>: level 1 DOM (i.e Depth of the Market) size on the ask
- <b>BidSize</b>: level 1 DOM (i.e Depth of the Market) size on the bid
- <b>TotalAskDepth</b>: total volume on the ask DOM side (till max level available)
- <b>TotalBidDepth</b>: total volume on the bid DOM side (till max level available)
- <b>AskDOM_XX / BidDOM_XX</b>: ask and bid values till level XX

Main tools in development for data fun and research are <b>VWAP, Volume Profile, Imbalances</b>.
Please, keep in mind that this is not a module for trading but <b>just</b> for research for those that love data !
