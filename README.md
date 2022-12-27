This is a short Python module to analyze tick-by-tick data.

The structure of the data is the following:

- Date: date of trade execution (YYYY-MM-DD)
- Time: time of trade execution (%H:%M:%S.ffffff)
- Price: price of execution trade
- Volume: volume executed
- TradeType: 1 if trade has been executed on the BID, 2 if the trade has been executed on the ASK
- AskPrice: price on the ask at the time of the execution
- BidPrice: price on the bid at the time of the execution
- AskSize: level 1 DOM (i.e Depth of the Market) size on the ask
- BidSize: level 1 DOM (i.e Depth of the Market) size on the bid
- TotalAskDepth: total volume on the ask DOM side (till max level available)
- TotalBidDepth: total volume on the bid DOM side (till max level available)
- AskDOM_XX / BidDOM_XX: ask and bid values till level XX
