CREATE TABLE futures.ticks_enriched (
        symbol          VARCHAR(10)     NOT NULL,
        month_key       VARCHAR(6)      NOT NULL,
        "Date" TEXT             NOT NULL,
        "Time" TEXT             NOT NULL,
        "Sequence" BIGINT       NOT NULL,
        "DepthSequence" BIGINT,
        "Price" FLOAT(53),
        "Volume" BIGINT,
        "TradeType" BIGINT,
        "AskPrice" FLOAT(53),
        "BidPrice" FLOAT(53),
        "AskSize" BIGINT,
        "BidSize" BIGINT,
        "TotalAskDepth" BIGINT,
        "TotalBidDepth" BIGINT,
        "AskDOMPrice" FLOAT(53),
        "BidDOMPrice" FLOAT(53),
        "AskDOM_0" BIGINT,
        "BidDOM_0" BIGINT,
        "AskDOM_1" BIGINT,
        "BidDOM_1" BIGINT,
        "AskDOM_2" BIGINT,
        "BidDOM_2" BIGINT,
        "AskDOM_3" BIGINT,
        "BidDOM_3" BIGINT,
        "AskDOM_4" BIGINT,
        "BidDOM_4" BIGINT,
        "AskDOM_5" BIGINT,
        "BidDOM_5" BIGINT,
        "AskDOM_6" BIGINT,
        "BidDOM_6" BIGINT,
        "AskDOM_7" BIGINT,
        "BidDOM_7" BIGINT,
        "AskDOM_8" BIGINT,
        "BidDOM_8" BIGINT,
        "AskDOM_9" BIGINT,
        "BidDOM_9" BIGINT,
        "AskDOM_10" BIGINT,
        "BidDOM_10" BIGINT,
        "AskDOM_11" BIGINT,
        "BidDOM_11" BIGINT,
        "AskDOM_12" BIGINT,
        "BidDOM_12" BIGINT,
        "AskDOM_13" BIGINT,
        "BidDOM_13" BIGINT,
        "AskDOM_14" BIGINT,
        "BidDOM_14" BIGINT,
        "AskDOM_15" BIGINT,
        "BidDOM_15" BIGINT,
        "AskDOM_16" BIGINT,
        "BidDOM_16" BIGINT,
        "AskDOM_17" BIGINT,
        "BidDOM_17" BIGINT,
        "AskDOM_18" BIGINT,
        "BidDOM_18" BIGINT,
        "AskDOM_19" BIGINT,
        "BidDOM_19" BIGINT,
        "AskDOM_20" BIGINT,
        "BidDOM_20" BIGINT,
        "AskDOM_21" BIGINT,
        "BidDOM_21" BIGINT,
        "AskDOM_22" BIGINT,
        "BidDOM_22" BIGINT,
        "AskDOM_23" BIGINT,
        "BidDOM_23" BIGINT,
        "AskDOM_24" BIGINT,
        "BidDOM_24" BIGINT,
        "AskDOM_25" BIGINT,
        "BidDOM_25" BIGINT,
        "AskDOM_26" BIGINT,
        "BidDOM_26" BIGINT,
        "AskDOM_27" BIGINT,
        "BidDOM_27" BIGINT,
        "AskDOM_28" BIGINT,
        "BidDOM_28" BIGINT,
        "AskDOM_29" BIGINT,
        "BidDOM_29" BIGINT,
        "Datetime" TIMESTAMP WITHOUT TIME ZONE,
        "Hour" SMALLINT,
        "SessionType" TEXT,
        "POC" FLOAT(53),
        "Prev_POC" FLOAT(53),
        "VA_Areas" TEXT,
        "ValleysPeaks" FLOAT(53),
        "CD_Ask" FLOAT(53),
        "CD_Bid" FLOAT(53),
        "CD_Total" FLOAT(53),
        "Session_High" FLOAT(53),
        "Session_Low" FLOAT(53),
        "Node_Volume" FLOAT(53),
        "Node_Ask_Volume" FLOAT(53),
        "Node_Bid_Volume" FLOAT(53),
        "Session_Volume" FLOAT(53),
        "LVN" SMALLINT,
        "Index" BIGINT,
        current_bar_open FLOAT(53),
        current_bar_high FLOAT(53),
        current_bar_low FLOAT(53),
        current_bar_close FLOAT(53),
        current_bar_volume BIGINT,
        current_bar_numberoftrades BIGINT,
        current_bar_askvolume BIGINT,
        current_bar_bidvolume BIGINT,
        current_bar_datetime TIMESTAMP WITHOUT TIME ZONE,
        next_bar_datetime TIMESTAMP WITHOUT TIME ZONE,
        next_bar_open FLOAT(53),
        next_bar_high FLOAT(53),
        next_bar_low FLOAT(53),
        next_bar_close FLOAT(53),
        next_bar_volume BIGINT,
        next_bar_num_trades BIGINT,
        next_bar_ask_volume BIGINT,
        next_bar_bid_volume BIGINT,
        vwap FLOAT(53),
        vwap_sd1_top FLOAT(53),
        vwap_sd1_bottom FLOAT(53),
        vwap_sd2_top FLOAT(53),
        vwap_sd2_bottom FLOAT(53),
        vwap_sd3_top FLOAT(53),
        vwap_sd3_bottom FLOAT(53),
        vwap_sd4_top FLOAT(53),
        vwap_sd4_bottom FLOAT(53),        
        created_at      TIMESTAMP       NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMP       NOT NULL DEFAULT NOW()
) PARTITION BY LIST (symbol);

ALTER TABLE futures.ticks_enriched
    ADD CONSTRAINT pk_ticks_enriched
    PRIMARY KEY (symbol, month_key, "Date", "Time", "Sequence");

CREATE TABLE futures.ticks_enriched_es PARTITION OF futures.ticks_enriched
    FOR VALUES IN ('ES')
    PARTITION BY RANGE (month_key);

CREATE TABLE futures.ticks_enriched_es_202501 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202501') TO ('202502');
CREATE TABLE futures.ticks_enriched_es_202502 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202502') TO ('202503');
CREATE TABLE futures.ticks_enriched_es_202503 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202503') TO ('202504');
CREATE TABLE futures.ticks_enriched_es_202504 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202504') TO ('202505');
CREATE TABLE futures.ticks_enriched_es_202505 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202505') TO ('202506');
CREATE TABLE futures.ticks_enriched_es_202506 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202506') TO ('202507');
CREATE TABLE futures.ticks_enriched_es_202507 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202507') TO ('202508');
CREATE TABLE futures.ticks_enriched_es_202508 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202508') TO ('202509');
CREATE TABLE futures.ticks_enriched_es_202509 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202509') TO ('202510');
CREATE TABLE futures.ticks_enriched_es_202510 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202510') TO ('202511');
CREATE TABLE futures.ticks_enriched_es_202511 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202511') TO ('202512');
CREATE TABLE futures.ticks_enriched_es_202512 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202512') TO ('202601');
CREATE TABLE futures.ticks_enriched_es_202601 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202601') TO ('202602');
CREATE TABLE futures.ticks_enriched_es_202602 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202602') TO ('202603');
CREATE TABLE futures.ticks_enriched_es_202603 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202603') TO ('202604');
CREATE TABLE futures.ticks_enriched_es_202604 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202604') TO ('202605');
CREATE TABLE futures.ticks_enriched_es_202605 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202605') TO ('202606');
CREATE TABLE futures.ticks_enriched_es_202606 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202606') TO ('202607');
CREATE TABLE futures.ticks_enriched_es_202607 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202607') TO ('202608');
CREATE TABLE futures.ticks_enriched_es_202608 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202608') TO ('202609');
CREATE TABLE futures.ticks_enriched_es_202609 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202609') TO ('202610');
CREATE TABLE futures.ticks_enriched_es_202610 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202610') TO ('202611');
CREATE TABLE futures.ticks_enriched_es_202611 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202611') TO ('202612');
CREATE TABLE futures.ticks_enriched_es_202612 PARTITION OF futures.ticks_enriched_es
    FOR VALUES FROM ('202612') TO ('202701');
