
TIME = 'time'
IS_CARRIED = 'isCarried'

LOW = 'low'
HIGH = 'high'
OPEN = 'open'
CLOSE = 'close'
VOLUME = 'volume'

NUM_TRADES = 'numTrades'
WEIGHTED_AVERAGE = 'weightedAverage'

NUM_BLOCKS = 'numBlocks'
AVERAGE_DIFFICULTY = 'avgDifficulty'
AVERAGE_TOTAL_DIFFICULTY = 'avgTotalDifficulty'
GAS_USED = 'gasUsed'
BLOCK_SIZE = 'blockSize'
NUM_TRANSACTIONS = 'numTransactions'
VALUE = 'value'
NUM_CONTRACTS = 'numContracts'
CONTRACT_GAS = 'contractGas'
CONTRACT_VALUE = 'contractValue'

ETHE_BLOCK = 'ethereum'
GDAX_CHART = 'gdax_chart'
KRAK_TRADE = 'krak_trade'
POLO_TRADE = 'polo_trade'

LABELS = {
    ETHE_BLOCK: [TIME,
                 NUM_BLOCKS,
                 AVERAGE_DIFFICULTY,
                 AVERAGE_TOTAL_DIFFICULTY,
                 GAS_USED,
                 BLOCK_SIZE,
                 NUM_TRANSACTIONS,
                 VALUE,
                 NUM_CONTRACTS,
                 CONTRACT_GAS,
                 CONTRACT_VALUE],
    GDAX_CHART: [TIME,
                 LOW,
                 HIGH,
                 OPEN,
                 CLOSE,
                 VOLUME,
                 IS_CARRIED],
    KRAK_TRADE: [TIME,
                 LOW,
                 HIGH,
                 OPEN,
                 CLOSE,
                 VOLUME,
                 NUM_TRADES,
                 WEIGHTED_AVERAGE,
                 IS_CARRIED],
    POLO_TRADE: [TIME,
                 LOW,
                 HIGH,
                 OPEN,
                 CLOSE,
                 VOLUME,
                 NUM_TRADES,
                 WEIGHTED_AVERAGE,
                 IS_CARRIED]
}

AUG2016 = {
    GDAX_CHART: ['ETH-USD', 'ETH-BTC', 'BTC-USD', 'BTC-EUR'],
    POLO_TRADE: ['BTC_AMP', 'BTC_BCN', 'BTC_BCY', 'BTC_BELA', 'BTC_BLK',
                 'BTC_BTCD', 'BTC_BTM', 'BTC_BTS', 'BTC_BURST', 'BTC_CLAM',
                 'BTC_DASH', 'BTC_DCR', 'BTC_DGB', 'BTC_DOGE', 'BTC_EMC2',
                 'BTC_ETC', 'BTC_ETH', 'BTC_EXP', 'BTC_FCT', 'BTC_FLDC',
                 'BTC_FLO', 'BTC_GAME', 'BTC_GRC', 'BTC_HUC', 'BTC_LBC',
                 'BTC_LSK', 'BTC_LTC', 'BTC_MAID', 'BTC_NAV', 'BTC_NEOS',
                 'BTC_NMC', 'BTC_NXT', 'BTC_OMNI', 'BTC_PINK', 'BTC_POT',
                 'BTC_PPC', 'BTC_RADS', 'BTC_RIC', 'BTC_SBD', 'BTC_SC',
                 'BTC_STEEM', 'BTC_STR', 'BTC_SYS', 'BTC_VIA', 'BTC_VRC',
                 'BTC_VTC', 'BTC_XBC', 'BTC_XCP', 'BTC_XEM', 'BTC_XMR',
                 'BTC_XPM', 'BTC_XRP', 'BTC_XVC', 'ETH_ETC', 'ETH_LSK',
                 'ETH_STEEM', 'USDT_BTC', 'USDT_DASH', 'USDT_ETC', 'USDT_ETH',
                 'USDT_LTC', 'USDT_NXT', 'USDT_STR', 'USDT_XMR', 'USDT_XRP',
                 'XMR_BCN', 'XMR_BLK', 'XMR_BTCD', 'XMR_DASH', 'XMR_LTC',
                 'XMR_MAID', 'XMR_NXT'],
    KRAK_TRADE: ['XETCXETH', 'XETCXXBT', 'XETCZEUR', 'XETCZUSD', 'XETHXXBT',
                 'XETHZCAD', 'XETHZEUR', 'XETHZGBP', 'XETHZJPY', 'XETHZUSD',
                 'XLTCXXBT', 'XLTCZEUR', 'XLTCZUSD', 'XXBTZCAD', 'XXBTZEUR',
                 'XXBTZGBP', 'XXBTZJPY', 'XXBTZUSD', 'XXDGXXBT', 'XXLMXXBT',
                 'XXRPXXBT'],
    ETHE_BLOCK: [ETHE_BLOCK]
    # TODO: add bitfinex chart data
    #     BITF_CHART: ['BTCUSD', 'LTCUSD', 'LTCBTC', 'ETHUSD', 'ETHBTC', 'ETCBTC',
    #                  'ETCUSD'],
}
