"""
Centralized IBKR Utilities

Provides common contract building and qualification functions
used across backfill, live ingestion, and paper trading modules.
"""
from ib_insync import Contract, Forex, Index, Future, ContFuture, Stock


def build_contract_from_config(t_conf: dict) -> Contract:
    """
    Build an IB contract from a target configuration dictionary.

    Args:
        t_conf: Dictionary with keys: secType, symbol, exchange, currency,
                and optionally lastTradeDate (for FUT)

    Returns:
        Unqualified Contract object

    Raises:
        ValueError: If secType is unknown
    """
    sec_type = t_conf.get("secType")

    if sec_type == "CASH":
        return Forex(t_conf["symbol"] + t_conf["currency"], exchange=t_conf["exchange"])
    elif sec_type == "IND":
        return Index(t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
    elif sec_type == "FUT":
        return Future(
            symbol=t_conf["symbol"],
            lastTradeDateOrContractMonth=t_conf.get("lastTradeDate"),
            exchange=t_conf["exchange"],
            currency=t_conf["currency"]
        )
    elif sec_type == "CONTFUT":
        return ContFuture(
            symbol=t_conf["symbol"],
            exchange=t_conf["exchange"],
            currency=t_conf["currency"]
        )
    elif sec_type == "STK":
        return Stock(t_conf["symbol"], t_conf["exchange"], t_conf["currency"])
    else:
        raise ValueError(f"Unknown secType: {sec_type}")


async def qualify_contract(ib, contract, logger=None):
    """
    Qualify a contract with IBKR.

    Args:
        ib: Connected IB instance
        contract: Contract to qualify
        logger: Optional logger for error messages

    Returns:
        Qualified contract or None if qualification failed
    """
    try:
        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            if logger:
                logger.error(f"Could not qualify {contract.symbol}")
            return None
        return qualified[0]
    except Exception as e:
        if logger:
            logger.error(f"Qualification failed for {contract.symbol}: {e}")
        return None


async def resolve_contfut(ib, contract, logger=None):
    """
    Resolve a continuous futures contract to its specific contract.

    After qualifying a CONTFUT, this gets the specific contract
    by conId to ensure proper data requests.

    Args:
        ib: Connected IB instance
        contract: Qualified CONTFUT contract
        logger: Optional logger

    Returns:
        Specific contract or None if resolution failed
    """
    try:
        specific = Contract(conId=contract.conId)
        qualified = await ib.qualifyContractsAsync(specific)
        if not qualified:
            if logger:
                logger.warning(f"Could not resolve CONTFUT {contract.symbol}")
            return contract  # Fall back to original
        return qualified[0]
    except Exception as e:
        if logger:
            logger.warning(f"CONTFUT resolution failed for {contract.symbol}: {e}")
        return contract


async def build_and_qualify(ib, t_conf: dict, logger=None):
    """
    Build and qualify a contract from config in one step.

    For CONTFUT types, also resolves to the specific contract.

    Args:
        ib: Connected IB instance
        t_conf: Target configuration dictionary
        logger: Optional logger

    Returns:
        Tuple of (qualified_contract, t_conf) or (None, t_conf) if failed
    """
    try:
        contract = build_contract_from_config(t_conf)
    except ValueError as e:
        if logger:
            logger.error(f"Failed to build contract for {t_conf.get('name')}: {e}")
        return None, t_conf

    qualified = await qualify_contract(ib, contract, logger)
    if not qualified:
        return None, t_conf

    # Resolve CONTFUT to specific contract
    if t_conf.get("secType") == "CONTFUT":
        qualified = await resolve_contfut(ib, qualified, logger)

    return qualified, t_conf
