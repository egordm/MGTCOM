class Error(Exception):
    """Base class for other exceptions"""
    pass


class TimeoutError(Error):
    """Timeout error"""
    pass


class NoCommunitiesFoundError(Error):
    """No communities found"""
    pass
