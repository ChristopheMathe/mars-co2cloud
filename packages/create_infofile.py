class FileName(object):
    def __init__(self):
        pass


class IdxDim(object):
    def __init__(self, time, altitude, latitude, longitude):
        self.time = time
        self.altitude = altitude
        self.latitude = latitude
        self.longitude = longitude


class DataDim(object):
    def __init__(self, time, altitude, latitude, longitude):
        self.time = time
        self.altitude = altitude
        self.latitude = latitude
        self.longitude = longitude


class DataTarget(object):
    def __init__(self):
        pass


class TargetName(object):
    def __init__(self):
        pass


class LocalTime(object):
    def __init__(self):
        pass


class InfoFile(FileName, IdxDim, DataDim, TargetName, DataTarget, LocalTime):
    def __init__(self):
        super().__init__()
        self.filename = FileName()
        self.idx_dim = IdxDim(time=None, altitude=None, latitude=None, longitude=None)
        self.data_dim = DataDim(time=None, altitude=None, latitude=None, longitude=None)
        self.target_name = TargetName()
        self.data_target = DataTarget()
        self.local_time = None
    pass
