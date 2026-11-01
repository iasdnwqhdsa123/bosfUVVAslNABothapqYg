from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import grib

@dataclass
class Source:
    id: str
    label: str
    runs: tuple[int, ...]
    max_fhour: int
    levels: tuple[int, ...]
    level_vars: dict[str, str]
    surface: dict[str, tuple[str, str]]
    idx_kind: str = 'noaa'
    omega: bool = True
    moisture: str = 'td'
    grid: str = 'latlon'
    fstep: callable = None
    windowed: dict[str, str] = field(default_factory=dict)
    zs_geopotential: bool = False
    zs_step0_only: bool = False
    hub_m: float = 100.0
    gust_window: bool = False
    cloud_levels: tuple[int, ...] = ()
    snd_levels: tuple[int, ...] = ()
    snd_step: int = 1
    cover_tops: tuple = ('p', 642.0, 350.0)
    rh_over_water: bool = False
    cover_calibrated: bool = False
    tcc_column_max: bool = False
    hpbl_season: bool = False
    alias: dict[str, str] = field(default_factory=dict)
    workers = 8
    notes: list[str] = field(default_factory=list)
    short_runs: dict[int, int] = field(default_factory=dict)
    derive: callable = None
    regular: bool = False
    om_extra: tuple = None
    between_to: int = 0
    om_between: dict = field(default_factory=dict)
    om_hourly: dict = field(default_factory=dict)

    def builds(self, fh: int) -> bool:
        return bool(self.fstep(fh)) or 0 < fh <= self.between_to

    def last_fh(self, run: datetime) -> int:
        return self.short_runs.get(run.hour, self.max_fhour)

    def url(self, run: datetime, fh: int) -> str:
        raise NotImplementedError

    def idx_url(self, run: datetime, fh: int) -> str:
        return self.url(run, fh) + '.idx'

    def fetch(self, run, fh, sel):
        return grib.fetch_messages(self.url(run, fh), sel, workers=self.workers)

    def wanted(self, sounding=True):
        w = [(g, lev) for g, lev in self.surface.values()]
        for lv in self.levels:
            for g in self.level_vars.values():
                w.append((g, f'{lv} mb'))
        for lv in self.cloud_levels:
            for c in self.cloud_vars():
                w.append((self.level_vars[c], f'{lv} mb'))
        for lv in self.snd_levels if sounding else ():
            if lv in self.levels:
                continue
            for c in self.snd_vars():
                if (self.level_vars[c], f'{lv} mb') not in w:
                    w.append((self.level_vars[c], f'{lv} mb'))
        return w

    def snd_vars(self):
        return [c for c in ('t', 'td', 'rh', 'u', 'v', 'z', 'w') if c in self.level_vars]

    def cloud_vars(self):
        return [c for c in ('t', 'td', 'rh', 'clw', 'cc') if c in self.level_vars]
    inventory_tries = 6

    def inventory(self, run, fh, wanted=None):
        text = grib.http_get(self.idx_url(run, fh), tries=self.inventory_tries).decode()
        rows = grib.parse_noaa_idx(text) if self.idx_kind == 'noaa' else grib.parse_ecmwf_idx(text)
        for r in rows:
            r['var'] = self.alias.get(r['var'], r['var'])
        return grib.select(rows, wanted or self.wanted())

class HRRR(Source):

    def url(self, run, fh):
        return f'https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrfprsf{fh:02d}.grib2'

class GFS(Source):

    def url(self, run, fh):
        return f'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{fh:03d}'

class IFS(Source):
    MIRRORS = ('https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com', 'https://data.ecmwf.int/forecasts')
    mirror = 0
    workers = 4
    inventory_tries = 2
    PARTS = ('sfc', 'pl')

    def url(self, run, fh, part='sfc', mirror=None):
        m = self.MIRRORS[self.mirror if mirror is None else mirror]
        return f'{m}/{run:%Y%m%d}/{run:%H}z/ifs/0p1/oper/{run:%Y%m%d%H}0000-{fh}h-oper-fc-{part}.grib2'

    def idx_url(self, run, fh, part='pl'):
        return self.url(run, fh, part)[:-len('.grib2')] + '.index'

    def _each_mirror(self, call):
        last = None
        start = IFS.mirror
        for k in range(len(self.MIRRORS)):
            m = (start + k) % len(self.MIRRORS)
            self.mirror = m
            try:
                out = call()
                IFS.mirror = m
                return out
            except Exception as e:
                last = e
        raise last

    def inventory(self, run, fh, wanted=None):

        def one():
            rows = []
            for part in self.PARTS:
                text = grib.http_get(self.idx_url(run, fh, part), tries=self.inventory_tries).decode()
                for r in grib.parse_ecmwf_idx(text):
                    r['var'], r['part'] = (self.alias.get(r['var'], r['var']), part)
                    rows.append(r)
            return grib.select(rows, wanted or self.wanted())
        return self._each_mirror(one)

    def fetch(self, run, fh, sel):

        def one():
            out = {}
            for part in self.PARTS:
                mine = {k: r for k, r in sel.items() if r.get('part', 'sfc') == part}
                if mine:
                    urls = [self.url(run, fh, part, (self.mirror + k) % len(self.MIRRORS)) for k in range(len(self.MIRRORS))]
                    out.update(grib.fetch_messages(urls, mine, workers=self.workers, gap=1 << 20, tries=4, timeout=30))
            return out
        return self._each_mirror(one)

@dataclass
class OpenMeteo(Source):
    BASE = 'https://openmeteo.s3.amazonaws.com'
    domain: str = 'dwd_icon'
    grid_def: tuple = (-90.0, -180.0, 0.125, 1441, 2879)

    def url(self, run, fh, suffix=''):
        v = run + timedelta(hours=fh)
        return f'{self.BASE}/data_spatial/{self.domain}/{run:%Y/%m/%d/%H%M}Z/{v:%Y-%m-%dT%H%M}{suffix}.om'

    def idx_url(self, run, fh):
        return self.url(run, fh, '_model-level')

    def static_url(self):
        return f'{self.BASE}/data/{self.domain}/static/HSURF.om'

    def inventory(self, run, fh, wanted=None):
        return wanted or self.wanted()

    def fetch(self, run, fh, sel):
        import om
        return om.fetch_hour([self.url(run, fh), self.url(run, fh, '_model-level')], self.static_url(), om.Grid(*self.grid_def), sel, workers=self.workers)

@dataclass
class ENS(Source):
    MIRRORS = ('https://data.ecmwf.int/forecasts', 'https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com')

    def url(self, run, fh, mirror=0, stream='enfo'):
        kind = 'ef' if stream == 'enfo' else 'fc'
        return f'{self.MIRRORS[mirror]}/{run:%Y%m%d}/{run:%H}z/ifs/0p25/{stream}/{run:%Y%m%d%H}0000-{fh}h-{stream}-{kind}.grib2'

    def idx_url(self, run, fh, mirror=0, stream='enfo'):
        return self.url(run, fh, mirror, stream)[:-len('.grib2')] + '.index'

def ifs_derive(s):
    import numpy as np
    sd, rsn, sit = (s.pop('sdwe', None), s.pop('rsn', None), s.pop('sithick', None))
    if sd is not None:
        dens = 250.0 if rsn is None else np.where(np.nan_to_num(rsn, nan=0) > 50, rsn, 250.0)
        s['snod'] = np.maximum(np.nan_to_num(sd, nan=0), 0) * 1000.0 / dens
    if sit is not None:
        s['ice'] = (np.nan_to_num(sit, nan=0) > 0.05).astype(np.float32)
    return s
GA_NCEP = {'ceil': ('HGT', 'cloud ceiling'), 'vis': ('VIS', 'surface'), 'frz': ('HGT', '0C isotherm'), 'refc': ('REFC', 'entire atmosphere'), 'crain': ('CRAIN', 'surface'), 'csnow': ('CSNOW', 'surface'), 'cfrzr': ('CFRZR', 'surface'), 'cicep': ('CICEP', 'surface')}
BAND_RAIN = {'refd1': ('REFD', '1000 m above ground')}
GFS_HOURLY_TO = 120
GFS_3H_TO = 240
PAUSED = ()
REQUIRED = ('hrrr', 'gfs')
OPEN_AHEAD_H = {'hrrr': 24, 'gfs': 48, 'icon': 48, 'icond2': 24, 'ecmwf': 24}
OPEN_SLACK_H = 7
GFS_RUNS = (0, 6, 12, 18)
ICON_HOURLY_TO = 78
ICON_RUNS = (0, 6, 12, 18)
ICON_SHORT = {6: 120, 18: 120}
HRRR_RUNS = tuple(range(24))
HRRR_SHORT = {h: 18 for h in HRRR_RUNS if h % 6}
LEVELS_FINE = (1000, 975, 950, 925, 900, 875, 850, 800, 750, 700, 650, 600, 550, 500, 400, 300)
ICOND2_RUNS = tuple(range(0, 24, 3))
ICOND2_GRID = (43.18, -3.94, 0.02, 746, 1215)
SOURCES = {'hrrr': HRRR(id='hrrr', label='HRRR', runs=HRRR_RUNS, max_fhour=48, short_runs=HRRR_SHORT, grid='lambert', levels=LEVELS_FINE, level_vars={'t': 'TMP', 'td': 'DPT', 'u': 'UGRD', 'v': 'VGRD', 'z': 'HGT', 'w': 'VVEL', 'clw': 'CLMR'}, surface={'zs': ('HGT', 'surface'), 'ps': ('PRES', 'surface'), 't2': ('TMP', '2 m above ground'), 'td2': ('DPT', '2 m above ground'), 'u10': ('UGRD', '10 m above ground'), 'v10': ('VGRD', '10 m above ground'), 'uh': ('UGRD', '80 m above ground'), 'vh': ('VGRD', '80 m above ground'), 'ustar': ('FRICV', 'surface'), 'gust': ('GUST', 'surface'), 'hpbl': ('HPBL', 'surface'), 'shf': ('SHTFL', 'surface'), 'sw': ('DSWRF', 'surface'), 'cape': ('CAPE', 'surface'), 'tcc': ('TCDC', 'entire atmosphere'), 'lcc': ('LCDC', 'low cloud layer'), 'mcc': ('MCDC', 'middle cloud layer'), 'hcc': ('HCDC', 'high cloud layer'), 'prate': ('APCP', 'surface ([1-9]\\d*-\\d+|0-1) hour acc fcst'), 'prate_now': ('PRATE', 'surface'), 'refmax': ('MAXREF', '1000 m above ground'), 'mslp': ('MSLMA', 'mean sea level'), 'smoke': ('MASSDEN', '8 m above ground'), 'tsfc': ('TMP', 'surface'), 'lsm': ('LAND', 'surface'), 'ice': ('ICEC', 'surface'), 'snod': ('SNOD', 'surface'), 'mstav': ('MSTAV', '0 m underground'), **GA_NCEP, 'ctop': ('HGT', 'cloud top'), **BAND_RAIN}, windowed={'prate': 'acc1h'}, fstep=lambda fh: True, hub_m=80.0, cloud_levels=(250, 200), tcc_column_max=True, hpbl_season=True, snd_levels=(1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150), snd_step=3), 'gfs': GFS(id='gfs', label='GFS', runs=GFS_RUNS, max_fhour=384, levels=(1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 400, 300), level_vars={'t': 'TMP', 'rh': 'RH', 'u': 'UGRD', 'v': 'VGRD', 'z': 'HGT', 'w': 'VVEL', 'clw': 'CLMR', 'cc': 'TCDC'}, moisture='rh', surface={'zs': ('HGT', 'surface'), 'ps': ('PRES', 'surface'), 't2': ('TMP', '2 m above ground'), 'td2': ('DPT', '2 m above ground'), 'u10': ('UGRD', '10 m above ground'), 'v10': ('VGRD', '10 m above ground'), 'uh': ('UGRD', '100 m above ground'), 'vh': ('VGRD', '100 m above ground'), 'ustar': ('FRICV', 'surface'), 'gust': ('GUST', 'surface'), 'hpbl': ('HPBL', 'surface'), 'shf': ('SHTFL', 'surface'), 'sw': ('DSWRF', 'surface'), 'cape': ('CAPE', 'surface'), 'tcc': ('TCDC', 'entire atmosphere'), 'lcc': ('LCDC', 'low cloud layer'), 'mcc': ('MCDC', 'middle cloud layer'), 'hcc': ('HCDC', 'high cloud layer'), 'prate': ('PRATE', 'surface \\d+-\\d+ hour ave fcst'), 'prate_now': ('PRATE', 'surface'), 'cprate': ('CPRAT', 'surface \\d+-\\d+ hour ave fcst'), 'cprate_now': ('CPRAT', 'surface'), 'mslp': ('PRMSL', 'mean sea level'), 'lsm': ('LAND', 'surface'), 'tsfc': ('TMP', 'surface'), 'ice': ('ICEC', 'surface'), 'snod': ('SNOD', 'surface'), 'soilw': ('SOILW', '0-0.1 m below ground'), 'wilt': ('WILT', 'surface'), 'fldcp': ('FLDCP', 'surface'), **GA_NCEP, **BAND_RAIN}, windowed={'shf': 'avg6', 'sw': 'avg6', 'prate': 'avg6', 'cprate': 'avg6'}, fstep=lambda fh: fh <= GFS_HOURLY_TO or (fh % 3 == 0 if fh <= GFS_3H_TO else fh % 6 == 0), cloud_levels=(250, 200), tcc_column_max=True, snd_levels=(1000, 975, 950, 925, 900, 850, 800, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150)), 'ecmwf': IFS(id='ecmwf', label='ECMWF', runs=(0, 6, 12, 18), max_fhour=360, short_runs={6: 144, 18: 144}, idx_kind='ecmwf', omega=True, moisture='rh', cover_tops=('sigma', 0.8, 0.45), levels=(1000, 925, 850, 700, 600, 500, 400, 300), level_vars={'t': 't', 'rh': 'r', 'u': 'u', 'v': 'v', 'z': 'gh', 'w': 'w'}, surface={'t2': ('2t', 'sfc'), 'td2': ('2d', 'sfc'), 'u10': ('10u', 'sfc'), 'v10': ('10v', 'sfc'), 'uh': ('100u', 'sfc'), 'vh': ('100v', 'sfc'), 'ps': ('sp', 'sfc'), 'mslp': ('msl', 'sfc'), 'cape': ('mucape', 'sfc'), 'tcc': ('tcc', 'sfc'), 'lcc': ('lcc', 'sfc'), 'mcc': ('mcc', 'sfc'), 'hcc': ('hcc', 'sfc'), 'prate': ('tp', 'sfc'), 'prate_now': ('tprate', 'sfc'), 'cprate': ('cp', 'sfc'), 'zs': ('z', 'sfc'), 'gust': ('10fg', 'sfc'), 'sw': ('ssrd', 'sfc'), 'lsm': ('lsm', 'sfc'), 'ptype': ('ptype', 'sfc'), 'rnsw': ('ssr', 'sfc'), 'rnlw': ('str', 'sfc'), 'olr': ('ttr', 'sfc'), 'tclw': ('tclw', 'sfc'), 'snowrate': ('sf', 'sfc'), 'tsfc': ('skt', 'sfc'), 'sdwe': ('sd', 'sfc'), 'rsn': ('rsn', 'sfc'), 'sithick': ('sithick', 'sfc'), 'soilm9': ('vsw', 'sol 1')}, windowed={'sw': 'acc', 'rnsw': 'acc', 'rnlw': 'acc', 'olr': 'acc', 'prate': 'accw', 'cprate': 'accw', 'snowrate': 'accw'}, gust_window=True, alias={'10fg3': '10fg', '10fg6': '10fg'}, cloud_levels=(250, 200), zs_geopotential=True, zs_step0_only=True, snd_levels=(1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150), fstep=lambda fh: fh % 3 == 0 if fh <= 144 else fh % 6 == 0, derive=lambda s: ifs_derive(s), regular=True, om_extra=('ecmwf_ifs', {'blh': 'boundary_layer_height', 'vis': 'visibility', 'cin': 'convective_inhibition'}), between_to=90, om_between={'t2': ('temperature_2m', 'K'), 'td2': ('dew_point_2m', 'K'), 'tsfc': ('surface_temperature', 'K'), 'u10': ('wind_u_component_10m', None), 'v10': ('wind_v_component_10m', None), 'uh': ('wind_u_component_100m', None), 'vh': ('wind_v_component_100m', None), 'gust': ('wind_gusts_10m', None), 'mslp': ('pressure_msl', 'Pa'), 'cape': ('cape', None), 'tcc': ('cloud_cover', None), 'lcc': ('cloud_cover_low', None), 'mcc': ('cloud_cover_mid', None), 'hcc': ('cloud_cover_high', None), 'sw': ('shortwave_radiation', 'sun'), 'soilm9': ('soil_moisture_0_to_7cm', None)}, om_hourly={'prate': ('precipitation', None), 'cprate': ('showers', None), 'snowrate': ('snowfall_water_equivalent', None)}, notes=['ECMWF IFS 9 km open data (CC BY 4.0)', "hourly to +90 h: the hours between its 3-hourly steps blend the steps' upper air, their surface from Open-Meteo's hourly copy of the run", "no boundary-layer height or heat flux in open data: parcel method, and the heat flux from the model's net radiation, humidity and soil water", "BLH, CIN and visibility from Open-Meteo's spatial files of the same run (O1280 grid)", "no ceiling, freezing-level or cloud-water-by-level fields: ceiling from the cloud cover diagnosed on each level, cloud tops also from the outgoing longwave, icing from the column's cloud liquid"]), 'icon': OpenMeteo(id='icon', label='ICON', runs=ICON_RUNS, max_fhour=180, short_runs=ICON_SHORT, grid='latlon', moisture='rh', omega=False, cover_tops=('icon', 0.7895, 0.3948), levels=(1000, 950, 925, 900, 850, 800, 700, 600, 500, 400, 300), level_vars={'t': 'temperature', 'rh': 'relative_humidity', 'u': 'wind_u_component', 'v': 'wind_v_component', 'z': 'geopotential_height'}, surface={'zs': ('zs', ''), 'ps': ('ps', ''), 't2': ('temperature_2m', ''), 'td2': ('td2', ''), 'u10': ('wind_u_component_10m', ''), 'v10': ('wind_v_component_10m', ''), 'uh': ('wind_u_component_80m', ''), 'vh': ('wind_v_component_80m', ''), 'gust': ('wind_gusts_10m', ''), 'shf': ('shf', ''), 'sw': ('sw', ''), 'cape': ('cape', ''), 'tcc': ('cloud_cover', ''), 'lcc': ('cloud_cover_low', ''), 'mcc': ('cloud_cover_mid', ''), 'hcc': ('cloud_cover_high', ''), 'prate': ('precipitation', ''), 'mslp': ('pressure_msl', ''), 'cprate': ('showers', ''), 'tsfc': ('soil_temperature_0cm', ''), 'lsm': ('lsm', ''), 'ice': ('ice', ''), 'snod': ('snow_depth', ''), 'frz': ('freezing_level_height', ''), 'ptype': ('ptype', ''), 'soilm9': ('soil_moisture_3_to_9cm', '')}, windowed={'sw': 'mean', 'shf': 'mean', 'prate': 'sum', 'cprate': 'sum'}, gust_window=True, hub_m=80.0, cloud_levels=(250, 200), rh_over_water=True, cover_calibrated=True, fstep=lambda fh: 1 <= fh <= ICON_HOURLY_TO or (fh % 3 == 0 and ICON_HOURLY_TO < fh <= 180), notes=["DWD's ICON regridded to 0.125° by Open-Meteo (CC BY 4.0)", 'no boundary-layer height, vertical wind, cloud water, ceiling or visibility: parcel method, ceiling from the cloud cover diagnosed on each level, the rest from the humidity profile']), 'icond2': OpenMeteo(id='icond2', label='ICON-D2', runs=ICOND2_RUNS, max_fhour=48, grid='latlon', moisture='rh', omega=False, domain='dwd_icon_d2', grid_def=ICOND2_GRID, snd_levels=(1000, 975, 950, 850, 700, 600, 500, 400, 300, 250, 200), snd_step=4, cover_tops=('icon', 0.7895, 0.3948), levels=(1000, 975, 950, 850, 700, 600, 500, 400, 300), level_vars={'t': 'temperature', 'rh': 'relative_humidity', 'u': 'wind_u_component', 'v': 'wind_v_component', 'z': 'geopotential_height'}, surface={'zs': ('zs', ''), 'ps': ('ps', ''), 't2': ('temperature_2m', ''), 'td2': ('td2', ''), 'u10': ('wind_u_component_10m', ''), 'v10': ('wind_v_component_10m', ''), 'uh': ('wind_u_component_80m', ''), 'vh': ('wind_v_component_80m', ''), 'gust': ('wind_gusts_10m', ''), 'shf': ('shf', ''), 'sw': ('sw', ''), 'cape': ('cape', ''), 'tcc': ('cloud_cover', ''), 'lcc': ('cloud_cover_low', ''), 'mcc': ('cloud_cover_mid', ''), 'hcc': ('cloud_cover_high', ''), 'prate': ('precipitation', ''), 'mslp': ('pressure_msl', ''), 'tsfc': ('soil_temperature_0cm', ''), 'lsm': ('lsm', ''), 'ice': ('ice', ''), 'snod': ('snow_depth', ''), 'frz': ('freezing_level_height', ''), 'ptype': ('ptype', ''), 'soilm9': ('soil_moisture_3_to_9cm', ''), 'vis': ('visibility', '')}, windowed={'sw': 'mean', 'shf': 'mean', 'prate': 'sum'}, gust_window=True, hub_m=80.0, cloud_levels=(250, 200), rh_over_water=True, cover_calibrated=True, fstep=lambda fh: 1 <= fh <= 48, notes=["DWD's ICON-D2 (2.2 km) on a 0.02° grid by Open-Meteo (CC BY 4.0)", 'no boundary-layer height, vertical wind, cloud water, ceiling or radar: parcel method, ceiling from the cloud cover diagnosed on each level']), 'ens': ENS(id='ens', label='ECMWF ensemble', runs=(0, 12), max_fhour=240, levels=(), level_vars={}, surface={}, idx_kind='ecmwf', fstep=lambda fh: 0 <= fh <= 240 and fh % 3 == 0 and (fh <= 144 or fh % 6 == 0), notes=["ECMWF's ENS, 50 members (0.25°, open data, CC BY 4.0): each member's thermals from its net radiation and a dry parcel, no terrain flows; percentiles and chances only"])}

def candidate_runs(src: Source, now: datetime | None=None, back_hours: int=30):
    now = now or datetime.now(timezone.utc)
    t = now.replace(minute=0, second=0, microsecond=0)
    for h in range(back_hours):
        r = t - timedelta(hours=h)
        if r.hour in src.runs:
            yield r
DAYS = {'hrrr': 2, 'gfs': 16, 'ecmwf': 15, 'icon': 8, 'icond2': 2, 'ens': 10}
DAY_UTC = range(10, 34)

def needed_times(src: Source, days: int | None=None, now: datetime | None=None):
    now = now or datetime.now(timezone.utc)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return [today + timedelta(days=d, hours=h) for d in range(-1, days or DAYS[src.id]) for h in DAY_UTC]

def latest_complete_run(src: Source, needed, log=print, now: datetime | None=None):
    for run in candidate_runs(src, now):
        fhs = [int((v - run).total_seconds() // 3600) for v in needed]
        fhs = [f for f in fhs if 0 <= f <= src.last_fh(run) and src.builds(f)]
        steps = [f for f in fhs if src.fstep(f)]
        if not steps:
            continue
        if grib.exists(src.idx_url(run, max(steps))):
            return (run, fhs)
        log(f'{src.id} {run:%Y-%m-%d %HZ} not complete yet')
    return (None, [])
