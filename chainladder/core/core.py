import polars as pl
from ._slicing import normalize_index

dcol = (
    (pl.col('__development__').dt.year() - 
     pl.col('__origin__').dt.year()) * 12 + 
    pl.col('__development__').dt.month() - 
    pl.col('__origin__').dt.month() + 1
).cast(pl.UInt16).alias('__development__')

vcol = pl.date(
    year=(pl.col('__origin__').dt.year() + 
          (pl.col('__origin__').dt.month() + 
           pl.col('__development__') - 1) // 12 - 
          pl.when((pl.col('__origin__').dt.month() + 
                   pl.col('__development__') - 1
                  ) % 12 == 0)
            .then(1).otherwise(0)),
    month=(pl.col('__origin__').dt.month() +
           pl.col('__development__') - 2) % 12 + 1, day=1
).dt.month_end().alias('__development__')


class TriangleBase:
    """ Triangle written exclusively in polars """
    def __init__(
        self, data=None, index=None, origin=None, columns=None, 
        valuation=None, origin_format=None, valuation_format=None,
        cumulative=None, pattern=False, trailing=False, lazy=False,
        *args, **kwargs
    ):
        if data is None:
            return
        # Static attributes
        self.columns = [columns] if type(columns) is str else columns
        self.is_cumulative = cumulative
        self.is_pattern = pattern
        self.origin_close = 'DEC'
        
        index = index or []
        if valuation is None:
            data = (
                data.with_columns(
                    TriangleBase._format_origin(
                        data, origin, origin_format
                    ).dt.truncate("1mo").alias('__origin__'),
                    TriangleBase._format_valuation(
                        data, origin, origin_format
                    ).dt.month_end().max().alias('__development__'))
                .select(
                    pl.col(index + self.columns + 
                        ['__origin__', '__development__'])))
        else:
            data = (
                data.with_columns(
                    TriangleBase._format_origin(
                        data, origin, origin_format
                    ).dt.truncate("1mo").alias('__origin__'),
                    TriangleBase._format_valuation(
                        data, valuation, valuation_format
                    ).dt.month_end().alias('__development__'))
                .select(
                    pl.col(index + self.columns + 
                        ['__origin__', '__development__'])))
            if data.select('__development__').lazy().collect().n_unique() > 1:
                data = data.select(pl.all().exclude('__development__'), dcol)
        self.data = (
            data
            .with_columns(pl.lit('Total').alias('Total') if not index else [])
            .group_by(pl.all().exclude(columns)) # Needed for cum_to_incr/incr_to_cum
            .agg(pl.col(columns).sum())
            .sort(index + ['__origin__', '__development__'])
            )
        self.is_lazy = lazy
        if not lazy:
            self.data = self.data.lazy().collect()

        if trailing:
            self.data = self.grain(
                f'O{self.origin_grain}D{self.development_grain}', 
                trailing=True).data
        self.properties = {}
    
    @staticmethod
    def from_triangle(triangle):
        obj = TriangleBase()
        obj.data = triangle.data
        obj.columns = triangle.columns
        obj.is_cumulative = triangle.is_cumulative
        obj.is_pattern = triangle.is_pattern
        obj.origin_close = triangle.origin_close
        obj.is_lazy = triangle.is_lazy
        obj.properties = triangle.properties.copy()
        return obj
        
    @property
    def key_labels(self):
        if 'key_labels' not in self.properties.keys():
            self.properties['key_labels'] = [
                c for c in self.data.columns 
                if c not in self.columns + 
                ['__origin__', '__development__']]
        return self.properties['key_labels']

    @property
    def shape(self):
        # requires index, columns, origin, valuation, development
        return (
            self.index.lazy().select(pl.count()).collect()[0, 0], 
            len(self.columns), 
            len(self.origin), 
            len(self.valuation) if self.is_val_tri else len(self.development))
    
    @property
    def valuation_date(self):
        # requires valuation
        return self.valuation.max()
    
    @property
    def index(self):
        if 'index' not in self.properties.keys():
            self.properties['index'] = (
                self.data.select(pl.col(self.key_labels)).unique().sort(pl.all())
            )
        return self.properties['index']    
    
    @property
    def origin_grain(self):
        if 'origin_grain' not in self.properties.keys():
            months = self.date_matrix.select(
                pl.col('__origin__').dt.month().sort().unique()
            )['__origin__']
            diffs = months.diff()[1:]
            if len(months) == 1:
                grain = "Y"
            elif (diffs == 6).all():
                grain = "2Q"
            elif (diffs == 3).all():
                grain = "Q"
            else:
                grain = "M"
            self.properties['origin_grain'] = grain
        return self.properties['origin_grain']
    
    @property
    def date_matrix(self):
        if 'date_matrix' not in self.properties.keys():
            if self.is_val_tri:
                self.properties['date_matrix'] = (
                    self.data
                    .group_by('__origin__')
                    .agg(pl.col('__development__').unique())
                    .explode('__development__').select(
                        pl.col('__origin__'),
                        pl.col('__development__').alias('__valuation__'),
                        dcol
                    )).lazy().collect()
            else:
                self.properties['date_matrix'] = (
                    self.data
                    .group_by('__origin__')
                    .agg(pl.col('__development__').unique())
                    .explode('__development__')
                    .with_columns(vcol.alias('__valuation__'))).lazy().collect()
        return self.properties['date_matrix']

        
    @property
    def origin(self):
        if 'origin' not in self.properties.keys():
            self.properties['origin'] = pl.date_range(
                start=self.date_matrix['__origin__'].min(),
                end=self.date_matrix['__origin__'].max(),
                interval={'Y': '12mo', 'M': '1mo', 
                        'Q': '3mo', '2Q': '6mo'}[self.origin_grain], 
                eager=True).alias('origin')
        return self.properties['origin']

    @property
    def odims(self):
        return pl.DataFrame({'odims': range(len(self.origin)), '__origin__': self.origin})
    
    @property
    def ddims(self):
        values = self.valutaion if self.is_val_tri else self.development
        return pl.DataFrame({'ddims': range(len(values)), '__development__': values})
    
    @property
    def development(self):
        if 'development' not in self.properties.keys():
            interval = {'Y': 12, '2Q': 6, 'Q': 3, 'M': 1}[self.development_grain]
            self.properties['development'] = pl.Series(
                'development', 
                range(self.date_matrix['__development__'].min(), 
                        self.date_matrix['__development__'].max() + interval, 
                        interval)).cast(pl.UInt16)
        return self.properties['development']

    @property
    def valuation(self):
        if 'valuation' not in self.properties.keys():
            interval={'Y': '12mo', 'M': '1mo', 
                    'Q': '3mo', '2Q': '6mo'}[self.development_grain]
            valuation_range = self.date_matrix.select(
                pl.col('__valuation__').min().alias('vmin').dt.month_start(),
                pl.col('__valuation__').max().alias('vmax'))
            self.properties['valuation'] = pl.date_range(
                start=valuation_range['vmin'][0],
                end=valuation_range['vmax'][0],
                interval=interval, 
                eager=True).dt.month_end().alias('valuation')
        return self.properties['valuation']

    @property
    def is_full(self):
        return (self.data.select(['__origin__', '__valuation__']).n_unique() == 
                self.shape[2] * self.shape[3])
    
    @property
    def is_val_tri(self):
        if 'is_val_tri' not in self.properties.keys(): 
            self.properties['is_val_tri'] = dict(
                zip(self.data.columns, self.data.dtypes)
            )['__development__'] != pl.UInt16
        return self.properties['is_val_tri'] 
        
    @property
    def development_grain(self):
        if 'development_grain' not in self.properties.keys():
            if len(self.date_matrix['__valuation__'].unique()) == 1:
                grain = 'M'
            else:
                months = self.data.select(
                    self.date_matrix['__valuation__']
                        .dt.month().unique().sort().alias('__development__')
                ).lazy().collect()['__development__']
                diffs = months.diff()[1:]
                if len(months) == 1:
                    grain = "Y"
                elif (diffs == 6).all():
                    grain = "2Q"
                elif (diffs == 3).all():
                    grain = "Q"
                else:
                    grain = "M"
            self.properties['development_grain'] = grain
        return self.properties['development_grain']

    @property
    def latest_diagonal(self):
        # requires valuation, valuation_date
        triangle = self[self.valuation==self.valuation_date]
        if not triangle.is_val_tri:
            triangle.data = triangle.data.select(
                pl.all().exclude('__development__'), vcol)
            triangle.properties['is_val_tri'] = True
            triangle.properties.pop('date_matrix', None)
            triangle.properties.pop('valuation', None)
            triangle.properties.pop('development_grain', None)
            triangle.properties.pop('development', None)
        return triangle
    
    def val_to_dev(self):
        if self.is_val_tri:
            obj = TriangleBase.from_triangle(self)
            obj.data = obj.data.select(pl.all().exclude('__development__'), dcol)
            obj.properties['is_val_tri'] = False
            return obj
        else:
            return self
        
    def dev_to_val(self):
        if not self.is_val_tri:
            obj = TriangleBase.from_triangle(self)
            obj.data = obj.data.select(pl.all().exclude('__development__'), vcol)
            obj.properties['is_val_tri'] = True
            return obj
        else:
            return self
        
    def lazy(self, *args, **kwargs):
        self.data = self.data.lazy(*args, **kwargs)
        self.is_lazy = True
        return self
    
    def collect(self, *args, **kwargs):
        self.data = self.data.collect(*args, **kwargs)
        self.is_lazy = False
        return self
    
    @staticmethod
    def _format_origin(data, column, format):
        if data.select(column).dtypes[0] in ([pl.Date, pl.Datetime]):
            return pl.col(column).cast(pl.Date).dt.month_start()
        else:
            for f in ['%Y%m', '%Y', format]:
                c = (
                    pl.col(column)
                    .cast(pl.Utf8).str.to_date(format=f)
                    .cast(pl.Date).dt.month_start())
                try:
                    data.head(10).select(c)
                    return c
                except:
                    pass

    @staticmethod
    def _format_valuation(data, column, format) -> pl.Expr:
        if data.select(column).dtypes[0] in ([pl.Date, pl.Datetime]):
            return pl.col(column).cast(pl.Date).dt.month_end()
        else:
            for f in ['%Y%m', '%Y', format]:
                c = (
                    pl.col(column)
                    .cast(pl.Utf8).str.to_date(format=f)
                    .cast(pl.Date).dt.month_start() 
                    .alias('__development__'))
                try:
                    data.head(10).select(c)
                    break
                except:
                    pass
            if f == '%Y':
                return (c.dt.offset_by('12mo')
                        .dt.offset_by('-1d').dt.month_end())
            else:
                return c.dt.month_end()
                
    
    def _agg(self, agg, axis=None, *args, **kwargs):
        if axis is None:
            if max(self.shape) == 1:
                axis = 0
            else:
                axis = min([num for num, _ in enumerate(self.shape) if _ != 1])
        else:
            axis = self._get_axis(axis)
        obj = TriangleBase.from_triangle(self)
        if axis == 0:
            obj.data = (
                obj.data
                .group_by(['__origin__', '__development__'])
                .agg(getattr(pl.col(self.columns).fill_null(0), agg)(*args, **kwargs))
                .with_columns(*[pl.lit('(All)').alias(c) for c in self.key_labels])
            )
            obj.properties.pop('index', None)
            obj.properties.pop('key_labels', None)
        elif axis == 1:
            obj.data = self.data.select(
                pl.col(self.key_labels + ['__origin__', '__development__']), 
                pl.sum_horizontal(self.columns).alias('0'))
            obj.columns = ['0']
        elif axis == 2:
            obj.data = (
                self.data
                .group_by(self.key_labels + ['__development__'])
                .agg(getattr(pl.col(self.columns).fill_null(0), agg)(*args, **kwargs))
                .with_columns(pl.lit(self.origin.min()).alias('__origin__')))
            obj.properties.pop('date_matrix', None)
            obj.properties.pop('origin', None)
            obj.properties.pop('origin_grain', None)
        elif axis == 3:
            obj.data = (
                self.data
                .group_by(self.key_labels + ['__origin__'])
                .agg(getattr(pl.col(self.columns).fill_null(0), agg)(*args, **kwargs))
                .with_columns(pl.lit(self.valuation_date).alias('__development__')))
            obj.properties['is_val_tri'] = True
            obj.properties.pop('date_matrix', None)
            obj.properties.pop('development', None)
            obj.properties.pop('development_grain', None)
            obj.properties.pop('valuation', None)
        else:
            raise ValueError(f'axis {axis} is not supported')
        return obj
        
    def sum(self, axis=None):
        return self._agg('sum', axis)
    
    def mean(self, axis=None):
        return self._agg('mean', axis)
    
    def min(self, axis=None):
        return self._agg('min', axis)

    def max(self, axis=None):
        return self._agg('max', axis)
    
    def median(self, axis=None):
        return self._agg('median', axis)
    
    def std(self, axis=None):
        return self._agg('std', axis)
    
    def var(self, axis=None):
        return self._agg('var', axis)
    
    def product(self, axis=None):
        return self._agg('product', axis)
    
    def quantile(self, axis=None, q=0.5):
        return self._agg('quantile', axis, quantile=q)
    
    def _get_axis(self, axis):
        ax = {
            **{0: 0, 1: 1, 2: 2, 3: 3},
            **{-1: 3, -2: 2, -3: 1, -4: 0},
            **{"index": 0, "columns": 1, "origin": 2, "development": 3},
        }
        return ax.get(axis, 0)
    
    def group_by(self, by, axis=0, *args, **kwargs):
        """Group Triangle by index values.  If the triangle is convertable to a
        DataFrame, then it defaults to pandas groupby functionality.

        Parameters
        ----------
        by: str or list
            The index to group by

        Returns
        -------
            GroupBy object (pandas or Triangle)
        """
        return PlTriangleGroupBy(self, by, axis)

    def incr_to_cum(self, inplace=False):
        """Method to convert an incremental triangle into a cumulative triangle.

        Parameters
        ----------
        inplace: bool
            Set to True will update the instance data attribute inplace

        Returns
        -------
            Updated instance of triangle accumulated along the origin
        """

        if self.is_cumulative:
            return self
        triangle = TriangleBase.from_triangle(self)
        if self.is_val_tri:
            col = pl.col('__development__')
        else:
            col = vcol
        expanded = (
            self.data.lazy()
            .select(pl.col(self.key_labels + ['__origin__']), col)
            .group_by(self.key_labels + ['__origin__'])
            .agg(pl.col('__development__').min())
            .join(self.valuation.to_frame().lazy(), how='cross')
            .filter(pl.col('__development__')<=pl.col('valuation'))
            .drop('__development__').rename({'valuation':'__development__'}))
        triangle.data = (
            self.data.lazy()
            .select(pl.all().exclude('__development__'), col)
            .join(expanded, how='outer', 
                  left_on=self.key_labels + ['__origin__', '__development__'], 
                  right_on=self.key_labels + ['__origin__', '__development__'])
            .sort(by=self.key_labels + ['__origin__', '__development__'])
            .group_by(self.key_labels + ['__origin__'])
            .agg(
                pl.col('__development__'),
                pl.col(self.columns).fill_null(pl.lit(0)).cumsum())
            .explode(["__development__"] + self.columns))
        if not self.is_lazy:
            triangle.data = triangle.data.collect()
        triangle.is_cumulative = True
        triangle.properties['is_val_tri'] = True
        triangle.properties.pop('date_matrix', None)
        if self.is_val_tri:
            triangle.properties.pop('valuation', None)
            return triangle
        else:
            triangle.properties.pop('development', None)
            return triangle.val_to_dev()
       
    def cum_to_incr(self, filter_zeros=False):
        """Method to convert an cumlative triangle into a incremental triangle.

        Parameters
        ----------
            inplace: bool
                Set to True will update the instance data attribute inplace

        Returns
        -------
            Updated instance of triangle accumulated along the origin
        """
        if not self.is_cumulative:
            return self
        else:
            triangle = TriangleBase.from_triangle(self)
            triangle.data = (
                self.data.lazy()
                .sort(self.key_labels + ['__origin__', '__development__'])
                .group_by(self.key_labels + ['__origin__'])
                .agg(
                    pl.col('__development__'),
                    pl.col(self.columns).diff().fill_null(pl.col(self.columns)))
                .explode(["__development__"] + self.columns)
                .filter(pl.any_horizontal(pl.col(self.columns) != 0) if filter_zeros else True)
                )
            triangle.is_cumulative = False
            triangle.properties.pop('date_matrix', None)
            if self.is_val_tri:
                triangle.properties.pop('valuation', None)
            else:
                triangle.properties.pop('development', None)
            if not self.is_lazy:
                triangle.data = triangle.data.collect()
            return triangle
        
    @property
    def link_ratio(self):
        triangle = TriangleBase.from_triangle(self.incr_to_cum().val_to_dev())
        interval = {'Y': 12, '2Q': 6, 'Q': 3, 'M': 1}[self.development_grain]
        triangle.data = (
            triangle.data.lazy()
            .sort(['__origin__', '__development__'])
            .group_by(self.key_labels + ['__origin__'])
            .agg(
                (pl.col('__development__') - 
                 pl.lit(interval)).cast(pl.UInt16).alias('__development__'),
                (pl.when(pl.col(self.columns).pct_change().is_infinite())
                   .then(pl.lit(None))
                   .otherwise(pl.col(self.columns).pct_change()) + pl.lit(1.0)
                ).keep_name())
            .explode(["__development__"] + self.columns)
            .filter(~pl.any_horizontal(pl.col(self.columns).is_null())))
        if not self.is_lazy:
            triangle.data = triangle.data.collect()
        triangle.is_pattern = True
        triangle.is_cumulative = False
        triangle.properties.pop('date_matrix', None)
        if self.is_val_tri:
            triangle.properties.pop('valuation', None)
        else:
            triangle.properties.pop('development', None)
        return triangle

    
    def grain(self, grain="", trailing=False, inplace=False):
        """Changes the grain of a cumulative triangle.

        Parameters
        ----------
        grain : str
            The grain to which you want your triangle converted, specified as
            'OXDY' where X and Y can take on values of ``['Y', 'S', 'Q', 'M'
            ]`` For example, 'OYDY' for Origin Year/Development Year, 'OQDM'
            for Origin quarter/Development Month, etc.
        trailing : bool
            For partial origin years/quarters, trailing will set the year/quarter
            end to that of the latest available from the origin data.
        inplace : bool
            Whether to mutate the existing Triangle instance or return a new
            one.

        Returns
        -------
            Triangle
        """
        ograin_new = grain[1:2]
        dgrain_new = grain[-1]
        ograin_new = "S" if ograin_new == "H" else ograin_new
        latest_month = self.valuation_date.month
        grain_dict = {'Y': 12, 'S': 6, 'Q': 3, 'M': 1}
        if trailing:
            offset = grain_dict[ograin_new]
            offset = str(-((offset - latest_month % offset) % offset)) + 'mo'
            origin_close = pl.date(2006, 1, 1).dt.strftime('%b').str.to_uppercase()
            origin_close = (
                pl.DataFrame().select(
                    pl.date(2000, latest_month, 1).dt.strftime('%b').str.to_uppercase()
                )[0, 0])
        else:
            origin_close = self.origin_close
            offset = '0mo'
        if self.is_val_tri:
            triangle = TriangleBase.from_triangle(self)
        else:
            triangle = self.dev_to_val()
        origin_map = (
            self.origin
            .to_frame().lazy()
            .group_by_dynamic(
                index_column=pl.col('origin'), 
                every=str(grain_dict[ograin_new]) + 'mo',
                offset=offset)
            .agg(pl.col('origin').alias('__origin__'))
            .explode(pl.col('__origin__'))
            .select(
                pl.col('origin'),
                pl.col('__origin__')))
        data = (
            triangle.data.lazy()
            .join(origin_map, how='inner', 
                  left_on='__origin__', 
                  right_on='__origin__')
            .drop('__origin__')
            .rename({'origin': '__origin__'}))
        self.origin_close = origin_close
        development_map = (
            self.valuation.dt.month_start().sort()
            .to_frame().lazy()
            .group_by_dynamic(
                index_column=pl.col('valuation'), 
                every=str(grain_dict[dgrain_new]) + 'mo',
                offset=offset)
            .agg(pl.col('valuation').alias('__development__'))
            .explode(pl.col('__development__'))
            .select(
                pl.col('valuation').dt.offset_by(str(grain_dict[dgrain_new])+'mo').dt.offset_by('-1d'),
                pl.col('__development__').dt.month_end()))
        if self.is_cumulative:
            development_map = (
                development_map
                .group_by('valuation')
                .agg(pl.col('__development__').max()))
        data = (
            data
            .join(development_map, how='inner', 
                  left_on='__development__', 
                  right_on='__development__')
            .drop('__development__')
            .rename({'valuation': '__development__'})
            .group_by(self.key_labels + ['__origin__', '__development__']).sum())
        triangle.data = data
        if not self.is_lazy:
            triangle.data = triangle.data.collect()
        triangle.properties.pop('date_matrix', None)
        if self.origin_grain != ograin_new:
            triangle.properties.pop('origin', None)
            triangle.properties.pop('origin_grain', None)
        if self.development_grain != dgrain_new:
            triangle.properties.pop('development', None)
            triangle.properties.pop('valuation', None)
            triangle.properties.pop('development_grain', None)
        if self.is_val_tri:
            return triangle
        else:
            return triangle.val_to_dev()
        
    def wide(self):
        if self.shape[:2] == (1, 1):
            return (
                self.data
                .with_columns(
                    (pl.col('__development__').dt.strftime('%Y-%m') 
                     if self.is_val_tri else
                     pl.col('__development__')).alias('development'),
                    pl.col('__origin__').alias('origin'),
                    pl.col(self.columns))
                .sort('development').lazy().collect(streaming=True)
                .pivot(
                    index='origin',
                    columns='development',
                    values=self.columns,
                    aggregate_function='first')
                .sort('origin'))
        else:
            raise ValueError(f'Wide format expects shape of (1, 1), but got {self.shape[:2]}')
    
    def _get_idx(self, idx):
        def _normalize_index(key):
            key = normalize_index(key, self.shape)
            l = []
            for n, i in enumerate(key):
                if type(i) is slice:
                    start = i.start if i.start > 0 else None
                    stop = i.stop if i.stop > -1 else None
                    stop = None if stop == self.shape[n] else stop
                    step = None if start is None and stop is None else i.step
                    l.append(slice(start, stop, step))
                else:
                    l.append(i)
            key = tuple(l)
            return key

        def _contig_slice(arr):
            """ Try to make a contiguous slicer from an array of indices """
            if type(arr) is slice:
                return arr
            if type(arr) in [int]:
                arr = [arr]
            if len(arr) == 1:
                return slice(arr[0], arr[0] + 1)
            if len(arr) == 0:
                raise ValueError("Slice returns empty Triangle")
            diff = pl.Series(arr).diff()
            if max(diff) == min(diff):
                step = max(diff)
            else:
                return arr
            step = None if step == 1 else step
            min_arr = None if min(arr) == 0 else min(arr)
            max_arr = max(arr) + 1
            if step and step < 0:
                min_arr, max_arr = max_arr - 1, min_arr - 1 if min_arr else min_arr
            return slice(min_arr, max_arr, step)
            
        idx = _normalize_index(idx)
        return (_contig_slice(idx[0]), _contig_slice(idx[1]), 
                _contig_slice(idx[2]), _contig_slice(idx[3]))

    def __getitem__(self, key):
        """ Only returns polars expressions. """
        if type(key) is str:
            key = [key]
        if type(key) is tuple or type(key) is slice or type(key) is int:
            s0, s1, s2, s3 = self._get_idx(key)
            return (
                [pl.col(c).is_in(self.index[c]) for c in self.key_labels[s0]],
                self.columns[s1],
                [pl.col('__origin__').is_in(self.origin[s2])],
                [pl.col('__development__').is_in(self.valuation[s3] if self.is_val_tri else self.development[s3])])
        elif type(key) is list:
            return self.select(key)
        elif type(key) is pl.Series:
            triangle = TriangleBase.from_triangle(self)
            triangle.properties.pop('date_matrix', None)
            if key.name == 'valuation':
                key = self.valuation.filter(key)
                triangle.properties.pop('development', None)
                triangle.properties['valuation'] = key
                return triangle.filter(pl.col('__development__').is_in(key) if self.is_val_tri else vcol.is_in(key))
            elif key.name == 'development':
                triangle.properties.pop('valuation', None)
                key = self.development.filter(key)
                triangle.properties['development'] = key
                return triangle.filter(dcol.is_in(key) if self.is_val_tri else pl.col('__development__').is_in(key))
            elif key.name == 'origin':
                key = self.origin.filter(key)
                triangle.properties['origin'] = key
                return triangle.filter(pl.col('__origin__').is_in(key))
        else:   
            raise NotImplementedError()
        
    def __setitem__(self, key, value):
        """ Function for pandas style column setting """
        if type(value) is pl.Expr:
            self.data = self.data.select(pl.all().exclude(key), value.alias(key))
        elif type(value) != type(self):
            value = self._triangle_literal(value)
            value.data = value.data.rename({'__value__': key})
            value.columns = [key]
            self.data = (
                self.data.select(pl.all().exclude(key))
                .join(value.data.select([key, '__origin__', '__development__']), 
                      how='left', on=['__origin__', '__development__']))
        else:
            if len(set(self.key_labels) - set(value.key_labels)) > 0:
                raise ValueError(
                    f"""Unable to assign triangle with unknown 
                    key_labels {set(self.key_labels) - set(value.key_labels)}.""")
            if len(value.columns) > 1:
                raise ValueError(
                    f"""Unable to assign triangle with multiple column values. 
                    Choose one of {value.columns}.""")
            value = TriangleBase.from_triangle(value)
            index_intersection = list(set(self.key_labels).intersection(set(value.key_labels)))
            if len(value.key_labels) == 1:
                index_intersection = []
            value.data = value.data.rename({value.columns[0]: key})
            value.columns = [key]
            self.data = (
                self.data.lazy().select(pl.all().exclude(key))
                .join(
                    value.data.lazy().select(
                        index_intersection + value.columns + ['__origin__', '__development__']), 
                    how='left', on=index_intersection + ['__origin__', '__development__'])
                .rename({value.columns[0]: key})
            )
        self.columns = self.columns + [key]
        if not self.is_lazy:
            self.data = self.data.lazy().collect()
    
    @staticmethod
    def _broadcast_axes(a, b):
        def broadcast_index(a, b):
            a = TriangleBase.from_triangle(a)
            a.data = (b.index.lazy().join(
                a.data.lazy().select(
                    pl.col(a.columns + ['__origin__', '__development__'])), 
                how='cross'))
            return a, b

        def broadcast_columns(a, b):
            a = TriangleBase.from_triangle(a)
            a.data = (a.data.select(
                pl.col(a.key_labels + ['__origin__', '__development__']),
                *[pl.col(a.columns[0]).alias(col) for col in b.columns]))
            a.columns = b.columns
            return a, b

        def broadcast_origin(a, b):
            a = TriangleBase.from_triangle(a)
            a.data = a.data.drop('__origin__').join(
                b.origin.alias('__origin__').to_frame().lazy(), 
                how='cross')
            return a, b

        def broadcast_development(a, b):
            a = TriangleBase.from_triangle(a)
            a.data = a.data.drop('__development__').join(
                (b.valuation if b.is_val_tri else b.development
                ).alias('__development__').to_frame().lazy(), 
                how='cross')
            return a, b
        a.data = a.data.lazy()
        b.data = b.data.lazy()
        if a.shape[0] == 1 and b.shape[0] > 1:
            a, b = broadcast_index(a, b)
        if a.shape[0] > 1 and b.shape[0] == 1:
            b, a = broadcast_index(b, a)
        if a.shape[1] == 1 and b.shape[1] > 1:
            a, b = broadcast_columns(a, b)    
        if a.shape[1] > 1 and b.shape[1] == 1:
            b, a = broadcast_columns(b, a)
        if a.shape[2] == 1 and b.shape[2] > 1:
            a, b = broadcast_origin(a, b)    
        if a.shape[2] > 1 and b.shape[2] == 1:
            b, a = broadcast_origin(b, a)
        if a.shape[3] == 1 and b.shape[3] > 1:
            a, b = broadcast_development(a, b)    
        if a.shape[3] > 1 and b.shape[3] == 1:
            b, a = broadcast_development(b, a)
        if not a.is_lazy:
            a.data = a.data.lazy().collect()
        if not b.is_lazy:
            b.data = b.data.lazy().collect()
        return a, b

    def head(self, n: 'int' = 5):
        triangle = TriangleBase.from_triangle(self)
        triangle.data = triangle.data.join(
            self.index.head(n), 
            how='semi', 
            on=self.key_labels)
        triangle.properties.pop('index', None)
        return triangle
    
    def tail(self, n: 'int' = 5):
        triangle = TriangleBase.from_triangle(self)
        triangle.data = triangle.data.join(
            self.index.tail(n), 
            how='semi', 
            on=self.key_labels)
        triangle.properties.pop('index', None)
        return triangle
    
    def filter(self, key, *args, **kwargs):
        triangle = TriangleBase.from_triangle(self)
        triangle.data = triangle.data.filter(key, *args, **kwargs)
        return triangle

    def select(self, key, *args, **kwargs):
        triangle = TriangleBase.from_triangle(self)
        if type(key) is str:
            key = [key]
        if len(set(key).intersection(self.key_labels)) ==len(key):
            triangle.data = triangle.data.select(pl.col(key + ['__origin__', '__development__'] + self.columns, *args, **kwargs))
            triangle.key_labels = key
        elif len(set(key).intersection(self.columns)) ==len(key):
            triangle.data = triangle.data.select(pl.col(self.key_labels + ['__origin__', '__development__'] + key, *args, **kwargs))
            triangle.columns = key
        else:
            raise NotImplementedError()
        return triangle
    
    def join(self, other, on, how, *args, **kwargs):
        triangle = TriangleBase.from_triangle(self)
        triangle.data = triangle.data.join(other, on, how, *args, **kwargs)
        return triangle

    def _compatibility_check(self, other):
        if (self.is_val_tri == other.is_val_tri) or (self.shape[3] == 1 or other.shape[3] == 1):
            join_index = list(set(self.key_labels).intersection(set(other.key_labels)))
            union_index = self.key_labels + [k for k in other.key_labels if k not in self.key_labels]
            destination_columns = self.columns
            if len(set(self.columns) - set(other.columns)) == 0:
                source_columns = list(zip(self.columns, [c + '_right' for c in self.columns]))
            else:
                source_columns = list(zip(self.columns, [c + '_right' for c in other.columns]))
        else:
            raise ValueError(
                """Triangle arithmetic requires triangles to be broadcastable 
                or on the same lag basis (development or valuation)."""
            )
        return join_index, union_index, source_columns, destination_columns
    
    def __arithmetic__(self, other, operation):
        if type(other) != type(self):
            other = self._triangle_literal(other)
        valuation = max(self.valuation_date, other.valuation_date)
        a, b = TriangleBase._broadcast_axes(self, other)
        join_index, union_index, source_columns, destination_columns = \
            a._compatibility_check(b)
        a = TriangleBase.from_triangle(a)
        if (not (a.is_lazy and b.is_lazy) and len(a.data) == len(b.data) and
            (a.data.select(a.key_labels + ['__origin__', '__development__']) == 
            b.data.select(b.key_labels + ['__origin__', '__development__'])).min().min(axis=1)[0]):
            a.data = (
                pl.concat(
                    (a.data.lazy().collect(), 
                     b.data.lazy().collect()
                      .rename({k: source_columns[num][1] for num, k in enumerate(b.columns)})
                      .drop(b.key_labels + ['__origin__', '__development__'])), how='horizontal')
                .lazy()
                .select(
                    pl.col(union_index + ['__origin__', '__development__']),
                    *[(getattr(pl.col(source_columns[num][0]).fill_null(0), operation)(
                    pl.col(source_columns[num][1]).fill_null(0))).alias(col) 
                    for num, col in enumerate(destination_columns)]))
        else:
            a.data = (
                a.data.lazy()
                .join(
                    b.data.lazy()
                    .rename({k: source_columns[num][1] for num, k in enumerate(b.columns)}), 
                    how='outer',
                    on=join_index + ['__origin__', '__development__'])
                .with_columns(
                    pl.col('__development__').alias('__valuation__') 
                    if a.is_val_tri else vcol.alias('__valuation__'))
                .filter(pl.col('__valuation__') <= valuation)
                .select(
                    pl.col(union_index + ['__origin__', '__development__']),
                    *[(getattr(pl.col(source_columns[num][0]).fill_null(0), operation)(
                    pl.col(source_columns[num][1]).fill_null(0))).alias(col) 
                    for num, col in enumerate(destination_columns)])
            )
        if not self.is_lazy:
            a.data = a.data.collect()
        a.properties = {}
        return a
    
    def _triangle_literal(self, value):
        """ Purpose is to densly populate all origin/development entries whether they 
        exist in triangle data or not."""
        other = TriangleBase.from_triangle(self)
        other.data = (
            self.origin.alias('__origin__').to_frame().lazy()
            .join(
                (self.valuation if self.is_val_tri 
                 else self.development).alias('__development__').to_frame().lazy(), 
                how='cross')
            .filter((pl.col('__development__') if self.is_val_tri 
                     else vcol) <= self.valuation_date)
        ).with_columns(pl.lit("Total").alias('Total'), 
                       pl.lit(value).alias('__value__'))
        other.columns = ['__value__']
        if not self.is_lazy:
            other.data = other.data.collect()
        return other
    
    def _triangle_unary(self, unary, *args):
        triangle = TriangleBase.from_triangle(self)
        triangle.data = (
            triangle.data.select(
            pl.col(self.key_labels + ['__origin__', '__development__']),
            *[getattr(pl.col(c), unary)(*args).alias(c) for c in self.columns])
        )
        return triangle

    def __add__(self, other):
        return self.__arithmetic__(other, '__add__')
    
    def __radd__(self, other):
        return self.__add__(other)  

    def __sub__(self, other):
        return self.__arithmetic__(other, '__sub__')

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        return self.__arithmetic__(other, '__mul__')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.__arithmetic__(other, '__truediv__')

    def __rtruediv__(self, other):
        return (self ** -1) * other
    
    def __neg__(self):
        return self._triangle_unary('__neg__')
    
    def __abs__(self):
        return self._triangle_unary('__abs__')
    
    def __pow__(self, n):
        return self._triangle_unary('__pow__', n)
    
    def __pos__(self):
        return self._triangle_unary('__pos__')
    
    def __round__(self, n):
        return self._triangle_unary('round', n)
    
    def __len__(self):
        return self.shape[0]
    
    

    def __contains__(self, value):
        raise NotImplementedError()

    def __lt__(self, value):
        raise NotImplementedError()

    def __le__(self, value):
        raise NotImplementedError()
        
    def copy(self):
        return TriangleBase.from_triangle(self)

    def to_frame(self, keepdims=False, implicit_axis=False, *args, **kwargs):
        """ Converts a triangle to a pandas.DataFrame.
        Parameters
        ----------
        keepdims : bool
            If True, the triangle will be converted to a DataFrame with all
            dimensions intact.  The argument will force a consistent DataFrame
            format regardless of whether any dimensions are of length 1.
        implicit_axis : bool
            When keepdims is True, this denotes whether to include the implicit
            valuation axis in addition to the origin and development.
        Returns
        -------
            pandas.DataFrame representation of the Triangle.
        """
        if self.shape[:2] == (1, 1) and not keepdims:
            return self.wide()
        if implicit_axis:
            if self.is_val_tri:
                return self.data.sort(
                    pl.col(self.key_labels + ['__origin__', '__development__'])).select(
                    pl.col(self.key_labels),
                    pl.col('__origin__').alias('origin'),
                    pl.col('__development__').alias('valuation'),
                    dcol.alias('development'),
                    pl.col(self.columns))
            else:
                return self.data.sort(
                    pl.col(self.key_labels + ['__origin__', '__development__'])).select(
                    pl.col(self.key_labels),
                    pl.col('__origin__').alias('origin'),
                    pl.col('__development__').alias('development'),
                    vcol.alias('valuation'),
                    pl.col(self.columns))
        else:
            return self.data.sort(
                pl.col(self.key_labels + ['__origin__', '__development__'])).select(
                pl.col(self.key_labels),
                pl.col('__origin__').alias('origin'),
                pl.col('__development__').alias('valuation' if self.is_val_tri else 'development'),
                pl.col(self.columns))
        
    def sort(self):
        self.data = self.data.sort(self.key_labels + ['__origin__', '__development__'])
        return self

        
class PlTriangleGroupBy:
    def __init__(self, obj, by, axis=0, **kwargs):
        self.obj = TriangleBase.from_triangle(obj)
        self.axis = self.obj._get_axis(axis)
        self.by = [by] if type(by) is str else by
        if self.axis == 0:
            self.groups = obj.data.group_by(
                self.by + ['__origin__', '__development__'])
        else:
            raise NotImplementedError()
        self.columns = self.obj.columns

    def __getitem__(self, key):
        self.columns = [key] if type(key) is str else key
        return self
    
    def _agg(self, agg, axis=1, *args, **kwargs):
        axis = self.obj._get_axis(axis)
        if axis == 0:
            self.obj.data = self.groups.agg(
                getattr(pl.col(self.columns), agg)(*args, **kwargs))
            self.obj.properties.pop('index', None)
            self.obj.properties.pop('key_labels', None)
        else:
            raise ValueError(f'axis {axis} is not supported')
        self.obj.columns = self.columns
        return self.obj

    def sum(self, axis=0):
        return self._agg('sum', axis)
    
    def mean(self, axis=0):
        return self._agg('mean', axis)
    
    def min(self, axis=0):
        return self._agg('min', axis)

    def max(self, axis=0):
        return self._agg('max', axis)
    
    def median(self, axis=0):
        return self._agg('median', axis)
    
    def std(self, axis=0):
        return self._agg('std', axis)
    
    def var(self, axis=0):
        return self._agg('var', axis)
    
    def product(self, axis=0):
        return self._agg('product', axis)
    
    def quantile(self, axis=0, quantile=0.5):
        return self._agg('quantile', axis, quantile=quantile)
    
