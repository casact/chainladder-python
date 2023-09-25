import polars as pl

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
        cumulative=None, pattern=False, lazy=False, trailing=False, 
        *args, **kwargs
    ):
        if data is None:
            return
        
        # Static attributes
        self.columns = [columns] if type(columns) is str else columns
        index = [index] if type(index) is str else index or []
        self.is_cumulative = cumulative
        self.is_pattern = pattern
        self.is_lazy = lazy
        self._properties = {}

        if valuation is None:
            __development__ = TriangleBase._format_valuation(
                data, origin, origin_format
            ).dt.month_end().max().alias('__development__')
        else:
            __development__ = TriangleBase._format_valuation(
                data, valuation, valuation_format
            ).dt.month_end().alias('__development__')
        __origin__ = TriangleBase._format_origin(
            data, origin, origin_format
        ).dt.truncate("1mo").alias('__origin__')

        data = data.with_columns(__origin__, __development__)
        if data.select('__development__').lazy().collect(streaming=True).n_unique() > 1:
            # Coerce to development triangle
            data = data.select(pl.all().exclude('__development__'), dcol)  
        self.data = (
            data
            .group_by(pl.col(index + ['__origin__', '__development__'])) # Needed for to_incremental/to_cumulative
            .agg(pl.col(columns).sum())
            .select(
                pl.lit('Total').alias('Total') if index == [] else pl.col(index),
                pl.col(['__origin__', '__development__'] + self.columns))
            .sort((index or ['Total']) + ['__origin__', '__development__']))
        if not lazy:
            self.data = self.data.lazy().collect(streaming=True)
        if not trailing:
            self.origin_close = 'DEC'
        else:
            self.origin_close = self.data.select(
                pl.col('__origin__').max().dt.offset_by(
                    {'Y': '12mo', 'M': '1mo', 
                    'Q': '3mo', '2Q': '6mo'}[self.origin_grain]
                ).dt.offset_by('-1d').dt.strftime('%b').str.to_uppercase())[0, 0]

        
    
    @staticmethod
    def from_triangle(triangle):
        obj = TriangleBase()
        obj.data = triangle.data
        obj.columns = triangle.columns
        obj.is_cumulative = triangle.is_cumulative
        obj.is_pattern = triangle.is_pattern
        obj.origin_close = triangle.origin_close
        obj.is_lazy = triangle.is_lazy
        obj._properties = triangle._properties.copy()
        return obj
        
    @property
    def key_labels(self):
        return [
            c for c in self.data.columns 
            if c not in self.columns + 
            ['__origin__', '__development__']]

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
        return self.valuation.max()
    
    @property
    def index(self):
        if 'index' not in self._properties.keys():
            self._properties['index'] = (
                self.data.select(pl.col(self.key_labels)).unique().sort(pl.all())
            )
        return self._properties['index']                                     
    
    @property
    def origin_grain(self):
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
        return grain
    
    @property
    def date_matrix(self):
        if 'date_matrix' not in self._properties.keys():
            if self.is_val_tri:
                self._properties['date_matrix'] = (
                    self.data
                    .group_by('__origin__')
                    .agg(pl.col('__development__').unique())
                    .explode('__development__').select(
                        pl.col('__origin__'),
                        pl.col('__development__').alias('__valuation__'),
                        dcol
                    )).lazy().collect(streaming=True)
            else:
                self._properties['date_matrix'] = (
                    self.data
                    .group_by('__origin__')
                    .agg(pl.col('__development__').unique())
                    .explode('__development__')
                    .with_columns(vcol.alias('__valuation__'))).lazy().collect(streaming=True)
        return self._properties['date_matrix']

        
    @property
    def origin(self):
        if 'origin' not in self._properties.keys():
            self._properties['origin'] = pl.date_range(
                start=self.date_matrix['__origin__'].min(),
                end=self.date_matrix['__origin__'].max(),
                interval={'Y': '12mo', 'M': '1mo', 
                        'Q': '3mo', '2Q': '6mo'}[self.origin_grain], 
                eager=True).alias('origin')
        return self._properties['origin']

    
    @property
    def development(self):
        if 'development' not in self._properties.keys():
            interval = {'Y': 12, '2Q': 6, 'Q': 3, 'M': 1}[self.development_grain]
            self._properties['development'] = pl.Series(
                'development', 
                range(self.date_matrix['__development__'].min(), 
                        self.date_matrix['__development__'].max() + interval, 
                        interval)).cast(pl.UInt16)
        return self._properties['development']

    @property
    def valuation(self):
        if 'valuation' not in self._properties.keys():
            interval={'Y': '12mo', 'M': '1mo', 
                    'Q': '3mo', '2Q': '6mo'}[self.development_grain]
            valuation_range = self.date_matrix.select(
                pl.col('__valuation__').min().alias('vmin').dt.month_start(),
                pl.col('__valuation__').max().alias('vmax'))
            self._properties['valuation'] = pl.date_range(
                start=valuation_range['vmin'][0],
                end=valuation_range['vmax'][0],
                interval=interval, 
                eager=True).dt.month_end().alias('valuation')
        return self._properties['valuation']


    @property
    def is_full(self):
        return (self.data.select(['__origin__', '__valuation__']).n_unique() == 
                self.shape[2] * self.shape[3])
    
    @property
    def is_val_tri(self):
        return dict(
                zip(self.data.columns, self.data.dtypes)
            )['__development__'] != pl.UInt16

        
    @property
    def development_grain(self):
        if len(self.date_matrix['__valuation__'].unique()) == 1:
            grain = 'M'
        else:
            months = self.data.select(
                self.date_matrix['__valuation__']
                    .dt.month().unique().sort().alias('__development__')
            ).lazy().collect(streaming=True)['__development__']
            diffs = months.diff()[1:]
            if len(months) == 1:
                grain = "Y"
            elif (diffs == 6).all():
                grain = "2Q"
            elif (diffs == 3).all():
                grain = "Q"
            else:
                grain = "M"
        return grain

    @property
    def latest_diagonal(self):
        return self[self.valuation==self.valuation_date].to_valuation()
    
    def _get_value_idx(self):
        index = self.index.with_row_count('index')
        origin = self.origin.alias('__origin__').to_frame().with_row_count('origin')
        development = (
            self.valuation 
            if self.is_val_tri else 
            self.development
            ).alias('__development__').to_frame().with_row_count('development')
        return index, origin, development

    @property
    def values(self) -> pl.DataFrame:
        """ Removes labels and replaces with index values """
        index, origin, development = self._get_value_idx()
        return (
            self.data
            .join(origin, how='left', on='__origin__')
            .join(development, how='left', on='__development__')
            .join(index, how='left', on=self.key_labels)
            .select(['index', 'origin', 'development'] + self.columns)
            .rename({i: str(num) for num, i in enumerate(self.columns)}))
    
    def apply_labels_to_values(self, other: pl.DataFrame):
        """ Removes index values and replaces with labels """
        index, origin, development = self._get_value_idx()
        triangle = TriangleBase.from_triangle(self)
        triangle.data = (
            other
            .join(origin, how='left', on='origin')
            .join(development, how='left', on='development')
            .join(index, how='left', on='index')
            .rename({str(num): i for num, i in enumerate(self.columns)})
            .select(self.key_labels + ['__origin__', '__development__'] + self.columns))
        return triangle
    
    def to_development(self):
        if self.is_val_tri:
            obj = TriangleBase.from_triangle(self)
            obj.data = obj.data.select(
                pl.col(self.key_labels + ['__origin__']), 
                dcol, pl.col(self.columns))
            return obj
        else:
            return self
        
    def to_valuation(self):
        if not self.is_val_tri:
            obj = TriangleBase.from_triangle(self)
            obj.data = obj.data.select(
                pl.col(self.key_labels + ['__origin__']), 
                vcol, pl.col(self.columns))
            #obj._properties.pop('valuation', None)
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
    def _format_origin(
        data : pl.DataFrame, 
        column : str, 
        format: str) -> pl.Expr:
        if data.select(column).dtypes[0] in ([pl.Date, pl.Datetime]):
            return pl.col(column).cast(pl.Date).dt.month_start()
        else:
            for f in ['%Y%m', '%Y-%m', '%Y', format]:
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
    def _format_valuation(
        data: pl.DataFrame, 
        column: str, 
        format: str) -> pl.Expr:
        if data.select(column).dtypes[0] in ([pl.Date, pl.Datetime]):
            return pl.col(column).cast(pl.Date).dt.month_end()
        else:
            for f in ['%Y%m', '%Y-%m', '%Y', format]:
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
                .select(self.key_labels + ['__origin__', '__development__'] + self.columns)
            )
            obj._properties.pop('index', None)
        elif axis == 1:
            obj.data = self.select(pl.sum_horizontal(self.columns).alias('0'))
        elif axis == 2:
            obj.data = (
                self.data
                .group_by(self.key_labels + ['__development__'])
                .agg(getattr(pl.col(self.columns).fill_null(0), agg)(*args, **kwargs))
                .with_columns(pl.lit(self.origin.min()).alias('__origin__')))
            obj._properties.pop('date_matrix', None)
            obj._properties.pop('origin', None)
        elif axis == 3:
            obj.data = (
                self.data
                .group_by(self.key_labels + ['__origin__'])
                .agg(getattr(pl.col(self.columns).fill_null(0), agg)(*args, **kwargs))
                .with_columns(pl.lit(self.valuation_date).alias('__development__')))
            obj._properties.pop('date_matrix', None)
            obj._properties.pop('development', None)
            obj._properties.pop('valuation', None)
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
        return {
            **{0: 0, 1: 1, 2: 2, 3: 3},
            **{-1: 3, -2: 2, -3: 1, -4: 0},
            **{"index": 0, "columns": 1, "origin": 2, "development": 3},
        }.get(axis, 0)
    
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

    def to_cumulative(self):
        """Method to convert an incremental triangle into a cumulative triangle.

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
        on=self.key_labels + ['__origin__', '__development__']
        triangle.data = (
            self.data.lazy()
            .select(pl.all().exclude('__development__'), col)
            .join(expanded, how='outer', on=on)
            .sort(by=on)
            .group_by(self.key_labels + ['__origin__'])
            .agg(
                pl.col('__development__'),
                pl.col(self.columns).fill_null(pl.lit(0)).cumsum())
            .explode(["__development__"] + self.columns)
            .select(on + self.columns))
        if not self.is_lazy:
            triangle.data = triangle.data.collect(streaming=True)
        triangle.is_cumulative = True
        triangle._properties.pop('date_matrix', None)
        if self.is_val_tri:
            triangle._properties.pop('valuation', None)
            return triangle
        else:
            triangle._properties.pop('development', None)
            return triangle.to_development()
       
    def to_incremental(self, filter_zeros=False):
        """Method to convert an cumlative triangle into a incremental triangle.

        Parameters
        ----------


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
            triangle._properties.pop('date_matrix', None)
            triangle._properties.pop('valuation', None)
            triangle._properties.pop('development', None)
            if not self.is_lazy:
                triangle.data = triangle.data.collect()
            return triangle
        
    @property
    def link_ratio(self):
        numer = self[..., 1:]
        denom = self[..., :numer.shape[2], :-1]
        triangle = 1 / denom * numer.values
        triangle = triangle[triangle.valuation<triangle.valuation_date]
        triangle.is_pattern = True
        return triangle

    
    def to_grain(self, grain="", trailing=False):
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
            triangle = self.to_valuation()
        origin_map = (
            self.origin
            .to_frame().lazy()
            .group_by_dynamic(
                index_column=pl.col('origin'), 
                every=str(grain_dict[ograin_new]) + 'mo',
                offset=offset)
            .agg(pl.col('origin').alias('__origin__'))
            .explode(pl.col('__origin__'))
            .select(pl.col('origin'), pl.col('__origin__')))
        data = (
            triangle.data.lazy()
            .join(origin_map, how='inner', on='__origin__')
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
            .join(development_map, how='inner', on='__development__')
            .drop('__development__')
            .rename({'valuation': '__development__'})
            .group_by(self.key_labels + ['__origin__', '__development__']).sum())
        triangle.data = data
        if not self.is_lazy:
            triangle.data = triangle.data.collect(streaming=True)
        triangle._properties.pop('date_matrix', None)
        if self.origin_grain != ograin_new:
            triangle._properties.pop('origin', None)
        if self.development_grain != dgrain_new:
            triangle._properties.pop('development', None)
            triangle._properties.pop('valuation', None)
        if self.is_val_tri:
            return triangle
        else:
            return triangle.to_development()
        
    def wide(self):
        if self.shape[:2] == (1, 1):
            index = pl.concat((
                pl.Series(range(len(self.origin))).alias('index').to_frame(), 
                self.origin.dt.strftime('%Y-%m').to_frame()), how='horizontal')
            columns = (self.valuation.dt.strftime('%Y-%m') 
                       if self.is_val_tri else 
                       self.development.cast(pl.Utf8))
            return (
                self.data
                .with_columns(
                    (pl.col('__development__').dt.strftime('%Y-%m') 
                    if self.is_val_tri else
                    pl.col('__development__')).alias('development'),
                    pl.col('__origin__').dt.strftime('%Y-%m').alias('origin'),
                    pl.col(self.columns))
                .lazy().collect(streaming=True)
                .pivot(
                    index='origin',
                    columns='development',
                    values=self.columns,
                    aggregate_function='first')
                .join(index, how='left', on='origin')
                .sort('index')
                .select(pl.col(['origin'] + columns.to_list())))
        else:
            raise ValueError(f'Wide format expects shape of (1, 1), but got {self.shape[:2]}')
    
    def _normalize_slice(self, key):
        key = [key] if type(key) is not tuple else list(key)
        key = [slice(item, item + 1 if item != -1 else None, None) if type(item) is int else item for item in key]
        ellipsis_index = [num for num, i in enumerate(key) if i == Ellipsis]
        if key[0] == Ellipsis:
            key = [slice(None, None, None)]*(5 - len(key)) + key[1:]
        if key[-1] == Ellipsis:
            key = key[:-1] + [slice(None, None, None)]*(5 - len(key))
        if len(ellipsis_index) > 0:
            key = key[:ellipsis_index[0]] + [slice(None, None, None)]*(5 - len(key)) + key[ellipsis_index[0] + 1:]
        if len(ellipsis_index) == 0 and len(key) < 4:
            key = key + [slice(None, None, None)]*(4 - len(key))
        return key if type(key) is tuple else tuple(key)
    
    def _slice(self, key):
        triangle = TriangleBase.from_triangle(self)
        s0, s1, s2, s3 = self._normalize_slice(key)
        if s2 != slice(None, None, None):
            s2_val = self.origin[s2]
            s2_expr = [pl.col('__origin__').is_in(s2_val)]
        else:
            s2_val = self.origin
            s2_expr = []
        if s3 != slice(None, None, None):
            s3_val = self.valuation[s3] if self.is_val_tri else self.development[s3]
            s3_expr = [pl.col('__development__').is_in(s3_val)]
        else:
            s3_val = self.valuation if self.is_val_tri else self.development
            s3_expr = []
        if len(s2_expr + s3_expr) > 0:
            triangle = triangle.filter(
                pl.fold(
                    acc=pl.lit(True),
                    function=lambda acc, x: acc & x,
                    exprs=(s2_expr + s3_expr)))
        if s0 != slice(None, None, None):
            s0_val = self.index[s0]
            triangle = triangle.filter_by_df(s0_val)
        triangle._properties['index'] = s0_val.join(
            triangle.index, how='semi', on=self.key_labels)
        triangle._properties['origin'] = s2_val.to_frame().join(
            triangle.origin.to_frame(), how='semi', on=['origin']).to_series()
        if self.is_val_tri:
            triangle._properties['valuation'] = s3_val.to_frame().join(
                triangle.valuation.to_frame(), how='semi', on=['valuation']).to_series()
        else:
            triangle._properties['development'] = s3_val.to_frame().join(
                triangle.development.to_frame(), how='semi', on=['development']).to_series()
        return triangle.select(pl.col(self.columns[s1]))

    def __getitem__(self, key):
        """ Eager materialization. Use select, with_columns and filter for optimized performance """
        triangle = TriangleBase.from_triangle(self)
        if type(key) is str and key in self.columns:
            return self.select(key)
        if type(key) is str and key in self.key_labels:
            return pl.col(key)
        if type(key) is str and key == 'origin':
            return pl.col('__origin__')
        if type(key) is str and key == 'development':
            return dcol if self.is_val_tri else pl.col('__development__')
        if type(key) is str and key == 'valuation':
            return pl.col('__development__') if self.is_val_tri else vcol
        if type(key) in [tuple, slice, int] or (type(key) is list and type(key[0]) is int):
            return self._slice(key)
        elif type(key) is list:
            triangle = triangle.select(pl.col(key))
            return triangle
        elif type(key) is pl.Series and key.dtype == pl.Boolean:
            if key.name == 'valuation':
                key = self.valuation.filter(key)
                return triangle.filter(
                    pl.col('__development__').is_in(key) 
                    if self.is_val_tri else vcol.is_in(key))
            elif key.name == 'development':
                key = self.development.filter(key)
                return triangle.filter(
                    dcol.is_in(key) if self.is_val_tri else 
                    pl.col('__development__').is_in(key))
            elif key.name == 'origin':
                key = self.origin.filter(key)
                return triangle.filter(pl.col('__origin__').is_in(key))
        elif type(key) == pl.Expr:
            return triangle.filter(key)
        else: 
            raise NotImplementedError()
        
    def __setitem__(self, key, value):
        """ Function for pandas style column setting """
        if type(value) is pl.Expr:
            self.data = self.with_columns(value.alias(key)).data
        elif type(value) != type(self):
            value = self._triangle_literal(value)
            value.data = value.data.rename({'__value__': key})
            value.columns = [key]
            self.data = (
                self.data.select(pl.col(self.columns).exclude(key))
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
            if self._is_aligned(value):
                self.data = (
                pl.concat(
                    (self.data.lazy().collect(streaming=True), 
                     value.data.select(pl.col(value.columns[0]).alias(key)).lazy().collect(streaming=True))
                    , how='horizontal')
                .lazy())
            else:
                index_intersection = list(set(self.key_labels).intersection(set(value.key_labels)))
                if len(value.key_labels) == 1:
                    index_intersection = []
                value.data = value.data.rename({value.columns[0]: key})
                value.columns = [key]
                self.data = (
                    self.data.lazy().select(pl.col(self.columns).exclude(key))
                    .join(
                        value.data.lazy().select(
                            index_intersection + value.columns + ['__origin__', '__development__']), 
                        how='left', on=index_intersection + ['__origin__', '__development__'])
                    .rename({value.columns[0]: key})
                )
        self.columns = self.columns + [key]
        if not self.is_lazy:
            self.data = self.data.lazy().collect(streaming=True)
    
    @staticmethod
    def _broadcast_axes(a, b):
        def broadcast_columns(a, b):
            a = TriangleBase.from_triangle(a)
            a.data = (a.data.select(
                pl.col(a.key_labels + ['__origin__', '__development__']),
                *[pl.col(a.columns[0]).alias(col) for col in b.columns]))
            a.columns = b.columns
            return a
        if a.shape[0] == 1 and b.shape[0] > 1:
            a.data = a.data.drop(a.key_labels)
        if a.shape[0] > 1 and b.shape[0] == 1:
            b.data = b.data.drop(b.key_labels)
        if a.shape[1] == 1 and b.shape[1] > 1:
            a = broadcast_columns(a, b)    
        if a.shape[1] > 1 and b.shape[1] == 1:
            b = broadcast_columns(b, a)
        if a.shape[2] == 1 and b.shape[2] > 1:
            a.data = a.data.drop('__origin__')
        if a.shape[2] > 1 and b.shape[2] == 1:
            b.data = b.data.drop('__origin__')
        if a.shape[3] == 1 and b.shape[3] > 1:
            a.data = a.data.drop('__development__')
        if a.shape[3] > 1 and b.shape[3] == 1:
            b.data = b.data.drop('__development__')
        return a, b

    def head(self, n: 'int' = 5):
        return self[:n]
    
    def tail(self, n: 'int' = 5):
        return self[-n:]
    
    def filter(self, *exprs):
        """ Function to apply polars filtering and re-trigger
        affected properties """
        triangle = TriangleBase.from_triangle(self)
        triangle.data = triangle.data.filter(*exprs)
        triangle = self._filter_axes(triangle)
        return triangle
    
    def _filter_axes(self, triangle):
        """ Filters axes, but preserves axis order """
        triangle._properties = {}
        shapes = list(zip(self.shape, triangle.shape))
        if shapes[0][0] == shapes[0][1]:
            triangle._properties['index'] = self.index
        else:
            triangle._properties['index'] = (
                self.index.lazy()
                .with_row_count('__row__')
                .join(triangle.index.lazy(), how='inner', on=triangle.key_labels)
                .sort('__row__')
                .drop('__row__'))
        if shapes[2][0] == shapes[2][1]:
            triangle._properties['origin'] = self.origin
        else:
            triangle._properties['origin'] = self.origin.filter(
                self.origin.is_in(triangle.origin))
        if shapes[3][0] == shapes[3][1]:
            triangle._properties['development'] = self.development
            triangle._properties['valuation'] = self.valuation
        else:
            if self.is_val_tri:
                triangle._properties['valuation'] = self.valuation.filter(
                    self.valuation.is_in(triangle.valuation))
            else:
                triangle._properties['development'] = self.development.filter(
                    self.development.is_in(triangle.development))
        return triangle


    def filter_by_df(self, df):
        triangle = TriangleBase.from_triangle(self)
        triangle.data = (
            triangle.data
            .lazy()
            .join(df.lazy(), how='inner', on=df.columns))
        triangle = self._filter_axes(triangle)
        if not self.is_lazy:
            triangle.data = triangle.data.collect(streaming=True)
        return triangle
        

    def select_index(self, *exprs):
        """ Function to apply polars selection and re-trigger
        affected properties. Does not support pl.all """
        triangle = TriangleBase.from_triangle(self)
        dims = self.key_labels + ['__origin__', '__development__']
        triangle.data = triangle.data.select(pl.col(dims), *exprs)
        return triangle

    def select(self, *exprs):
        """ Function to apply polars selection and re-trigger
        affected properties. Does not support pl.all """
        triangle = TriangleBase.from_triangle(self)
        dims = self.key_labels + ['__origin__', '__development__']
        triangle.data = triangle.data.select(pl.col(dims), *exprs)
        triangle.columns = [c for c in triangle.data.columns if c not in dims]
        return triangle

    def with_columns(self, *exprs):
        """ Function to apply polars selection and re-trigger
        affected properties """
        triangle = TriangleBase.from_triangle(self)
        dims = self.key_labels + ['__origin__', '__development__']
        triangle.data = triangle.data.with_columns(*exprs)
        triangle.columns = [c for c in triangle.data.columns if c not in dims]
        return triangle
    
    def join(self, other):
        """ Method to horizonatl join two triangles together """
        shared_cols = set(self.columns).intersection(set(other.columns))
        if len(shared_cols) > 0:
            raise ValueError(
                f"Column values must be unique, but both triangle have {shared_cols}."
            )
        triangle = TriangleBase.from_triangle(self)
        on =(list(set(self.key_labels).intersection(other.key_labels)) + 
             ['__origin__', '__development__'])
        triangle.data = self.data.join(other.data, on=on, how='inner')
        triangle.columns = self.columns + other.columns
        triangle._properties = {}
        return triangle
    
    def join_index(self, other : pl.DataFrame, *args, **kwargs):
        """ Method to join the index to another df """
        triangle = TriangleBase.from_triangle(self)
        triangle.data = self.data.join(other, how='left', *args, **kwargs)
        triangle._properties.pop('index', None)
        return triangle
    

    def union(self, other):
        """ Method to union two triangles together """
        triangle = TriangleBase.from_triangle(self)
        triangle.data = pl.concat((self.data, other.data), how='align')
        triangle._properties = {}
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
        if (self.origin_grain != other.origin_grain
            or (self.development_grain != other.development_grain 
                and min(self.shape[-1], other.shape[-1]) > 1)):
            raise ValueError(
                "Triangle arithmetic requires both triangles to be the same grain."
            )
        a, b = set(self.key_labels), set(other.key_labels)
        common = a.intersection(b)
        if common not in [a, b]:
            raise ValueError('Index broadcasting is ambiguous between', str(a), 'and', str(b))
        return join_index, union_index, source_columns, destination_columns
    
    def _is_aligned(self, other):
        """ Helper to determine whether horizontal concat is feasible """
        return (
            not (self.is_lazy or other.is_lazy) and # must be eager
            len(self.data) == len(other.data) and # must have same underlying rows
            # must have all non-measure columns be equal
            (self.data.select(self.key_labels + ['__origin__', '__development__']) == 
             other.data.select(other.key_labels + ['__origin__', '__development__'])
             ).min().min(axis=1)[0])
    
    def rename(self, mapping):
        triangle = TriangleBase.from_triangle(self)
        triangle.data = triangle.data.rename(mapping)
        triangle.columns = [mapping.get(c, c) for c in self.columns]
        return triangle


    def __arithmetic__(self, other, operation):
        if type(other) == pl.DataFrame:
            other = self.apply_labels_to_values(other)
        if type(other) != type(self):
            other = self._triangle_literal(other)
            valuation = self.valuation_date
            a, b = self, other
        else:
            valuation = max(self.valuation_date, other.valuation_date)
            a, b = TriangleBase._broadcast_axes(self, other)
        join_index, union_index, source_columns, destination_columns = \
            a._compatibility_check(b)
        a = TriangleBase.from_triangle(a)
        if a._is_aligned(b):
            a.data = (
                pl.concat(
                    (a.data.lazy().collect(streaming=True), 
                     b.data.lazy().collect(streaming=True)
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
                .filter((
                    pl.col('__development__').alias('__valuation__') 
                    if a.is_val_tri else vcol.alias('__valuation__')) <= valuation)
                .select(
                    pl.col(union_index + ['__origin__', '__development__']),
                    *[(getattr(pl.col(source_columns[num][0]).fill_null(0), operation)(
                    pl.col(source_columns[num][1]).fill_null(0))).alias(col) 
                    for num, col in enumerate(destination_columns)])
            )
        if not self.is_lazy:
            a.data = a.data.collect(streaming=True)
        a._properties = {}
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
        ).with_columns(pl.lit(value).alias('__value__'))
        other.columns = ['__value__']
        other.is_lazy = True
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
    
    def __eq__(self, other):
        if (type(other) != type(self) or 
            self.shape != other.shape
            or len(self.data) != len(other.data)):
            return False
        return (
            self.sort_data().data.select(self.columns).lazy().collect(streaming=True) == 
            other.sort_data().data.select(other.columns).lazy().collect(streaming=True)
        ).min().min(axis=1)[0]
        

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
            index = self.origin.to_frame().with_row_count('index')
            columns = (self.valuation
                        if self.is_val_tri else 
                        self.development).cast(pl.Utf8)
            return (
                self.data
                .with_columns(
                    pl.col('__development__').alias('development'),
                    pl.col('__origin__').alias('origin'),
                    pl.col(self.columns))
                .lazy().collect(streaming=True)
                .pivot(
                    index='origin',
                    columns='development',
                    values=self.columns,
                    aggregate_function='first')
                .join(index, how='left', on='origin')
                .sort('index')
                .select(pl.col(['origin'] + columns.to_list())))
        else:
            alias = 'valuation' if self.is_val_tri else 'development'
            implicit = dcol.alias('development') if self.is_val_tri else vcol.alias('valuation')
            return self.data.sort(
                pl.col(self.key_labels + ['__origin__', '__development__'])).select(
                pl.col(self.key_labels),
                pl.col('__origin__').alias('origin'),
                pl.col('__development__').alias(alias),
                implicit if implicit_axis else pl.col([]),
                pl.col(self.columns))
        
    def sort_data(self):
        self.data = self.data.sort(self.key_labels + ['__origin__', '__development__'])
        return self
    
    def _summary_frame(self):
        return pl.DataFrame({
            "": ["Valuation:", "Grain:", "Shape:", "Index:", "Columns:"],
            "Triangle Summary": [
                self.valuation_date.strftime("%Y-%m"),
                "O" + self.origin_grain + "D" + self.development_grain,
                str(self.shape),
                str(self.key_labels),
                str(self.columns),],})
    
    def __repr__(self):
        if self.shape[:2] == (1, 1):
            data = self.wide()
            return data.__repr__()
        else:
            return self._summary_frame().__repr__()
        
    def sort_index(self, by=None, descending=False):
        self._properties['index'] = self.index.sort(by, descending)


        
class PlTriangleGroupBy:
    def __init__(self, obj, by, axis=0):
        self.obj = TriangleBase.from_triangle(obj)
        self.axis = self.obj._get_axis(axis)
        self.by = [by] if type(by) is str else by
        if self.axis == 0:
            self.groups = obj.data.group_by(
                self.by + ['__origin__', '__development__'])
        elif self.axis == 1:
            if callable(by):
                self.by = [by(c) for c in self.obj.columns]
            elif len(by) == len(self.obj.columns):
                self.by = by
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        self.columns = self.obj.columns

    def __getitem__(self, key):
        self.columns = [key] if type(key) is str else key
        return self
    
    def _agg(self, agg):
        axis = self.obj._get_axis(self.axis)
        if axis == 0:
            self.obj.data = self.groups.agg(
                getattr(pl.col(self.columns), agg)())
            self.obj._properties.pop('index', None)
        elif axis == 1:
            maps = pl.DataFrame(
                {'by': self.by, 'columns': list(self.columns)}
                ).group_by('by').agg(pl.col('columns'))
            maps = dict(
                zip(maps['by'].cast(pl.Utf8).to_list(), 
                    maps['columns'].to_list()))
            self.obj.data = self.obj.data.select(
                pl.col(self.obj.key_labels + ['__origin__', '__development__']), 
                *[getattr(pl, agg + '_horizontal')(pl.col(v)).alias(str(k)) 
                  for k, v in maps.items()])
            self.obj.columns = [
                c for c in self.self.columns 
                if c not in self.obj.key_labels + ['__origin__', '__development__']]
            raise ValueError(f'axis {axis} is not supported')
        self.obj.columns = self.columns
        return self.obj

    def sum(self):
        return self._agg('sum')
    
    def mean(self):
        return self._agg('mean')
    
    def min(self):
        return self._agg('min')

    def max(self):
        return self._agg('max')
    
    def median(self):
        return self._agg('median')
    
    def std(self):
        return self._agg('std')
    
    def var(self):
        return self._agg('var')
    
    def product(self):
        return self._agg('product')
    
    def quantile(self, quantile=0.5):
        return self._agg('quantile', quantile=quantile)
    
