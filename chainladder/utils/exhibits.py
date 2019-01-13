import pandas as pd


class Exhibits:
    """ Wrapper to pandas.to_excel that allows for composite exhibits,
        multiple sheets, and custom formatting using xlsxwriter.

        Arguments: None
        Returns: instance of Exhibits
    """
    # Class level formats - can be overridden by end-user
    title1_format = {'font_size': 20, 'align': 'center'}
    title2_format = {'font_size': 16, 'align': 'center'}
    title3_format = {'font_size': 16, 'align': 'center'}
    title4_format = {'font_size': 13, 'align': 'center'}
    index_format = {'num_format': '0;(0)', 'text_wrap': True,
                    'bold': True, 'valign': 'bottom', 'align': 'right'}
    header_format = {'num_format': '0;(0)', 'text_wrap': True, 'bottom': 1,
                     'bold': True, 'valign': 'bottom', 'align': 'right'}
    all_columns = {'align': 'center'}
    money_format = {'num_format': '#,##'}
    percent_format = {'num_format': '0.0%'}
    decimal_format = {'num_format': '0.000'}
    date_format = {'num_format': 'm/d/yyyy'}
    int_format = {'num_format': 'General'}
    text_format = {'align': 'left'}


    def __init__(self):
        ''' Instantiate a workbook '''
        self.sheet_names = []
        self.titles = []
        self.columns = []
        self.col_nums = []
        self.start_row = []
        self.start_col = []
        self.index = []
        self.index_label = []
        self.header = []
        self.formats = []

    def __repr__(self):
        return ""

    def add_exhibit(self, sheet_name, data, header=True, formats='money',
                    start_row=0, start_col=0, index=True, index_label='',
                    title=None, col_nums=True):
        """  Add sheet_names to the class instance.  Each sheet_name will show up on
             its own sheet.
        """
        if type(data) is not list:
            data = [data]
        for num, item in enumerate(data):
            if type(item) is not pd.DataFrame:
                data[num] = item.to_frame()
        if title is None or title == []:
            title = None
        elif len(title) < 4:
            title = title + ['' for item in range(4-len(title))]
        if type(header) is not list:
            if header:
                header = [col for sublist in [item.columns for item in data]
                          for col in sublist]
            elif not header:
                header = ['' for sublist in [item.columns for item in data]
                          for col in sublist]
        if type(formats) is str:
            formats = [formats]*len(header)
        self.sheet_names.append(sheet_name)
        self.titles.append(title)
        self.col_nums.append(col_nums)
        self.columns.append(data)
        self.start_row.append(start_row)
        self.start_col.append(start_col)
        self.index.append(index)
        self.index_label.append(index_label)
        self.header.append(header)
        self.formats.append(formats)
        return self

    def del_exhibit(self, sheet_name):
        """ Removes sheet_name by name from the sheet_name instance. """
        exh_num = self.sheet_names.index(sheet_name)
        del self.sheet_names[exh_num]
        del self.titles[exh_num]
        del self.columns[exh_num]
        del self.col_nums[exh_num]
        del self.start_row[exh_num]
        del self.start_col[exh_num]
        del self.index[exh_num]
        del self.index_label[exh_num]
        del self.header[exh_num]
        del self.formats[exh_num]
        return self

    def to_excel(self, workbook_path):
        """ Creates Excel file at specified path with all sheet_names.
        """
        writer = pd.ExcelWriter(workbook_path, date_format='m/d/yyyy')
        workbook = writer.book
        title1_format = workbook.add_format(self.title1_format)
        title2_format = workbook.add_format(self.title2_format)
        title3_format = workbook.add_format(self.title3_format)
        title4_format = workbook.add_format(self.title4_format)
        money = workbook.add_format(self.money_format)
        percent = workbook.add_format(self.percent_format)
        decimal = workbook.add_format(self.decimal_format)
        integer = workbook.add_format(self.int_format)
        text = workbook.add_format(self.text_format)
        date = workbook.add_format(self.date_format)
        header_format = workbook.add_format(self.header_format)
        index_format = workbook.add_format(self.index_format)
        format_dict = {'money': money, 'percent': percent,
                       'decimal': decimal, 'text': text,
                       'date': date, 'integer': integer}

        for ex_num in range(len(self.sheet_names)):
            title_offset = 0 if self.titles[ex_num] is None else 4
            ex = pd.concat([item for item in self.columns[ex_num]], axis=1)
            columns = [item for item in self.header[ex_num]]
            formats = [item for item in self.formats[ex_num]]

            # Set Data
            start_row = self.start_row[ex_num] + title_offset + 1 + \
                self.col_nums[ex_num]
            ex.to_excel(writer, sheet_name=self.sheet_names[ex_num],
                        header=False, startrow=start_row,
                        startcol=self.start_col[ex_num],
                        index=self.index[ex_num])
            worksheet = writer.sheets[self.sheet_names[ex_num]]
            worksheet.hide_gridlines(2)
            # Set column formats
            for item in range(len(columns)):
                col = item+self.index[ex_num]
                worksheet.set_column(col, col, 12)
                # workaround until xlsxwriter includes a range format
                format = {'type': 'cell', 'criteria': '>=',
                          'value': 0, 'format': format_dict[formats[item]]}
                rng = start_row, col, start_row+len(ex), col
                worksheet.conditional_format(*rng, format)
                format['criteria'] = '<'
                worksheet.conditional_format(*rng, format)

            # Set Header
            if list(set(self.header[ex_num])) != ['']:
                if not self.index[ex_num]:
                    headers = columns
                else:
                    headers = [self.index_label[ex_num]]+columns
                for col_num, value in enumerate(headers):
                    worksheet.write(title_offset+self.start_row[ex_num],
                                    col_num+self.start_col[ex_num],
                                    value, header_format)
                    if self.col_nums[ex_num]:
                        worksheet.write(self.start_row[ex_num]+1+title_offset,
                                        col_num,-col_num-1, header_format)
            # Set Index
            if self.index[ex_num]:
                for row_num, value in enumerate(ex.index.astype(str)):
                    worksheet.write(row_num+start_row, self.start_col[ex_num],
                                    value, index_format)
                worksheet.set_column(0, 0, 15)
            # Set Title
            cols = len(columns)
            if title_offset == 4:
                title_formats = [title1_format, title2_format,
                                 title3_format, title4_format]
                for num, item in enumerate(title_formats):
                    worksheet.merge_range(num+self.start_row[ex_num],
                                          self.start_col[ex_num],
                                          num+self.start_row[ex_num],
                                          cols+self.start_col[ex_num],
                                          self.titles[ex_num][num], item)
        writer.save()
        writer.close()
