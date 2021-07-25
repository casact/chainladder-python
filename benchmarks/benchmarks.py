# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import chainladder as cl

class TimeSuite:
    def setup(self):
        self.prism = cl.load_sample('prism')

    def time_incr_to_cum(self):
        self.prism.incr_to_cum()

    def time_groupby(self):
        self.prism.groupby(['Line']).sum()

    def time_index_broadcasting(self):
        self.prism / self.prism.groupby(['Line']).sum()

    def time_grain(self):
        self.prism.grain('OYDY')

    def time_dev_to_val(self):
        self.prism.dev_to_val()

    def time_val_to_dev(self):
        self.prism.dev_to_val().val_to_dev()

    def time_fit_chainladder(self):
        cl.Chainladder().fit(
            cl.Development(groupby=lambda x : 1).fit_transform(self.prism['Paid'])
        ).ibnr_

class MemSuite:
    def setup(self):
        self.prism = cl.load_sample('prism')

    def peakmem_incr_to_cum(self):
        self.prism.incr_to_cum()

    def peakmem_groupby(self):
        self.prism.groupby(['Line']).sum()

    def peakmem_index_broadcasting(self):
        self.prism / self.prism.groupby(['Line']).sum()

    def peakmem_grain(self):
        self.prism.grain('OYDY')

    def peakmem_dev_to_val(self):
        self.prism.dev_to_val()

    def peakmem_val_to_dev(self):
        self.prism.dev_to_val().val_to_dev()

    def peakmem_fit_chainladder(self):
        cl.Chainladder().fit(
            cl.Development(groupby=lambda x : 1).fit_transform(self.prism['Paid'])
        ).ibnr_
