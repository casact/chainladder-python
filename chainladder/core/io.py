# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import pandas as pd
from sklearn.base import BaseEstimator
import json
import joblib
import dill


class TriangleIO:
    def to_pickle(self, path, protocol=None):
        """ Serializes triangle object to pickle.

        Parameters
        ----------
        path : str
            File path and name of pickle object.
        protocol :
            The pickle protocol to use.

        """
        with open(path, "wb") as pkl:
            dill.dump(self, pkl)

    def to_json(self):
        """ Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format
        """
        metadata = {
            "is_val_tri": self.is_val_tri,
            "is_cumulative": self.is_cumulative,
            "is_pattern": self.is_pattern,
            "columns": list(self.columns),
        }
        out = self.cum_to_incr().dev_to_val().to_frame(keepdims=True).fillna(0)
        x = out.reset_index().to_json(orient="split", date_unit="ns")
        json_dict = {"metadata": json.dumps(metadata), "data": x}
        sub_tris = [k for k, v in vars(self).items() if isinstance(v, TriangleIO)]
        json_dict["sub_tris"] = {
            sub_tri: getattr(self, sub_tri).to_json() for sub_tri in sub_tris
        }
        dfs = [k for k, v in vars(self).items() if isinstance(v, pd.DataFrame)]
        json_dict["dfs"] = {df: getattr(self, df).to_json() for df in dfs}
        dfs = [k for k, v in vars(self).items() if isinstance(v, pd.Series)]
        json_dict["dfs"].update(
            {df: getattr(self, df).to_frame().to_json() for df in dfs}
        )
        return json.dumps(json_dict)


class EstimatorIO:
    """ Class intended to allow persistence of estimator objects """

    def to_pickle(self, path, protocol=None):
        """ Serializes triangle object to pickle.

        Parameters
        ----------
        path : str
            File path and name of pickle object.
        protocol :
            The pickle protocol to use.
        """
        with open(path, "wb") as pkl:
            dill.dump(self, pkl)

    def to_json(self):
        """ Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format
        """
        params = self.get_params(deep=False)
        j = lambda v: v.to_json() if isinstance(v, BaseEstimator) else v
        params = {k: j(v) for k, v in params.items()}
        return json.dumps({"params": params, "__class__": self.__class__.__name__})
