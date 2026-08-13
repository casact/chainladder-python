"""
Support Triangle I/O capabilities.
"""
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
import dill
import json
import pandas as pd

from sklearn.base import BaseEstimator


class TriangleIO:
    def to_pickle(self, path, protocol=None):
        """ Serializes triangle object to pickle.

        Parameters
        ----------
        path : str
            File path and name of pickle object.
        protocol :
            The pickle protocol to use.

        Examples
        --------
        Write a Triangle to disk and restore it with :func:`chainladder.read_pickle`.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            import os
            import tempfile

            raa = cl.load_sample('raa')
            fd, path = tempfile.mkstemp(suffix='.pkl')
            os.close(fd)
            raa.to_pickle(path)
            restored = cl.read_pickle(path)
            os.remove(path)
            print(restored.shape)
            print(restored == raa)

        .. testoutput::

            (1, 1, 10, 10)
            True
        """
        with open(path, "wb") as pkl:
            dill.dump(self, pkl)

    def to_json(self):
        """ Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format

        Examples
        --------
        ``to_json`` returns a string that :func:`chainladder.read_json` can
        turn back into a Triangle.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            import json

            raa = cl.load_sample('raa')
            payload = json.loads(raa.to_json())
            print(sorted(payload.keys()))
            print(cl.read_json(raa.to_json()) == raa)

        .. testoutput::

            ['data', 'dfs', 'metadata', 'sub_tris']
            True
        """
        metadata = {
            "is_val_tri": self.is_val_tri,
            "is_cumulative": self.is_cumulative,
            "is_pattern": self.is_pattern,
            "columns": list(self.columns),
        }
        out = self.cum_to_incr().dev_to_val().to_frame(
            keepdims=True, origin_as_datetime=True).fillna(0)
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

        Examples
        --------
        Fitted estimators pickle the same way as Triangles. Restore with
        :func:`chainladder.read_pickle`.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            import os
            import tempfile

            fd, path = tempfile.mkstemp(suffix='.pkl')
            os.close(fd)
            cl.Development(n_periods=3).to_pickle(path)
            restored = cl.read_pickle(path)
            os.remove(path)
            print(type(restored).__name__)
            print(restored.n_periods)

        .. testoutput::

            Development
            3
        """
        with open(path, "wb") as pkl:
            dill.dump(self, pkl)

    def to_json(self):
        """ Serializes triangle object to json format

        Returns
        -------
            string representation of object in json format

        Examples
        --------
        Estimator JSON stores constructor parameters and the class name, so
        :func:`chainladder.read_json` can rebuild the same estimator.

        .. testsetup::

            import chainladder as cl

        .. testcode::

            import json

            payload = json.loads(cl.Development(n_periods=3).to_json())
            print(payload['__class__'])
            print(payload['params']['n_periods'])

        .. testoutput::

            Development
            3
        """
        params = self.get_params(deep=False)
        j = lambda v: v.to_json() if isinstance(v, BaseEstimator) else v
        params = {k: j(v) for k, v in params.items()}
        return json.dumps({"params": params, "__class__": self.__class__.__name__})
