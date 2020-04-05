# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
from xlcompose import (
    DataFrame, Series, Row, Column, Tabs, CSpacer, RSpacer, Title, Image,
    VSpacer, HSpacer, Sheet, load_yaml)
import os

def load_template(template, env=None, **kwargs):
    path = os.path.dirname(os.path.abspath(__file__))
    try:
        return load_yaml(template, env, **kwargs)
    except:
        template = os.path.join(path, 'templates', template.lower() + '.yaml')
        return load_yaml(template, env, **kwargs)
