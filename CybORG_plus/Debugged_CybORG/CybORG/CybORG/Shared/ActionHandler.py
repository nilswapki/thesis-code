# Copyright DST Group. Licensed under the MIT license.
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared.Actions.Action import Action
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared.Observation import Observation


class ActionHandler:
    def __init__(self):
        pass

    def perform(self, action: Action) -> Observation:
        raise NotImplementedError
