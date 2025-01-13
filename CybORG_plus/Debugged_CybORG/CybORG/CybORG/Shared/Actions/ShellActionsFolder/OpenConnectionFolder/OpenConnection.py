# Copyright DST Group. Licensed under the MIT license.
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared.Actions.ShellActionsFolder.ShellAction import ShellAction
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Simulator.State import State


class OpenConnection(ShellAction):
    def __init__(self, session, agent):
        super().__init__(session, agent)

    def sim_execute(self, state: State):
        pass
