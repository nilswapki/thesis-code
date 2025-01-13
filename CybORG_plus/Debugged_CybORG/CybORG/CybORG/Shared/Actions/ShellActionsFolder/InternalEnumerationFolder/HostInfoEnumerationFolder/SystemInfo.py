# Copyright DST Group. Licensed under the MIT license.
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared.Actions.ShellActionsFolder.InternalEnumerationFolder.HostInfoEnumerationFolder.HostInfoEnumeration import HostInfoEnumeration
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared.Enums import OperatingSystemType
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared.Observation import Observation


class SystemInfo(HostInfoEnumeration):
    def __init__(self, session, agent):
        super().__init__(session, agent)

    def sim_execute(self, state):
        obs = Observation()
        obs.set_success(False)
        if self.session not in state.sessions[self.agent]:
            return obs

        if state.sessions[self.agent][self.session].active:
            host = state.sessions[self.agent][self.session].host
            if host.os_type == OperatingSystemType.WINDOWS:
                obs.set_success(True)
                obs.add_system_info(**(host.get_state()))
            else:
                obs.add_system_info(os_type=host.os_type)
                obs.set_success(False)
        else:
            obs.set_success(False)
        return obs
