# Copyright DST Group. Licensed under the MIT license.
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared import Observation
from CybORG_plus.Debugged_CybORG.CybORG.CybORG.Shared.Actions import Action


class GlobalAction(Action):
    """Abstract class for a global action.

    A global action is one that operates outside the context of any single
    scenario/game instance.

    Examples would be:
    - creating a new game
    - joining an existing game,
    - listing available games
    """

    def emu_execute(self, team_server) -> Observation:
        raise NotImplementedError
