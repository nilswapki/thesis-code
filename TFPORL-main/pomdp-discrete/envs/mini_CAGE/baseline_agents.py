
import numpy as np

class Base_agent:
    def __init__(self):
        pass

    def train(self):
        pass

    def get_action(self):
        pass

    def end_episode(self):
        pass

    def set_initial_values(self):
        pass



class B_line_minimal(Base_agent):
    '''
    B_line implementation that is compatible 
    with both minimal and default CAGE.
    '''
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):
    
        # convert to 2D
        # decompose the state
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        host_info = observation[:, 1:].reshape(observation.shape[0], -1, 3)
        actions = np.zeros(observation.shape[0])
        action_selected = np.zeros_like(actions)
        
        # scan subnet 3
        actions, action_selected = self._scan_subnet(
            subnet_idx=2, host_idx=12, host_info=host_info,
             actions=actions, action_selected=action_selected)

        # select action for user 1
        actions, action_selected = self._check_host(
            host_index=9, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # select action for ent 1
        actions, action_selected = self._check_host(
            host_index=2, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # scan subnet 1
        actions, action_selected = self._scan_subnet(
            subnet_idx=0, host_idx=0, host_info=host_info,
             actions=actions, action_selected=action_selected)

        # select action for ent 2
        actions, action_selected = self._check_host(
            host_index=3, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # select action for opserver
        actions, action_selected = self._check_host(
            host_index=7, action_selected=action_selected, 
            actions=actions, host_info=host_info)

        # impact the operational server
        at_op_server = np.invert(action_selected.astype(bool))
        if np.any(at_op_server):
            actions[at_op_server] = 50
            action_selected[at_op_server] = 1

        return actions.reshape(-1, 1).astype(int)


    def _check_host(self, host_index, action_selected, actions, host_info):
        '''Given the index of a host select to scan, exploit or escalate. '''
        
        # scan the host
        num_hosts = host_info.shape[1]
        host_scanned = host_info[:, host_index, 0] == 1
        scan_host = np.logical_and(
            np.invert(host_scanned), np.invert(action_selected.astype(bool)))
        if np.any(scan_host):
            actions[scan_host] = host_index+4
            action_selected[scan_host] = 1

        # exploit user1
        host_exploited = host_info[:, host_index, 1] == 1
        host_privileged = host_info[:, host_index, -1] == 1
        host_access = np.logical_or(host_exploited, host_privileged)
        host_exp = np.logical_and(host_scanned, np.invert(host_access))
        exp_host = np.logical_and(
            host_exp, np.invert(action_selected.astype(bool)))
        if np.any(exp_host):
            actions[exp_host] = num_hosts+4+host_index
            action_selected[exp_host] = 1

        # priv access user 1
        host_priv = np.logical_and(host_exploited, np.invert(host_privileged))
        priv_host = np.logical_and(
            host_priv, np.invert(action_selected.astype(bool)))
        if np.any(host_priv):
            actions[priv_host] = num_hosts*2+4+host_index
            action_selected[priv_host] = 1

        return actions, action_selected

    def _scan_subnet(
        self, subnet_idx, host_idx, host_info, actions, action_selected):

        subnet_unknown = host_info[:, host_idx, -1] == -1
        subnet_unknown = np.logical_and(
            subnet_unknown, np.invert(action_selected.astype(bool)))
        if np.any(subnet_unknown):
            actions[subnet_unknown] = subnet_idx+1
            action_selected[subnet_unknown] = 1
        
        return actions, action_selected

class Meander_minimal(Base_agent):
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)
        self.subnet_structure = np.array([
            0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        self.num_hosts = len(self.subnet_structure)

    def get_action(self, observation, *args, **kwargs):

        # convert to 2D
        # decompose the state
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        host_info = observation[:, 1:].reshape(observation.shape[0], -1, 3)
        actions = np.zeros(observation.shape[0])
        action_selected = np.zeros_like(actions).astype(bool)

        # impact opserver ---------------------------------------

        opserver_priv = (host_info[:, 7, -1] == 1)
        if np.any(opserver_priv):
            actions[opserver_priv] = 50
            action_selected[opserver_priv] = 1

        # subnet scan --------------------------------------------

        host_priv = (host_info[:, :, -1] == 1)[..., None]
        host_unknown = (host_info[:, :, 0] == -1)[..., None]

        # get unique subnet configuration
        unique_subnets = np.unique(self.subnet_structure)
        subnet_idx = (self.subnet_structure[:, None] == unique_subnets).astype(int)
        subnet_idx = subnet_idx[None]

        # identifty subnets with privilege and unknown hosts
        subnet_unknown = np.any(subnet_idx*host_unknown, axis=1)
        subnet_priv = np.any(subnet_idx*host_priv, axis=1)
        subnet_scan = np.logical_and(subnet_unknown, subnet_priv)
        network_scan = np.any(subnet_scan, axis=-1)
        network_scan = np.logical_and(network_scan, np.invert(action_selected))
        if np.any(network_scan):
            selected_subnet = np.argmax(subnet_scan[network_scan], axis=-1)
            actions[network_scan] = selected_subnet+1
            action_selected[network_scan] = 1

        # network scan -------------------------------------------

        host_unscanned = (host_info[:, :, 0] == 0)
        network_unscanned = np.any(host_unscanned, axis=-1)
        network_unscanned = np.logical_and(
            network_unscanned, np.invert(action_selected))
        if np.any(network_unscanned):
            row_probs = (host_unscanned[network_unscanned] != 0)/(
                np.sum((host_unscanned[network_unscanned]  != 0), axis=-1)).reshape(-1, 1)
            selected_host = (
                np.random.rand(len(row_probs), 1) < row_probs.cumsum(axis=-1)).argmax(axis=-1)
            actions[network_unscanned] = selected_host+4
            action_selected[network_unscanned] = 1
        
        # escalate exploited network -------------------------------

        host_exploited = (host_info[:, :, 1] == 1)
        network_exploits = np.any(host_exploited, axis=-1)
        network_exploits = np.logical_and(
            network_exploits, np.invert(action_selected))
        if np.any(network_exploits):
            selected_host = np.argmax(host_exploited[network_exploits], axis=-1)
            actions[network_exploits] = selected_host+2*self.num_hosts+4
            action_selected[network_exploits] = 1

        # exploit host ---------------------------------------------

        # ensure you always ignore the defender with exploitation
        host_scanned = (host_info[:, :, 0] == 1)
        host_priv = (host_info[:, :, -1] == 1)
        host_scanned[:, 0] = False
        host_exploitable = np.logical_and(host_scanned, np.invert(host_priv))
        network_exploitable = np.any(host_exploitable, axis=-1)
        network_exploitable = np.logical_and(
            network_exploitable, np.invert(action_selected))
        if np.any(network_exploitable):
            row_probs = (host_exploitable[network_exploitable] != 0)/(
                np.sum((host_exploitable[network_exploitable]  != 0), axis=-1)).reshape(-1, 1)
            selected_host = (
                np.random.rand(len(row_probs), 1) < row_probs.cumsum(axis=-1)).argmax(axis=-1)
            actions[network_exploitable] = selected_host+self.num_hosts+4
            action_selected[network_exploitable] = 1

        return actions.reshape(-1, 1)
        

class React_restore_minimal(Base_agent):
    '''
    React-restore agent compatible with minimal and default simulator.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.num_hosts = 13
        self.host_list = None
    
    def get_action(self, observation, *args, **kwargs):
        
        # reformat observation
        batch_size = observation.shape[0]
        if self.host_list is None:
            self.host_list = np.zeros((batch_size, self.num_hosts))
        host_info = observation[:, :4*self.num_hosts].reshape(-1, self.num_hosts, 4)

        # update the host list
        exploited_host = host_info[:, :, -1] == 1
        host_idxs = np.where(exploited_host == 1)
        if len(host_idxs[0]) > 0:
            self.host_list[host_idxs[0], host_idxs[1]] = 1

        # restore a host to default
        actions = np.zeros(batch_size)
        host_to_restore = np.any(self.host_list == 1, axis=-1)
        if np.any(host_to_restore):
            selected_host = np.argmax(self.host_list[host_to_restore], axis=-1)
            actions[host_to_restore] = selected_host+40
            filtered_hosts = np.arange(len(observation))[host_to_restore] 
            self.host_list[filtered_hosts, selected_host] = 0

        return actions.reshape(-1, 1).astype(int)
        

class Blue_sleep(Base_agent):
    '''
    Inactive agent compatible with CAGE implementations.
    '''
    def __init__(self, *args, **kwargs):
        super(Base_agent).__init__(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):
        return np.zeros((observation.shape[0], 1)).astype(int)
    
    def end_episode(self):
        pass


class Restore_decoys(React_restore_minimal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.decoy_order = np.array([3, 7, 2, 1, 7, 7, 1, 7, 1, 1])
        self.obs_set = None
        super().reset(*args, **kwargs)

    def get_action(self, observation, *args, **kwargs):

        # get the order of decoys
        if self.obs_set is None:
            self.obs_set = observation.shape[0]
            self.decoy_order = np.tile(
                self.decoy_order.reshape(1, -1), (self.obs_set, 1))
                
        # select the action using react restore logic
        action = super().get_action(observation, *args, **kwargs)

        # if action is zero add an appropriate decoy
        # interested in decoys on opserver, ent2, ent1, ent0
        is_sleep = (action == 0).reshape(-1)
        has_decoys = np.sum(self.decoy_order != -1, axis=-1).reshape(-1)
        valid = np.logical_and(is_sleep, has_decoys)
        if np.any(valid):
            new_action_idx = np.argmax(self.decoy_order != -1, axis=-1)
            new_action = self.decoy_order[np.arange(len(self.decoy_order)), new_action_idx] 
            self.decoy_order[np.arange(len(self.decoy_order))[is_sleep], new_action_idx[is_sleep]] = -1             
            action[is_sleep] = new_action.reshape(-1, 1)[is_sleep]+14

        return action.reshape(-1, 1).astype(int)


# TODO: Check if this actually works as intended
class AdaptiveDefender(Base_agent):
    '''
    Adaptive Defender for CybORG Cage 2 (miniCage version).
    Prioritizes restoring high-value hosts and adapts to repeated attacks.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.num_hosts = 13
        self.threat_levels = np.zeros(self.num_hosts)

    def get_action(self, observation, *args, **kwargs):
        batch_size = observation.shape[0]
        host_info = observation[:, :4 * self.num_hosts].reshape(-1, self.num_hosts, 4)

        # Identify compromised hosts (privileged access or exploited)
        compromised_hosts = (host_info[:, :, 1] == 1) | (host_info[:, :, -1] == 1)

        # Update threat levels (increment if compromised, decay over time)
        self.threat_levels = np.maximum(self.threat_levels * 0.9, compromised_hosts.sum(axis=0))

        # Choose the most critical host to restore
        actions = np.zeros(batch_size)
        for i in range(batch_size):
            if np.any(compromised_hosts[i]):
                target_host = np.argmax(self.threat_levels)
                actions[i] = target_host + 40  # Assuming 40+ corresponds to restore actions
                self.threat_levels[target_host] = 0  # Reset after restoration

        return actions.reshape(-1, 1).astype(int)
    

class AdaptiveDefenderV2(Base_agent):
    """
    Adaptive Defender V2 for CybORG Cage 2 (miniCage version).
    This defender maintains a decaying threat level for each host based on its 
    compromise status over time. It then prioritizes restoring the host with 
    the highest threat.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
    
    def reset(self, *args, **kwargs):
        self.num_hosts = 13
        self.threat_levels = np.zeros(self.num_hosts)
        self.last_restored = np.full(self.num_hosts, -np.inf)
        self.timestep = 0
        # Hyperparameters for threat dynamics:
        self.decay_factor = 0.9      # Decay factor for threat levels per timestep
        self.attack_weight = 2.0     # How much a compromise increases threat
        self.continuous_bonus = 0.3 # Bonus added if a host is continuously compromised
        
    def get_action(self, observation, *args, **kwargs):
        self.timestep += 1
        batch_size = observation.shape[0]
        # Reshape observation: assume first 4*num_hosts entries correspond to host info
        host_info = observation[:, :4 * self.num_hosts].reshape(-1, self.num_hosts, 4)
        
        # Define "compromised" as having either privileged access (index 1) or exploited (last index) 
        # set to 1.
        compromised = ((host_info[:, :, 1] == 1) | (host_info[:, :, -1] == 1)).astype(float)
        
        # Compute average compromise per host over the batch
        compromised_avg = np.mean(compromised, axis=0)  # shape: (num_hosts,)
        
        # Update threat levels: decay the old threat and add new compromise signals
        self.threat_levels = self.decay_factor * self.threat_levels + self.attack_weight * compromised_avg
        
        # Add a continuous bonus for hosts that are compromised in the majority of the batch
        bonus_mask = (compromised_avg > 0.5).astype(float)
        self.threat_levels += self.continuous_bonus * bonus_mask
        
        # For each sample in the batch, select an action if any host is compromised.
        actions = np.zeros(batch_size)
        for i in range(batch_size):
            # For sample i, check which hosts are compromised.
            compromised_hosts = np.where(compromised[i] == 1)[0]
            if compromised_hosts.size > 0:
                # Among the compromised hosts, choose the one with the highest threat level.
                host_threats = self.threat_levels[compromised_hosts]
                selected_host = compromised_hosts[np.argmax(host_threats)]
                # The action is assumed to be "restore host" with index offset of 40.
                actions[i] = selected_host + 40
                # Reset the threat level for the restored host.
                self.threat_levels[selected_host] = 0
                self.last_restored[selected_host] = self.timestep
            else:
                # No host compromised: no restoration action (return 0)
                actions[i] = 0
        
        return actions.reshape(-1, 1).astype(int)
    


class ProactiveDefender(Base_agent):
    """
    ProactiveDefender uses a risk-scoring mechanism for each host.
    
    It keeps a running "risk score" per host that:
      - Increases when the host is compromised (based on observation indicators).
      - Decays over time.
      - Receives a bonus if the host is continuously compromised.
    
    When a host's risk score exceeds a threshold, the defender restores it.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self, *args, **kwargs):
        self.num_hosts = 13
        # Initialize risk scores for each host
        self.risk_scores = np.zeros(self.num_hosts)
        self.timestep = 0
        # Hyperparameters for risk dynamics:
        self.risk_increase_factor = 1.0   # Increase in risk per compromise unit
        self.risk_decay_factor = 0.95     # Decay factor per timestep
        self.hysteresis_bonus = 0.2       # Bonus if host is continuously compromised
        self.restoration_threshold = 0.3  # Threshold above which restoration is triggered

    def get_action(self, observation, *args, **kwargs):
        self.timestep += 1
        batch_size = observation.shape[0]
        # Reshape observation: assuming first 4*num_hosts entries encode host info
        host_info = observation[:, :4 * self.num_hosts].reshape(-1, self.num_hosts, 4)
        
        # Define "compromised" as having privileged access (index 1) or exploited (last index) set to 1.
        compromised = ((host_info[:, :, 1] == 1) | (host_info[:, :, -1] == 1)).astype(float)
        
        # Compute average compromise per host across the batch
        comp_avg = np.mean(compromised, axis=0)  # shape: (num_hosts,)
        
        # Update risk scores: decay old risk and add current compromise signal.
        self.risk_scores = self.risk_scores * self.risk_decay_factor + self.risk_increase_factor * comp_avg
        
        # Add a bonus if a host is compromised by more than half the batch
        continuous_compromise = (comp_avg > 0.5).astype(float)
        self.risk_scores += self.hysteresis_bonus * continuous_compromise

        # Initialize actions as zeros (no restoration action)
        actions = np.zeros(batch_size)
        
        # For each sample in the batch, if any host has a risk score above threshold, restore the most risky one.
        for i in range(batch_size):
            # Find indices of hosts with risk above threshold
            risky_hosts = np.where(self.risk_scores > self.restoration_threshold)[0]
            if risky_hosts.size > 0:
                # Among risky hosts, choose the one with the highest risk score.
                chosen_host = risky_hosts[np.argmax(self.risk_scores[risky_hosts])]
                # Set action to restore that host (assuming action = host index + 40)
                actions[i] = chosen_host + 40
                # Reset risk for the restored host
                self.risk_scores[chosen_host] = 0
            else:
                actions[i] = 0

        return actions.reshape(-1, 1).astype(int)
    


class ProactiveDefenderWithDecoys(Base_agent):
    """
    ProactiveDefenderWithDecoys uses a risk-scoring mechanism for each host to decide on
    restoration actions and decoy placements. It increases a host's risk when compromised,
    decays the risk over time, and deploys either a restoration or decoy action depending on 
    the current risk level.

    - Restoration: if risk exceeds the restoration_threshold, restore that host.
    - Decoy: if risk is moderately high (>= decoy_threshold but < restoration_threshold), deploy a decoy.
    
    The restoration action is assumed to be host index + restoration_offset,
    and the decoy action is assumed to be host index + decoy_offset.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()
    
    def reset(self, *args, **kwargs):
        self.num_hosts = 13
        self.risk_scores = np.zeros(self.num_hosts)
        self.timestep = 0
        # Hyperparameters for risk dynamics:
        self.risk_increase_factor = 1.0    # Increase per unit compromise
        self.risk_decay_factor = 0.95      # Risk decays by this factor each timestep
        self.hysteresis_bonus = 0.2        # Bonus for continuous compromise
        
        # Thresholds for decision-making:
        self.restoration_threshold = 0.3   # Restore if risk >= this
        self.decoy_threshold = 0.15        # Deploy decoy if risk >= this but < restoration_threshold
        
        # Action offsets (adjust according to your environment's action space)
        self.restoration_offset = 40       # Action = host index + 40 for restoration
        self.decoy_offset = 14             # Action = host index + 14 for decoy deployment
    
    def get_action(self, observation, *args, **kwargs):
        self.timestep += 1
        batch_size = observation.shape[0]
        # Reshape observation assuming first 4*num_hosts entries encode host info.
        host_info = observation[:, :4 * self.num_hosts].reshape(-1, self.num_hosts, 4)
        
        # Determine compromised status: here, a host is considered compromised if either
        # "privileged access" (index 1) or "exploited" (last index) equals 1.
        compromised = ((host_info[:, :, 1] == 1) | (host_info[:, :, -1] == 1)).astype(float)
        
        # Compute average compromise per host across the batch.
        comp_avg = np.mean(compromised, axis=0)  # shape: (num_hosts,)
        
        # Update risk scores: decay old risk and add current compromise signal.
        self.risk_scores = self.risk_scores * self.risk_decay_factor + self.risk_increase_factor * comp_avg
        
        # Add bonus if a host is continuously compromised (e.g. compromised by >50% of observations).
        continuous_compromise = (comp_avg > 0.5).astype(float)
        self.risk_scores += self.hysteresis_bonus * continuous_compromise
        
        actions = np.zeros(batch_size)
        
        for i in range(batch_size):
            # If any host exceeds the restoration threshold, choose the most risky one for restoration.
            if np.any(self.risk_scores >= self.restoration_threshold):
                risky_hosts = np.where(self.risk_scores >= self.restoration_threshold)[0]
                chosen_host = risky_hosts[np.argmax(self.risk_scores[risky_hosts])]
                actions[i] = chosen_host + self.restoration_offset
                # Reset the risk for that host once restored.
                self.risk_scores[chosen_host] = 0
            # Otherwise, if any host has a risk above the decoy threshold, deploy a decoy.
            elif np.any(self.risk_scores >= self.decoy_threshold):
                decoy_hosts = np.where(self.risk_scores >= self.decoy_threshold)[0]
                chosen_host = decoy_hosts[np.argmax(self.risk_scores[decoy_hosts])]
                actions[i] = chosen_host + self.decoy_offset
                # Optionally, reduce the risk score after deploying a decoy.
                self.risk_scores[chosen_host] *= 0.5
            else:
                # No action if no host is sufficiently risky.
                actions[i] = 0
        
        return actions.reshape(-1, 1).astype(int)




if __name__ == '__main__':
        
    from minimal import SimplifiedCAGE

    seeds = range(10) # random.randint(1, 100000)
    

    # initialise environment
    batch_size = 1
    env = SimplifiedCAGE(num_envs=batch_size)
    for seed in seeds:
        np.random.seed(seed)
        s, _ = env.reset()

        # initialise the agents 
        red_agent = Meander_minimal()
        blue_agent = React_restore_minimal()

        reward_log = []
        actions = []
        total_reward = np.zeros(batch_size)
        for i in range(1000):
            #print('###################')

            #print(f"{i} - {s['Red']}")

            #print(s['Blue'][:, :-26].reshape(s['Blue'].shape[0], -1, 4))

            blue_action = blue_agent.get_action(observation=s['Blue'])
            red_action = red_agent.get_action(observation=s['Red']) 
            #print(f'Red: {red_action.reshape(-1)} - Blue: {blue_action.reshape(-1)}')
            s, r, d, i = env.step(
                blue_action=blue_action, red_action=red_action)
            total_reward += r['Blue'].reshape(-1)
            reward_log.append(r['Blue'].reshape(-1))
            #print('Reward ', r['Blue'])
            #print(actions.append(red_action[0]))


        #print(actions)
        print(f'Total Reward: {total_reward}' )
        #print(np.stack(reward_log, axis=-1))
        #print('SEED', seed)
            


    
    