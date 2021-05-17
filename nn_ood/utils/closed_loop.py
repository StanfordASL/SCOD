def rollout(policy, env, T):
    obs = env.reset()
    
    obs_list = [obs]
    action_list = []
    rew_list = []
    
    for t in range(T):
        ac = policy.act(obs)
        obs, rew, done, _ = env.step(ac)
        
        obs_list.append(obs)
        action_list.append(ac)
        rew_list.append(rew)
    
    return obs_list, action_list, rew_list

class ObsWrapper:
    def __init__(self, state_policy, sensor_model):
        self.pol = state_policy
        self.model = sensor_model
        
    def act(self, obs):
        state = self.model(obs[None,:,:,:])
        state = state.detach().cpu().numpy()[0,0]
        ac = self.pol.act(state)
        return ac

class WrappedPolicy:
    def __init__(self, nominal_policy, safety_policy, model, thresh):
        self.nom = nominal_policy
        self.safe = safety_policy
        self.model = model
        self.thres = thresh
        
    def act(self, obs):
        state, unc = self.model(obs[None,:,:,:])
        state = state.detach().cpu().numpy()[0,0]
        unc = unc.detach().cpu().numpy()
        
        if unc < self.thres:
            ac = self.nom.act(state)
        else:
            ac = self.safe.act(state)

        return ac