# ===================== AE自编码器 =====================
class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

def cluster_states(Z_train, Z_test, n_clusters, seed):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=seed
    )

    train_label = model.fit_predict(Z_train)
    test_label = model.predict(Z_test)

    return train_label, test_label

# ===================== 行为策略计算 =====================
def get_behavior_policy_discrete(df, n_actions=6):
    behavior_policy = {}
    global_act_prob = df["medication_action"].value_counts(normalize=True).to_dict()
    for a in range(n_actions):
        if a not in global_act_prob:
            global_act_prob[a] = 1e-8
    for _, row in df.iterrows():
        s = int(row["state_id_dl"])
        a = int(row["medication_action"])
        if s not in behavior_policy:
            behavior_policy[s] = {act: 1e-8 for act in range(n_actions)}
        behavior_policy[s][a] += 1
    for s in behavior_policy:
        total = sum(behavior_policy[s].values())
        for a in behavior_policy[s]:
            behavior_policy[s][a] /= total
    return behavior_policy, global_act_prob

# ===================== 构建MDP Transition =====================
def build_transitions(df):
    rows = []
    df = df.sort_values(["PTID", "VISDATE"]).copy()
    for pid, d in df.groupby("PTID"):
        d = d.sort_values("VISDATE").reset_index(drop=True)
        for i in range(len(d)):
            rows.append({
                "state": int(d.loc[i, "state_id_dl"]),
                "action": int(d.loc[i, "medication_action"]),
                "reward": float(d.loc[i, "reward"]),
                "next_state": int(d.loc[i, "next_state_id_dl"]),
                "done": True if i == len(d) - 1 else False
            })
    transitions = pd.DataFrame(rows)
    return transitions.reset_index(drop=True)

# ===================== QLearning MDP模型 =====================
class QLearningMDP:
    def __init__(self, n_states, n_actions=6, gamma=0.95, alpha=0.1, epsilon_eval=0.05):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_eval = epsilon_eval
        self.Q = np.zeros((n_states, n_actions))

    def train(self, trans_df, n_epochs=200):
        data = trans_df.values.tolist()
        for epoch in range(n_epochs):
            random.shuffle(data)
            for s, a, r, s_next, done in data:
                s = int(s)
                a = int(a)
                s_next = int(s_next)
                r = float(r)
                done = bool(done)
                if done:
                    target = r
                else:
                    target = r + self.gamma * np.max(self.Q[s_next])
                self.Q[s, a] += self.alpha * (target - self.Q[s, a])

    def select_action(self, s):
        return int(np.argmax(self.Q[int(s)]))

    def get_action_probs(self, s):
        p = np.ones(self.n_actions) * (self.epsilon_eval / self.n_actions)
        best_a = self.select_action(s)
        p[best_a] += 1.0 - self.epsilon_eval
        return p

    @property
    def policy(self):
        return np.argmax(self.Q, axis=1)

# ===================== 反事实指标   =====================
def calculate_counterfactual_metrics(model, test_df, train_trans_df):
    eval_df = test_df.dropna(subset=["reward"]).copy()
    reward_table = train_trans_df.groupby(["state", "action"])["reward"].mean().to_dict()
    global_reward = train_trans_df["reward"].mean()
    delta_r_list = []
    success_list = []
    doctor_cf_rewards = []
    model_cf_rewards = []
    for _, row in eval_df.iterrows():
        s = int(row["state_id_dl"])
        a_doc = int(row["medication_action"])
        a_model = int(model.select_action(s))
        r_doc_real = float(row["reward"])
        r_doc_cf = reward_table.get((s, a_doc), global_reward)
        r_model_cf = reward_table.get((s, a_model), global_reward)
        delta_r = r_model_cf - r_doc_real
        doctor_cf_rewards.append(r_doc_cf)
        model_cf_rewards.append(r_model_cf)
        delta_r_list.append(delta_r)
        success_list.append(1 if r_model_cf > r_doc_cf else 0)
    return {
        "doctor_estimated_delta_mmse": np.mean(doctor_cf_rewards) if doctor_cf_rewards else 0.0,
        "model_estimated_delta_mmse": np.mean(model_cf_rewards) if model_cf_rewards else 0.0,
        "counterfactual_delta_mmse": np.mean(delta_r_list) if delta_r_list else 0.0,
        "success_rate": np.mean(success_list) if success_list else 0.0,
        "n_cf_eval": len(delta_r_list)
    }
