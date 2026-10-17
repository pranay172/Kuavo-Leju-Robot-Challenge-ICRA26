"""Trace ACT selection/dispatch and apply explicit gripper-only controls."""


def apply_gripper_selection(action, record, mode="policy"):
    """Keep all arm values; optionally use fresh, denormalized gripper values."""
    if mode not in ("policy", "latest"):
        raise ValueError("Unknown gripper selection")
    if record is not None:
        record["policy_action"] = action.copy()
    if mode == "policy":
        return action
    if record is None or not record["chunk_refreshed"] or record["chunk_age_steps"] != 0:
        raise ValueError("Newest grippers require one fresh traced prediction per cycle")
    if action.shape != (16,) or record["newest_action"].shape != (16,):
        raise ValueError("Gripper-only control requires the audited 16-dimensional action")
    result = action.copy()
    result[[7, 15]] = record["newest_action"][[7, 15]]
    return result


class PublisherTrace:
    def __init__(self, publisher, monitor):
        self.publisher = publisher
        self.monitor = monitor

    def publish(self, message, *args, **kwargs):
        result = self.publisher.publish(message, *args, **kwargs)
        self.monitor.gripper_commands.append(list(message.position))
        return result

    def __getattr__(self, name):
        return getattr(self.publisher, name)


class ActionSelectionTrace:
    def __init__(self, policy):
        self.policy = policy
        self.original_predict = policy.predict_action_chunk
        self.environment = None
        self.chunk = None
        self.chunk_age = -1
        self.chunk_calls = 0
        self.dispatched = []
        self.gripper_commands = []

        def predict(batch):
            chunk = self.original_predict(batch)
            self.chunk = chunk.detach().clone()
            self.chunk_age = 0
            self.chunk_calls += 1
            return chunk

        policy.predict_action_chunk = predict

    def attach_environment(self, environment):
        self.detach_environment()
        self.environment = environment
        self.chunk = None
        self.chunk_age = -1
        self.original_exec = environment.exec_action
        self.original_publisher = environment.pub_eef_joint

        def execute(action):
            self.dispatched.append(action.copy())
            return self.original_exec(action)

        environment.exec_action = execute
        environment.pub_eef_joint = PublisherTrace(self.original_publisher, self)

    def begin_step(self):
        if self.chunk is not None:
            self.chunk_age += 1
        self.chunk_calls = 0
        self.dispatched.clear()
        self.gripper_commands.clear()

    def prediction_record(self, normalized_action, postprocessor):
        queued = self.policy.config.temporal_ensemble_coeff is None and self.policy.config.n_action_steps > 1
        if self.chunk is None or self.chunk_calls not in (0, 1) or (not queued and self.chunk_calls != 1):
            raise RuntimeError("Expected one fresh ACT prediction or a valid cached chunk")
        if not 0 <= self.chunk_age < self.chunk.shape[1]:
            raise RuntimeError("Cached ACT chunk age is outside its prediction horizon")
        return {
            "chunk_refreshed": self.chunk_calls == 1,
            "chunk_age_steps": self.chunk_age,
            "normalized_chunk": self.chunk.squeeze(0).cpu().numpy().copy(),
            "normalized_selected_action": normalized_action.squeeze(0).detach().cpu().numpy().copy(),
            "newest_action": postprocessor(self.chunk[:, 0].clone()).squeeze(0).cpu().numpy().copy(),
            "active_chunk_action": postprocessor(self.chunk[:, self.chunk_age].clone()).squeeze(0).cpu().numpy().copy(),
        }

    def finish_step(self, record, action, elapsed_sec):
        if len(self.dispatched) != 1 or len(self.gripper_commands) != 1:
            raise RuntimeError("Expected one arm dispatch and one published gripper command per cycle")
        record.update(action=action.copy(), dispatched_action=self.dispatched[0].copy(),
                      gripper_command_255=self.gripper_commands[0], elapsed_sec=elapsed_sec)
        return record

    def detach_environment(self):
        if self.environment is not None:
            self.environment.exec_action = self.original_exec
            self.environment.pub_eef_joint = self.original_publisher
            self.environment = None

    def close(self):
        self.policy.predict_action_chunk = self.original_predict
        self.detach_environment()
