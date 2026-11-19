"""Episode-local evidence for completed Wait skills with contained objects.

Only explicit Wait skills contribute evidence. Time spent thinking, navigating,
or running another skill never counts. A pair must hold at every sampled skill
step, including the start and completion of the wait.
"""


class WaitEvaluationTracker:
    def __init__(self, sim, propositions, ao_link_map=None):
        self.sim = sim
        self.ao_link_map = ao_link_map
        self.pairs = {
            (obj, receptacle)
            for prop in propositions
            if prop.function_name == "is_waited_inside"
            for obj in prop.args["object_handles"]
            for receptacle in prop.args["receptacle_handles"]
        }
        self.active = {}
        self.completed = set()

    def begin(self, agent_uid):
        self.active[agent_uid] = set(self.pairs)
        self.sample(agent_uid)

    def sample(self, agent_uid):
        from habitat_llm.agent.env.evaluation.predicate_wrappers import SimBasedPredicates

        candidates = self.active.get(agent_uid, set())
        self.active[agent_uid] = {
            (obj, receptacle)
            for obj, receptacle in candidates
            if SimBasedPredicates.is_inside(
                self.sim, [obj], [receptacle], ao_link_map=self.ao_link_map
            ).is_satisfied
        }

    def complete(self, agent_uid):
        self.sample(agent_uid)
        self.completed.update(self.active.pop(agent_uid, set()))

    def cancel(self, agent_uid):
        self.active.pop(agent_uid, None)
