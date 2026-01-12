import numpy as np

"""Custom reward shaping per Crafter (risorse +0.1, health +0.02, tools +0.3, morte -1.0)."""

class CrafterRewardShaper:
    """Modella reward custom per incentivare raccolta risorse e sopravvivenza."""
    def __init__(self):
        self.bonus_tracker = {
            'resource_collection': [],
            'health_management': [],
            'tool_usage': [],
            'death_penalty': []
        }

    def calculate_shaped_reward(self, native_reward, info, previous_info):
        shaped_reward = native_reward
        bonuses = {
            'resource_collection': 0.0,
            'health_management': 0.0,
            'tool_usage': 0.0,
            'death_penalty': 0.0
        }

        if previous_info is None:
            return shaped_reward, bonuses

        # Penalità morte (health passa da >0 a 0)
        curr_health = info.get('inventory', {}).get('health', 10)
        prev_health = previous_info.get('inventory', {}).get('health', 10)
        if curr_health == 0 and prev_health > 0:
            bonuses['death_penalty'] = -1.0

        # Bonus +0.1 per ogni risorsa raccolta (wood, stone, coal, iron, diamond)
        bonuses['resource_collection'] = self._calculate_resource_bonus(info, previous_info)

        # Bonus +0.02 per health alto (incentiva sopravvivenza)
        bonuses['health_management'] = self._calculate_health_bonus(info)

        # Bonus +0.3 per crafting tools (pickaxe, sword - importante per progressione)
        bonuses['tool_usage'] = self._calculate_tool_bonus(info, previous_info)

        total_bonus = sum(bonuses.values())
        shaped_reward += total_bonus

        for key in bonuses:
            self.bonus_tracker[key].append(bonuses[key])

        return shaped_reward, bonuses

    def _calculate_resource_bonus(self, info, previous_info):
        bonus = 0.0
        resources = ['wood', 'stone', 'coal', 'iron', 'diamond', 'sapling']
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        for resource in resources:
            curr_count = curr_inv.get(resource, 0)
            prev_count = prev_inv.get(resource, 0)
            if curr_count > prev_count:
                bonus += 0.1 * (curr_count - prev_count)
        return bonus

    def _calculate_health_bonus(self, info):
        bonus = 0.0
        curr_inv = info.get('inventory', {})
        if curr_inv.get('health', 0) > 5:
            bonus += 0.02
        if curr_inv.get('food', 0) > 5:
            bonus += 0.02
        if curr_inv.get('drink', 0) > 5:
            bonus += 0.02
        return bonus

    def _calculate_tool_bonus(self, info, previous_info):
        bonus = 0.0
        curr_inv = info.get('inventory', {})
        prev_inv = previous_info.get('inventory', {})
        tools = ['wood_pickaxe', 'stone_pickaxe', 'iron_pickaxe',
                 'wood_sword', 'stone_sword', 'iron_sword']
        for tool in tools:
            if curr_inv.get(tool, 0) > prev_inv.get(tool, 0):
                bonus += 0.3
        return min(bonus, 0.3)

    def get_statistics(self):
        stats = {}
        for key, values in self.bonus_tracker.items():
            stats[key] = {'mean': np.mean(values) if values else 0.0,
                          'total': np.sum(values) if values else 0.0,
                          'count': len(values)}
        return stats

    def reset_episode(self):
        for key in self.bonus_tracker:
            self.bonus_tracker[key] = []