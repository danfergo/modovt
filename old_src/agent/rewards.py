from old_src.utils.img import color_map
import numpy as np, math


class TactilePainPleasureReward:

    def in_contact_area(self, rel_depth):
        area = np.array(rel_depth)
        area[area < 1e-05] = 0
        area[area >= 1e-05] = 1
        return area

    def tactile_pixel_reward(self, rel_depth):
        pain_min = 0.005
        pain_range = 0.005
        max_v = (pain_min / 2)
        pain_coeff = 50
        pleasure_coff = 5
        return (-1 * pain_coeff * min(1.0, (rel_depth - pain_min) / pain_range)) \
            if rel_depth > pain_min else (pleasure_coff * math.cos(((rel_depth - max_v) / max_v) * (math.pi / 2)))

    def reward(self, rel_depth, return_maps=False):
        rel_depth[rel_depth < 1e-05] = 0.0

        in_contact_map = self.in_contact_area(rel_depth)
        reward_map = np.vectorize(self.tactile_pixel_reward)(rel_depth)

        area_size = np.sum(in_contact_map)
        r = 0 if area_size == 0 else np.sum(reward_map) / area_size

        if return_maps:
            return r, in_contact_map, reward_map
        return r

    def tactile_color_map(self, reward_map):
        pain_color = (0, 0, 255)
        pleasure_color = (0, 255, 0)
        map_shape = reward_map.shape

        pain_map = np.array(reward_map)
        pain_map[reward_map >= 1e-03] = 0.0
        pain_map *= -1
        pain_map3 = np.stack([pain_map, pain_map, pain_map], axis=2)

        pleasure_map = np.array(reward_map)
        pleasure_map[reward_map < 1e-05] = 0.0
        pleasure_map3 = np.stack([pleasure_map, pleasure_map, pleasure_map], axis=2)

        return np.multiply(pain_map3, color_map(map_shape, pain_color)) + \
               np.multiply(pleasure_map3, color_map(map_shape, pleasure_color))
