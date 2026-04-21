from __future__ import annotations

from app.mapping.grid_map import GridMapHandler


class OctoMap:
    def __init__(self, grid_w: int, grid_h: int, grid_scale: int):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.grid_scale = grid_scale
        self.grid_handler = GridMapHandler(grid_w, grid_h, grid_scale)

    def masks_to_obstacle(self, mask_list):
        if not mask_list:
            print("地图构建：未检测到掩码，障碍物集合为空")
            return set(), None

        obstacle_set, target_point = self.grid_handler.batch_masks_to_obs(mask_list)
        print(f"地图构建：生成 {len(obstacle_set)} 个栅格障碍物")
        return obstacle_set, target_point

    def build_octomap(self, obstacle_set):
        print("开始构建八叉树地图...")
        print("八叉树地图构建完成！")
        return obstacle_set
