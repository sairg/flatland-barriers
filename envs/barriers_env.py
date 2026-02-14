import numpy as np
import gymnasium as gym
import time
from .base_grid_env import BaseGridEnv

"""         Barriers Environment

Place barriers at random a certain proportion.  
 

"""


class BarriersEnv(BaseGridEnv):
    def __init__(
        self,
        seed=None,
        np_random = None,
        render_mode=None,
        size=16,
        prop=0.1,  # proportions of barriers
        food_prop=0.1,  # proportions of food (if more than 1)
        place_at_random=0,
        multiple_foods=False,
        food_loc=None, # single location?
        change_rate=0.0,  # rate of change for existing barriers
        motion_noise=0,  # noise probability in agent movements.
    ):
        if seed is not None:
            self.set_seed(seed=seed)
        if np_random is not None:
            self.np_random = np_random

        self.prop = prop  # proportions of barriers
        # Place multiple foods, instead of 1?
        self.multiple_foods = multiple_foods
        # In case of multiple foods, the proportion of food.
        self.food_prop = food_prop  # proportion of food
        #print('\n# initialized! barrier prop:', prop, ' food prop:', food_prop)
        self.place_at_random = place_at_random  # random food placement?
        self.change_rate = change_rate

        self.motion_noise_prob = motion_noise
        
        if size <= 0:
            return
        
        super().__init__(
            render_mode,
            size,
            seed=seed,
            place_at_random=self.place_at_random,
            food_loc=food_loc,
            motion_noise=motion_noise,
        )
        
        if np_random is not None:
            self.np_random = np_random
            

    def create_env_map(self, food_loc=None):
        """environment map  barrier distribution"""
        env_map_np = np.zeros((self.map_planes, self.size, self.size), dtype=np.uint8)

        # prop = 0.1 # proportion of barriers
        #print('\n# size is:', self.size, ', barrier proportion:', self.prop)
        x0, y0 = self.home  # home coordinates
        for i in range(self.size):
            for j in range(self.size):
                if i == x0 and j == y0:
                    continue
                # use np_random since seed is set using it..
                if self.np_random.uniform(0, 1) < self.prop:
                    env_map_np[self.barrier, i, j] = 1
                #if np.random.uniform(0, 1) < self.prop:
                #   env_map_np[self.barrier, i, j] = 1

        self.env_map = env_map_np
        # place the food(s)
        if food_loc is not None:
            x, y = food_loc
            self.place_food(False, x, y)
            self.multiple_foods = False
            #print('HERE765', x, y)
        elif self.multiple_foods:
            self.spread_food(env_map_np)
        else:  # place one food, possibly at random.
            self.place_food(self.place_at_random)

        # target (food) locations
        target_location = np.array(
            list(
                np.unravel_index(
                    np.argmax(env_map_np[self.food, :, :]),
                    env_map_np[self.food, :, :].shape,
                )
            )
        )
        #print('\n# target locs:', target_location)
        #print('HERE765:', target_location)
        return env_map_np, target_location

    def spread_food(self, env_map_np):
        for i in range(self.size):
            for j in range(self.size):
                if np.random.uniform(0, 1) < self.food_prop:
                    if env_map_np[self.barrier, i, j] == 0:
                        env_map_np[self.food, i, j] = 1
 
 
    ####
    
    # returns a deep copy of the env
     # (note: certain fields, such as
    # day , or render_mode need to be set perhaps later).
    def make_copy(self):
        newv = BarriersEnv(size=0)

        newv.change_rate = self.change_rate
        newv.prop = self.prop

        newv.food_loc = self.food_loc
        
        newv.render_mode = self.render_mode

        # So behavior remains the same, if desired.
        newv.np_random = self.np_random
        
        newv.sleeper = self.sleeper
        newv.day = self.day  # set day to day 1 ?
        newv.size = self.size
        self.copy_environment_map_to( newv )

        newv.setup_observations()
        newv.setup_actions(self.orientation_frame)
        newv.setup_pygame(None) # batch
        
        newv.step_count = self.step_count 
        newv.time_cost = self.time_cost
        newv.remove_consumed_food = self.remove_consumed_food
        # flag for placing food again
        newv.place_at_random = self.place_at_random

        assert not self.multiple_foods
        newv.multiple_foods = self.multiple_foods
        
        newv.motion_noise_prob = self.motion_noise_prob
        newv.noise_in_direction  = self.noise_in_direction
        
        # If set to true, when injecting noise into motion, motion
        # will be biased (towards staying put).
        newv.no_2steps_fwd = self.no_2steps_fwd
        # remain (don't change location).
        newv.remain_if_barrier = self.remain_if_barrier 
        return newv

    ##

    def copy_environment_map_to(self, newv):
        newv._target_location = self._target_location
        newv.food, newv.barrier = self.food, self.barrier
        newv.home = np.array(
            [self.size / 2, self.size / 2], dtype=np.int32
        )  # start position
        newv.initial_heading = np.array([1, 0])  # initial heading direction
        
        newv.map_planes = self.map_planes
        newv.env_map = np.zeros( (self.map_planes, self.size, self.size), dtype=np.uint8 )

        # copy the entries
        for i in range(newv.size):
            for j in range(newv.size):
                newv.env_map[self.barrier, i, j] = self.env_map[self.barrier, i, j]
                newv.env_map[self.food, i, j] = self.env_map[self.food, i, j]
        
    ####
    
    def change_env_map(self, change_type=1, pit=0, food_loc=None):
        """function that changes environment during experiments"""
        # change the barrier locations (possibly other things too)?

        assert not self.multiple_foods 
        
        if change_type == 2:  # Complete change.
            self.env_map, self._target_location = self.create_env_map(food_loc=food_loc)
            if pit:
                print('\n\n# ** completely changing the barriers.. **** \n\n')
        elif self.change_rate > 0:  # change some of the barriers.
            self.env_map = self.modify_env_map()
            if pit:
                print(
                    '\n\n# ** changed the barriers.. at rate:%.2f.. *** \n\n'
                    % self.change_rate
                )
        # make sure you render, so removed barriers are removed on
        # display..
        if self.render_mode == 'human':
            time.sleep(2)
            print('# rendering display!\n')
            self._render_frame(force=True)

    #
    # Randomly removes barriers and inserts new ones, keeping the
    # total at about the same rate.
    #
    # Alternatives/future: 1) make changes that are correlated (not
    # independent).  2) Maze: create new paths in a maze, remove old
    # ones!
    #
    def modify_env_map(self, pit=0):
        # initialize
        env_map_np = np.zeros((self.map_planes, self.size, self.size), dtype=np.uint8)
        removed, num_food = 0, 0
        x0, y0 = self.home  # home coordinates
        assert self.env_map[self.barrier, x0, y0] != 1, 'home: %d %d' % (x0, y0)
        avail = 0  # available (to place a barrier on)
        old_b, new_b = 0, 0  # num old and new barriers
        # First remove a few barriers (each with prob change_rate).
        for i in range(self.size):
            for j in range(self.size):
                if self.env_map[self.barrier, i, j]:
                    old_b += 1
                    # remove barrier
                    # Use self.np_random to be consistent..
                    # if np.random.uniform(0, 1) < self.change_rate:
                    if self.np_random.uniform(0, 1) < self.change_rate:
                        env_map_np[self.barrier, i, j] = 0
                        avail += 1
                        removed += 1
                    else:
                        env_map_np[self.barrier, i, j] = 1
                        new_b += 1
                else:  # no barrier
                    if self.env_map[self.food, i, j]:  # not on food!
                        env_map_np[self.food, i, j] = 1
                        num_food += 1
                    elif i != x0 or j != y0:
                        avail += 1  # available

        # To place new barriers: current rate of barriers (after
        # removing a few).
        for_barriers = self.size * self.size - num_food
        # current or old rate..
        # old_rate = 1.0 * old_b / for_barriers
        # existing rate
        current_rate = 1.0 * new_b / for_barriers

        # If desired/target barrier density is below current rate no
        # need to add more.
        if self.prop <= current_rate:
            return env_map_np

        # the extra new barriers needed
        # needed = self.prop * for_barriers - old_b
        needed = self.prop * for_barriers - new_b

        # The new rate  ...
        # rate = 1.0 * removed / avail # another way of computing the rate
        rate = 1.0 * needed / avail
        adds = 0
        for i in range(self.size):
            for j in range(self.size):
                if self.env_map[self.barrier, i, j]:
                    continue
                if self.env_map[self.food, i, j]:
                    continue
                # Skip home base (dont place barrier on it!).
                if i == x0 and j == y0:
                    continue
                # NOTE: use np_random!
                if self.np_random.uniform(0, 1) <= rate:
                #if np.random.uniform(0, 1) <= rate:
                    env_map_np[self.barrier, i, j] = 1
                    new_b += 1
                    adds += 1

        assert env_map_np[self.barrier, x0, y0] != 1, 'home: %d %d' % (x0, y0)

        # Does it still have a path to food?  We are not guaranteeing
        # that it does (with high barrier density, this could be a
        # problem...). TODO? Check if there is a path, and if none,
        # regenerate?  The agent may have a deadline too, and if it
        # can't find a path in that budgetted time, we give up (report
        # the budget) and generate a new world (in the experiments).
        rate = 1.0 * new_b / (self.size * self.size - num_food - 1) # new barrier rate or proportion
        if pit:
            print(
                '\n# prop:%.2f old barrier count:%d, new count:%d (removed:%d added:%d) new_prop:%.2f'
                % (self.prop, old_b, new_b, removed, adds, rate)
            )
        return env_map_np
